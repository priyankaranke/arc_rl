import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo

def pad_to_bounding_box(
    grid_a: torch.Tensor,
    grid_b: torch.Tensor,
    pad_val: int = 10,
):
    aH, aW = grid_a.shape
    bH, bW = grid_b.shape
    H = max(aH, bH)
    W = max(aW, bW)

    padded_a = torch.full((H, W), pad_val, dtype=grid_a.dtype, device=grid_a.device)
    padded_b = torch.full((H, W), pad_val, dtype=grid_b.dtype, device=grid_b.device)
    padded_a[0:aH, 0:aW] = grid_a
    padded_b[0:bH, 0:bW] = grid_b
    return padded_a, padded_b

def next_power_of_2(n: int) -> int:
    """
    Returns the smallest power-of-two integer >= n.
    e.g. if n=45, returns 64.
    """
    return 1 << (n - 1).bit_length()

@torch.compile
class ParamPolicyValueTransformer(nn.Module):
    """
    A parametric policy that:
      - Produces param (row, col, color) logits to be used for the next action,
      - Produces a scalar value estimate for the current guess.
    """

    def __init__(
        self,
        # Problem parameters
        max_rows,
        max_cols,
        num_colors,
        max_puzzles,
        # Model parameters
        hidden_dim,
        n_heads,
        n_layers,
        puzzle_embed_dim,
        row_embed_dim,
        col_embed_dim,
        input_color_embed_dim,
        guess_color_embed_dim,
        puzzle_id_to_index,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.num_colors = num_colors
        self.max_puzzles = max_puzzles
        self.puzzle_embed_dim = puzzle_embed_dim
        self.puzzle_id_to_index = puzzle_id_to_index

        # Row/Column/Color embeddings
        self.row_embed = nn.Embedding(max_rows, row_embed_dim)
        self.col_embed = nn.Embedding(max_cols, col_embed_dim)
        self.input_color_embed = nn.Embedding(num_colors, input_color_embed_dim)
        self.guess_color_embed = nn.Embedding(num_colors, guess_color_embed_dim)

        # Puzzle embedding
        self.puzzle_embed = nn.Embedding(max_puzzles, puzzle_embed_dim)

        input_projection_in_dim = (
            row_embed_dim
            + col_embed_dim
            + input_color_embed_dim
            + guess_color_embed_dim
            + puzzle_embed_dim
        )
        assert (
            input_projection_in_dim & (input_projection_in_dim - 1)
        ) == 0, f"Try to use powers of 2 for the input projection dimension: {input_projection_in_dim}"

        self.input_projection = nn.Linear(input_projection_in_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 6) Param-based heads for the shape action
        self.row_head = nn.Linear(hidden_dim, max_rows)
        self.col_head = nn.Linear(hidden_dim, max_cols)
        self.color_head = nn.Linear(
            hidden_dim, num_colors - 1
        )  # Should not predict padding color
        self.done_head = nn.Linear(hidden_dim, 2)

    def forward(self, puzzle_batch):
        """
        puzzle_batch: a list of puzzle states, each like:
          {
            'puzzle_id': string,
            'grid_height': int,
            'grid_width': int,
            'demo_input': 2D array of shape [H, W] with color IDs in [0..max_colors-1],
            'current_guess': 2D array of shape [H, W] with color IDs in [0..max_colors-1]
          }

        Passes through a transformer to produce a dict:
          {
            'row_logits':    [batch_size, max_rows],
            'col_logits':    [batch_size, max_cols],
            'width_logits':  [batch_size, max_cols],
            'height_logits': [batch_size, max_rows],
            'color_logits':  [batch_size, max_colors],
            'done_logits':   [batch_size, 2],
            'value':         [batch_size, 1]
          }
        """
        device = next(self.parameters()).device

        batch_cell_features = []
        seq_lengths = []

        # Step 1: Build a cell feature sequence for each puzzle
        for puzzle_data in puzzle_batch:
            puzzle_id = puzzle_data["puzzle_id"]
            demo_input = puzzle_data["demo_input"].to(device)
            guess_grid = puzzle_data["current_guess"].to(device)

            demo_input, guess_grid = pad_to_bounding_box(
                demo_input, guess_grid, pad_val=10
            )
            H, W = demo_input.shape

            if puzzle_id not in self.puzzle_id_to_index:
                raise ValueError(f"Puzzle {puzzle_id} should be in puzzle index map.")
            puzzle_index = self.puzzle_id_to_index[puzzle_id]

            # Create cell level embeddings
            rgrid, cgrid = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij",  # "ij" -> rgrid is shape [H,W], cgrid is shape [H,W]
            )

            row_coords = rgrid.flatten()
            col_coords = cgrid.flatten()
            in_colors = demo_input.view(-1)
            guess_colors = guess_grid.view(-1)

            # Lookup embeddings
            row_emb = self.row_embed(row_coords)
            col_emb = self.col_embed(col_coords)
            in_col_emb = self.input_color_embed(in_colors)
            guess_col_emb = self.guess_color_embed(guess_colors)

            puzzle_index_tensor = torch.tensor(
                [puzzle_index], device=device, dtype=torch.int32
            )
            puzzle_emb = self.puzzle_embed(puzzle_index_tensor)
            puzzle_emb_for_cells = puzzle_emb.repeat(H * W, 1)

            cell_level_features = torch.cat(
                [row_emb, col_emb, in_col_emb, guess_col_emb, puzzle_emb_for_cells],
                dim=1,
            )  # shape [H*W, 128]
            batch_cell_features.append(cell_level_features)
            seq_lengths.append(cell_level_features.size(0))

        padded_seq_len = next_power_of_2(max(seq_lengths))
        padded_sequences = []
        attn_masks = []

        # Step 2: Pad cell sequences to the same length for the batch
        for cell_features, seq_len in zip(batch_cell_features, seq_lengths):
            pad_len = padded_seq_len - seq_len
            if pad_len > 0:
                pad_tensor = torch.zeros(
                    (pad_len, cell_features.shape[1]), device=device
                )
                cell_features = torch.cat([cell_features, pad_tensor], dim=0)

            # shape [padded_seq_len, feat_dim] -> [1, padded_seq_len, feat_dim]
            padded_sequences.append(cell_features.unsqueeze(0))
            # Make an attention mask for this puzzle. True means "ignore"
            mask_puzzle = [False] * seq_len + [True] * pad_len
            attn_masks.append(mask_puzzle)

        # Now we have a list of [padded_seq_len, 1, feature_dim]. Concat along batch dim -> [B, padded_seq_len, feature_dim]
        x = torch.cat(padded_sequences, dim=0)  # shape [B, padded_seq_len, feature_dim]
        # Create key_padding_mask of shape [B, padded_seq_len].
        key_padding_mask = torch.tensor(attn_masks, device=device)

        # Step 3: Project from feature_dim -> hidden_dim
        x = self.input_projection(x)  # [B, padded_seq_len, hidden_dim]

        # Step 4: Pass through the transformer
        x = self.transformer(
            x,
            src_key_padding_mask=key_padding_mask,
        )  # still [B, padded_seq_len, hidden_dim]

        # Step 5: Mean-pool over the seq dimension to get shape [batch_size, hidden_dim]
        x = x.mean(dim=1)  # [B, hidden_dim]

        # Step 6: Produce param-based heads
        row_logits = self.row_head(x)
        col_logits = self.col_head(x)
        color_logits = self.color_head(x)
        done_logits = self.done_head(x)

        return {
            "row_logits": row_logits,
            "col_logits": col_logits,
            "color_logits": color_logits,
            "done_logits": done_logits,
        }
