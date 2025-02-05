from arc_environment import PuzzleState
import torch

def puzzle_reward(
    mismatch_count_int: int, total_cells: int, success_bonus: float = 5.0
):
    frac_correct = 1.0 - (float(mismatch_count_int) / float(total_cells))
    bin_bonus = success_bonus if mismatch_count_int == 0 else 0.0
    return bin_bonus + frac_correct

def compute_naive_policy_and_done(
    puzzle_state: PuzzleState,
    max_rows: int,
    max_cols: int,
    device: torch.device,
):
    guess = puzzle_state.current_guess.to(device)  # [H,W], int
    correct = puzzle_state.correct_output.to(device)  # [H,W], int
    num_colors = puzzle_state.num_colors - 1
    H, W = guess.shape
    total_cells = H * W

    # If puzzle is already solved, policy should not take action
    if puzzle_state.is_terminal():
        row_vec = torch.zeros(max_rows, device=device, dtype=torch.float)
        col_vec = torch.zeros(max_cols, device=device, dtype=torch.float)
        color_vec = torch.zeros(num_colors, device=device, dtype=torch.float)
        done_fraction = 1.0
        return row_vec, col_vec, color_vec, done_fraction

    local_scores = torch.zeros((H, W, num_colors), dtype=torch.float, device=device)
    mismatch_mask = guess != correct  # [H,W], bool

    # For every mismatched cell, we set local_scores[r,c, correct_colors[r,c]] = +1/total_cells
    # shape [N,2], each row => (r,c)
    mismatch_indices = mismatch_mask.nonzero(as_tuple=False)
    # mismatch_indices[:,0] => row coords, mismatch_indices[:,1] => col coords
    mismatch_r = mismatch_indices[:, 0]
    mismatch_c = mismatch_indices[:, 1]
    # For each mismatched cell, we find the correct color
    mismatch_correct_colors = correct[mismatch_r, mismatch_c]  # shape [N]

    # Now we scatter +1/total_cells into local_scores for each mismatch cell's correct color:
    improvement_value = 1.0 / float(total_cells)
    local_scores[mismatch_r, mismatch_c, mismatch_correct_colors] = improvement_value
    total_improvement = local_scores.sum()
    local_scores /= total_improvement

    row_probs = local_scores.sum(axis=(1, 2))  # shape [H]
    col_probs = local_scores.sum(axis=(0, 2))  # shape [W]
    color_probs = local_scores.sum(axis=(0, 1))  # shape [num_colors]

    row_vec = torch.zeros(max_rows, device=device, dtype=torch.float)
    col_vec = torch.zeros(max_cols, device=device, dtype=torch.float)
    color_vec = torch.zeros(num_colors, device=device, dtype=torch.float)
    row_vec[:H] = row_probs[:H]
    col_vec[:W] = col_probs[:W]
    color_vec[:num_colors] = color_probs[:num_colors]
    done_prob = 0.0

    return row_vec, col_vec, color_vec, done_prob
