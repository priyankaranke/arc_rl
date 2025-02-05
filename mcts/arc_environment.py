import numpy as np

class PuzzleState:
    """
    Represents a single puzzle state, i.e. (puzzle_id, demo_input, current_guess, correct_output).
    """
    def __init__(
        self, puzzle_id, demo_input, current_guess, num_colors, correct_output=None
    ):
        self.puzzle_id = puzzle_id
        self.demo_input = demo_input
        self.current_guess = current_guess
        self.correct_output = correct_output
        self.terminal = False
        self.success_bonus_multiplier = 5.0

        self.grid_height = len(self.current_guess)
        self.grid_width = len(self.current_guess[0])
        self.num_colors = num_colors

    def is_terminal(self):
        if self.correct_output is None:
            # Do not terminate if we don't have the correct output
            # Val time stopping is handled by the done head of the model
            return False

        # Mark as terminal if puzzle is exactly matched
        if np.array_equal(self.current_guess, self.correct_output):
            self.terminal = True
        return self.terminal

    def get_reward(self):
        # Combination of binary and hamming
        def binary_reward(state: np.ndarray, target: np.ndarray) -> float:
            return float(np.array_equal(state, target))

        def hamming_distance_reward(state: np.ndarray, target: np.ndarray) -> float:
            if state.shape != target.shape:
                return 0.0
            total_positions = state.size
            differing_positions = np.sum(state != target)
            return 1.0 - (differing_positions / total_positions)

        bin_r = binary_reward(self.current_guess, self.correct_output)
        ham_r = hamming_distance_reward(self.current_guess, self.correct_output)
        return self.success_bonus_multiplier * bin_r + ham_r

    def clone(self):
        # Return a deep copy
        new_guess = np.copy(self.current_guess)
        return PuzzleState(
            puzzle_id=self.puzzle_id,
            demo_input=self.demo_input,
            current_guess=new_guess,
            correct_output=self.correct_output,
            num_colors=self.num_colors,
        )

    def apply_single_cell_action(self, action):
        """
        Applies an action to the current state.
        The action is a tuple: (row_idx, col_idx, color_idx)
        """
        (row_idx, col_idx, color_idx) = action
        self.current_guess[row_idx, col_idx] = color_idx

    def apply_action(self, action):
        """
        Applies an action to the current state.
        The action is a tuple: (row_idx, col_idx, width_idx, height_idx, color_idx)

        We'll interpret it as "fill the rectangle
        [row_idx : row_idx+height_idx, col_idx : col_idx+width_idx]
        with color_idx, clamped if needed."
        """
        (row_idx, col_idx, width_idx, height_idx, color_idx) = action
        actual_width = width_idx + 1
        actual_height = height_idx + 1

        # clamp in case out-of-bounds
        row_idx = max(0, min(self.grid_height - 1, row_idx))
        col_idx = max(0, min(self.grid_width - 1, col_idx))
        row_end = row_idx + max(
            1, actual_height
        )  # 'actual_height' is how tall the rectangle is
        col_end = col_idx + max(
            1, actual_width
        )  # 'actual_width' is how wide the rectangle is

        row_end = min(row_end, self.grid_height)
        col_end = min(col_end, self.grid_width)

        self.current_guess[row_idx:row_end, col_idx:col_end] = color_idx

    def get_action_space(self):
        """
        For large grids, this is big. Useful for debugging.
        """
        action_list = []
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                for w in range(1, self.grid_width + 1):  # or maybe self.grid_width
                    for h in range(1, self.grid_height + 1):
                        for color in range(
                            11
                        ):  # or self.num_colors + 1 (padding color)
                            action_list.append((r, c, w, h, color))
        return action_list
