"""Module for reward computation."""

import torch

reward_params = {
    "success_reward": 5.0,
    "wrong_shape_reward": -2.0,
    "invalid_guess": -3.0,
}


def parse_llm_output(text, dec_sep="\n", limit_rows=None):
    """
    Convert the model's text output into a 2D torch.Tensor of ints.
    If row lengths are inconsistent (or any other ValueError arises),
    we return None to indicate parse failure.
    """
    lines = text.split(dec_sep)
    by_rows = []
    for line in lines:
        # Extract digits from each character
        row_digits = [int(ch) for ch in line if ch.isdigit()]
        if row_digits:
            by_rows.append(row_digits)

    if limit_rows and len(by_rows) > limit_rows:
        by_rows = by_rows[:limit_rows]

    # Now try to build a rectangular torch tensor
    try:
        guess = torch.tensor(by_rows, dtype=torch.int32)
    except ValueError:
        # e.g. mismatch in row lengths => can't form a 2D tensor
        return None
    return guess


def tensor_to_text(tensor: torch.Tensor, dec_sep="\n") -> str:
    """Convert a 2D torch.Tensor of ints to a string with rows separated by dec_sep."""
    lines = []
    for row in tensor:
        # Convert each int in the row to a digit character, then join
        row_str = "".join(str(int(elem.item())) for elem in row)
        lines.append(row_str)
    # Join all rows with the desired row separator
    return dec_sep.join(lines) + dec_sep


def is_valid_solution(guess):
    """Check if guess is a valid 2D torch.Tensor solution within 1 to 30 rows and columns."""
    if not isinstance(guess, torch.Tensor):
        return False
    if guess.dim() != 2:
        return False
    rows, cols = guess.shape
    if rows < 1 or rows > 30:
        return False
    if cols < 1 or cols > 30:
        return False
    return True


def environment_reward_fn(completions, correct_outputs, dec_sep="\n"):
    """
    For each LLM completion (string) + corresponding correct_output (torch.Tensor),
    parse the text into a guess (torch.Tensor). If invalid, reward = -5.
    If valid but shape mismatch => reward=0.
    Else compute Hamming ratio, plus +5 bonus if perfect match.

    :param completions: list of strings from the model
    :param correct_outputs: list of torch.Tensor with shape [r, c]
    :return: list of float rewards
    """
    rewards = []
    for text, target in zip(completions, correct_outputs):
        # Parse into torch.Tensor
        guess = parse_llm_output(text, dec_sep=dec_sep)

        # If invalid or None => -5
        if guess is None or not is_valid_solution(guess):
            rewards.append(reward_params["invalid_guess"])
            continue

        if guess.shape != target.shape:
            rewards.append(reward_params["wrong_shape_reward"])
            continue

        # Compute mismatch
        total_positions = guess.numel()
        mismatch_count = (guess != target).sum().item()  # .item() => integer
        hamming_r = 1.0 - (mismatch_count / total_positions)
        # Perfect match => +5 bonus
        if mismatch_count == 0:
            r = hamming_r + reward_params["success_reward"]
        else:
            r = hamming_r

        rewards.append(r)

    return rewards
