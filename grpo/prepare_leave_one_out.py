"""Module for preparing leave-one-out data for ARC puzzles."""

import os
import json
import random
import numpy as np


def rotate90(grid: np.ndarray) -> np.ndarray:
    """Rotate the grid 90 degrees clockwise."""
    return np.rot90(grid, k=-1)  # clockwise 90


def rotate180(grid: np.ndarray) -> np.ndarray:
    """Rotate the grid 180 degrees."""
    return np.rot90(grid, k=2)


def rotate270(grid: np.ndarray) -> np.ndarray:
    """Rotate the grid 270 degrees clockwise."""
    return np.rot90(grid, k=1)  # effectively 270 clockwise


def transpose(grid: np.ndarray) -> np.ndarray:
    """Transpose the grid."""
    return grid.T


def reflect_horizontal(grid: np.ndarray) -> np.ndarray:
    """Reflect the grid horizontally."""
    return np.fliplr(grid)


def reflect_vertical(grid: np.ndarray) -> np.ndarray:
    """Reflect the grid vertically."""
    return np.flipud(grid)


def apply_color_permutation(inp: np.ndarray, perm_string: str) -> np.ndarray:
    """Apply a color permutation to the input array using the given permutation string."""
    permutation = [int(x) for x in perm_string]
    perm_map = np.array(permutation, dtype=inp.dtype)
    return perm_map[inp]


def create_augmentations(
    withheld_input: np.ndarray,
    withheld_output: np.ndarray,
    other_demos: list[dict],
    puzzle_id: str,
    withheld_j: int,
    desired_count: int,
    max_spatial_transforms: int,
) -> dict:
    """Create augmentations for a given puzzle's withheld example and associated demos."""
    spatial_ops = {
        "rot90": rotate90,
        "rot180": rotate180,
        "rot270": rotate270,
        "transpose": transpose,
        "reflectH": reflect_horizontal,
        "reflectV": reflect_vertical,
    }
    key_counts = {}
    results = {}
    base_aug_key = f"{puzzle_id}_{withheld_j}"

    for _ in range(desired_count):
        # Create the augmentation scheme
        # First, random # of spatial transforms
        aug_key = base_aug_key
        augs = []
        num_spatial = random.randint(0, max_spatial_transforms)
        for _ in range(num_spatial):
            op_name = random.choice(list(spatial_ops.keys()))
            aug_key += f".{op_name}"
            augs.append(spatial_ops[op_name])

        # Then, 50% chance to additionally apply color permutation
        if random.random() < 0.5:
            perm = list(range(10))
            random.shuffle(perm)
            perm_string = "".join(str(x) for x in perm)
            aug_key += f".perm{perm_string}"
            augs.append(lambda x: apply_color_permutation(x, perm_string))

        # Now, apply the augmentation scheme to the withheld input and output
        withheld_input_aug = withheld_input.copy()
        withheld_output_aug = withheld_output.copy()
        for aug in augs:
            withheld_input_aug = aug(withheld_input_aug)
            withheld_output_aug = aug(withheld_output_aug)

        # Now, apply the augmentation scheme to the other demos
        aug_other_demos = []
        for other_demo in other_demos:
            other_demo_input = np.array(other_demo["input"], dtype=np.int32)
            other_demo_output = np.array(other_demo["output"], dtype=np.int32)
            for aug in augs:
                other_demo_input = aug(other_demo_input)
                other_demo_output = aug(other_demo_output)
            aug_other_demos.append(
                {
                    "input": other_demo_input.tolist(),
                    "output": other_demo_output.tolist(),
                }
            )

        if aug_key in results:
            key_counts[aug_key] += 1
            aug_key = f"{aug_key}.{key_counts[aug_key]}"
        else:
            key_counts[aug_key] = 0

        results[aug_key] = {
            "withheld_input": withheld_input_aug.tolist(),
            "withheld_output": withheld_output_aug.tolist(),
            "other_demos": aug_other_demos,
        }
    return results


def generate_leave_one_out_data(
    base_key: str,
    output_dir: str,
    challenges_file_path: str,
    augmented_count: int,
    max_spatial_transforms: int,
):
    """Generate leave-one-out data for the specified base key and save it."""
    os.makedirs(output_dir, exist_ok=True)
    allowed_base_keys = [base_key]

    with open(challenges_file_path, "r", encoding="utf-8") as f:
        challenges_data = json.load(f)

    for puzzle_id in challenges_data.keys():
        if puzzle_id not in allowed_base_keys:
            continue

        test_array = challenges_data[puzzle_id]["test"]
        assert len(test_array) > 0, (
            f"Must have test data for puzzle {puzzle_id}, are you using the right file?"
        )

        # The demonstration pairs:
        train_ex = challenges_data[puzzle_id]["train"]
        n_train = len(train_ex)
        assert n_train > 0, (
            f"Must have train data for puzzle {puzzle_id}, are you using the right file?"
        )

        # Each subtask needs to generate augmented_count examples total, so split them across demo pairs
        augmented_count_per_subtask = augmented_count // n_train
        augmented_count_remainder = augmented_count % n_train

        # We'll produce a single output file named puzzleId.json
        out_filename = f"{puzzle_id}.json"
        out_path = os.path.join(output_dir, out_filename)
        puzzle_data_out = {}

        withheld_indices = list(range(n_train))
        for j in withheld_indices:
            augs_per_demo = augmented_count_per_subtask + (1 if j < augmented_count_remainder else 0)

            withheld = train_ex[j]
            withheld_input = np.array(withheld["input"], dtype=np.int32)
            withheld_output = np.array(withheld["output"], dtype=np.int32)

            # Build a 'demo_input' from the other demos.
            other_demos = [train_ex[k] for k in range(n_train) if k != j]

            # Create augmentations. Each withheld scenario produces 'augs_per_demo' entries.
            aug_dict = create_augmentations(
                withheld_input=withheld_input,
                withheld_output=withheld_output,
                other_demos=other_demos,
                puzzle_id=puzzle_id,
                withheld_j=j,
                desired_count=augs_per_demo,
                max_spatial_transforms=max_spatial_transforms,
            )
            puzzle_data_out.update(aug_dict)

        # Save puzzle_data_out
        with open(out_path, "w", encoding="utf-8") as f_out:
            json.dump(puzzle_data_out, f_out)
