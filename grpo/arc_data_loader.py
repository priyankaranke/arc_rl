"""Module for loading ARC dataset and providing bucketed data loaders."""

import os
import glob
import json
import random
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class ARCDataset(Dataset):
    """Dataset for ARC challenges and solutions with leave-one-out augmentations."""

    def __init__(
        self,
        split,
        challenges_file_path,
        solutions_file_path,
        leave_one_out_dir_path,
    ):
        super().__init__()
        self.split = split

        # Load the challenges file (contains both demonstrations and test inputs)
        with open(challenges_file_path, "r", encoding="utf-8") as f:
            self.challenges_data = json.load(f)

        # Load demo analysis files
        self.examples = []
        file_paths = glob.glob(os.path.join(leave_one_out_dir_path, "*.json"))

        for fp in file_paths:
            base_puzzle_id = os.path.basename(fp).split(".")[0]
            if split == "train":
                with open(fp, "r", encoding="utf-8") as f:
                    leave_one_out_data = json.load(f)

                for augmented_key, content in leave_one_out_data.items():
                    withheld_input = content["withheld_input"]  # shape = 2D array
                    withheld_output = content["withheld_output"]  # shape = 2D array
                    other_demos = content["other_demos"]  # list of {input, output}

                    # Convert to torch
                    withheld_input_t = torch.tensor(withheld_input, dtype=torch.int32)
                    withheld_output_t = torch.tensor(withheld_output, dtype=torch.int32)
                    other_demos_t = [
                        (
                            torch.tensor(demo["input"], dtype=torch.int32),
                            torch.tensor(demo["output"], dtype=torch.int32),
                        )
                        for demo in other_demos
                    ]

                    ex = {
                        "puzzle_id": base_puzzle_id,
                        "augmented_key": augmented_key,
                        "withheld_input": withheld_input_t,
                        "withheld_output": withheld_output_t,
                        "other_demos": other_demos_t,
                    }
                    self.examples.append(ex)
            elif split == "val":
                with open(solutions_file_path, "r", encoding="utf-8") as f:
                    self.solutions_data = json.load(f)

                train_examples = self.challenges_data[base_puzzle_id]["train"]
                other_demos_t = [
                    (
                        torch.tensor(demo["input"], dtype=torch.int32),
                        torch.tensor(demo["output"], dtype=torch.int32),
                    )
                    for demo in train_examples
                ]
                test_examples = self.challenges_data[base_puzzle_id]["test"]
                solution_examples = self.solutions_data[base_puzzle_id]
                assert len(test_examples) == len(solution_examples), (
                    f"Mismatch in test/solution counts for puzzle {base_puzzle_id}: "
                    f"{len(test_examples)} vs {len(solution_examples)}"
                )

                # For each test example, get current guess from submission
                for _, (test_ex, solution) in enumerate(zip(test_examples, solution_examples)):
                    test_input = torch.tensor(test_ex["input"], dtype=torch.int32)
                    correct_output = torch.tensor(solution, dtype=torch.int32)
                    ex = {
                        "puzzle_id": base_puzzle_id,
                        "withheld_input": test_input,
                        "withheld_output": correct_output,
                        "other_demos": other_demos_t,
                    }
                    self.examples.append(ex)
            else:
                raise ValueError(f"Invalid split: {split}")

        print(f"Loaded {len(self.examples)} examples for split: {split}")

        random.shuffle(self.examples)
        self.examples = sorted(
            self.examples,
            key=lambda e: self.calculate_example_size(e),
        )

    def calculate_example_size(self, example: dict) -> int:
        """Calculate a size metric for an example based on grid sizes."""
        input_size = example["withheld_input"].numel()
        output_size = example["withheld_output"].numel()
        demos_size = sum(demo[0].numel() + demo[1].numel() for demo in example["other_demos"])
        total_size = input_size + output_size + demos_size
        return total_size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


class BucketedBatchSampler(Sampler[List[int]]):
    """
    Minimizes sorting overhead by doing one-time bucket creation:
      1) The dataset is already sorted by size.
      2) We slice the sorted list into consecutive buckets of size 'bucket_size'.
      3) We shuffle each bucket, then yield mini-batches from that bucket.
    """

    def __init__(
        self,
        dataset: ARCDataset,
        batch_size=8,
        bucket_size=64,
        drop_last=False,
        max_batches=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.drop_last = drop_last
        self.max_batches = max_batches

        self.indices = list(range(len(dataset)))
        # Split the sorted dataset indices into consecutive buckets of size bucket_size
        self.buckets = []
        current = []
        for idx in self.indices:
            current.append(idx)
            if len(current) == bucket_size:
                self.buckets.append(current)
                current = []
        if current and not drop_last:
            self.buckets.append(current)

    def __len__(self):
        total = 0
        for bucket in self.buckets:
            nb = len(bucket) // self.batch_size
            if not self.drop_last and (len(bucket) % self.batch_size != 0):
                nb += 1
            total += nb
        return total

    def __iter__(self):
        random.shuffle(self.buckets)
        for bucket in self.buckets:
            random.shuffle(bucket)
        total_batches = 0
        for bucket in self.buckets:
            start = 0
            while start < len(bucket):
                end = start + self.batch_size
                batch_indices = bucket[start:end]
                if self.drop_last and len(batch_indices) < self.batch_size:
                    break
                yield batch_indices
                start = end
                total_batches += 1
                if self.max_batches is not None and total_batches >= self.max_batches:
                    return


def collate_fn(batch: List[dict]):
    """Collate function to return the batch as a list of dictionaries."""
    return batch


def get_arc_loader(
    split,
    challenges_file_path,
    solutions_file_path,
    leave_one_out_dir_path,
    batch_size,
    bucket_size,
    max_batches=None,
):
    """Get the ARC DataLoader for the specified split."""
    dataset = ARCDataset(
        split=split,
        challenges_file_path=challenges_file_path,
        solutions_file_path=solutions_file_path,
        leave_one_out_dir_path=leave_one_out_dir_path,
    )
    sampler = BucketedBatchSampler(
        dataset,
        batch_size=batch_size,
        bucket_size=bucket_size,
        drop_last=False,
        max_batches=max_batches,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
    return loader
