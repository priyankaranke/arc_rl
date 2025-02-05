import os
import glob
import json
import torch
from typing import List
from torch.utils.data import Dataset, DataLoader, Sampler
import random

class ARCDataset(Dataset):
    """
    Loads puzzles from several sources and packages them into a single dataset.

    For split="train":
    We are concerned with the demo input, the current guess for the demo, and
    the correct output for the demo. So we load the:
    - Demo input: from the appropriate *challenges.json file.
    - Current guess: from the appropriate *analysis.json in the demo analysis directory.
    - Correct output: from the appropriate *analysis.json in the demo analysis directory.

    For split="val":
    We are concerned with the test input, the current guess for the test, and the correct output
    for the test. So we load the:
    - Test input: from the appropriate *challenges.json file.
    - Current guess: from the submission.json file.
    - Correct output: from the appropriate *solutions.json file.
    """

    def __init__(
        self,
        split,
        challenges_file_path,
        solutions_file_path,
        submission_file_path,
        analysis_json_dir_path,
    ):
        super().__init__()
        self.split = split

        # Load the challenges file (contains both demonstrations and test inputs)
        with open(challenges_file_path, "r") as f:
            self.challenges_data = json.load(f)

        # Load demo analysis files
        self.examples = []
        file_paths = glob.glob(os.path.join(analysis_json_dir_path, "*.json"))

        for fp in file_paths:
            base_puzzle_id = os.path.basename(fp).split(".")[0]
            if base_puzzle_id not in self.challenges_data:
                print(f"Missing puzzle {base_puzzle_id} in challenges data")
                continue

            with open(fp, "r") as f:
                demo_data = json.load(f)

            if split == "train":
                for augmented_key in demo_data.keys():
                    demo_examples = demo_data[augmented_key]
                    for demo in demo_examples:
                        demo_input = torch.tensor(demo["demo_input"], dtype=torch.int32)
                        predicted = torch.tensor(demo["predicted"], dtype=torch.int32)
                        solution = torch.tensor(demo["solution"], dtype=torch.int32)

                        ex = {
                            "puzzle_id": base_puzzle_id,
                            "augmented_key": augmented_key,
                            "demo_input": demo_input,
                            "current_guess": predicted,
                            "correct_output": solution,
                        }
                        self.examples.append(ex)

            elif split == "val":
                with open(submission_file_path, "r") as f:
                    self.submission_data = json.load(f)
                with open(solutions_file_path, "r") as f:
                    self.solutions_data = json.load(f)

                test_examples = self.challenges_data[base_puzzle_id]["test"]
                solution_examples = self.solutions_data[base_puzzle_id]
                assert len(test_examples) == len(solution_examples), (
                    f"Mismatch in test/solution counts for puzzle {base_puzzle_id}: "
                    f"{len(test_examples)} vs {len(solution_examples)}"
                )

                if base_puzzle_id not in self.submission_data:
                    print(f"Puzzle {base_puzzle_id} not found in submission data")
                    continue
                sub_attempts = self.submission_data[base_puzzle_id]

                # For each test example, get current guess from submission
                for test_idx, (test_ex, solution) in enumerate(
                    zip(test_examples, solution_examples)
                ):
                    test_input = torch.tensor(test_ex["input"], dtype=torch.int32)
                    correct_output = torch.tensor(solution, dtype=torch.int32)
                    # Arbitrarily take the first attempt as the current guess
                    guess = torch.tensor(
                        sub_attempts[test_idx]["attempt_1"], dtype=torch.int32
                    )
                    if guess.shape == correct_output.shape:
                        ex = {
                            "puzzle_id": base_puzzle_id,
                            "demo_input": test_input,
                            "current_guess": guess,
                            "correct_output": correct_output,
                        }
                    else:
                        # Make a fake guess that is the same shape as the correct output
                        # Start with correct_output, then remove random non 0 cells
                        fake_guess = correct_output.clone()
                        faked_cells = 0
                        while faked_cells < 6:
                            r, c = random.randint(0, fake_guess.shape[0] - 1), random.randint(0, fake_guess.shape[1] - 1)
                            if fake_guess[r, c] != 0:
                                fake_guess[r, c] = 0
                                faked_cells += 1
                        ex = {
                            "puzzle_id": base_puzzle_id,
                            "demo_input": test_input,
                            "current_guess": fake_guess,
                            "correct_output": correct_output,
                        }
                    self.examples.append(ex)
            else:
                raise ValueError(f"Invalid split: {split}")

        # Sort examples by grid size to minimize padding overhead
        random.shuffle(self.examples)
        self.examples = sorted(
            self.examples,
            key=lambda e: e["demo_input"].numel() + e["current_guess"].numel(),
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


class BucketedBatchSampler(Sampler[List[int]]):
    """
    Minimizes sorting overhead by doing one-time bucket creation:
      1) We already sorted the dataset by size (demo_input.size + current_guess.size).
      2) We slice the sorted list into consecutive 'buckets' of size 'bucket_size'.
      3) We shuffle each bucket, then yield mini-batches from that bucket.
    """

    def __init__(
        self, dataset: ARCDataset, batch_size=8, bucket_size=64, drop_last=False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.drop_last = drop_last

        self.indices = list(range(len(dataset)))
        # Split the sorted dataset indices into consecutive 'buckets' of size bucket_size
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
        # The total number of mini-batches
        total = 0
        for bucket in self.buckets:
            nb = len(bucket) // self.batch_size
            if not self.drop_last and (len(bucket) % self.batch_size != 0):
                nb += 1
            total += nb
        return total

    def __iter__(self):
        # Shuffle each bucket before yielding mini-batches
        for bucket in self.buckets:
            random.shuffle(bucket)

        # Yield mini-batches within each shuffled bucket
        for bucket in self.buckets:
            start = 0
            while start < len(bucket):
                end = start + self.batch_size
                batch_indices = bucket[start:end]
                if self.drop_last and len(batch_indices) < self.batch_size:
                    break
                yield batch_indices
                start = end

def collate_fn(batch: List[dict]):
    # Return the batch as a list of dictionaries (one dictionary per example)
    return batch

def get_arc_loader(
    split,
    challenges_file_path,
    solutions_file_path,
    submission_file_path,
    analysis_json_dir_path,
    batch_size,
    bucket_size,
):
    dataset = ARCDataset(
        split=split,
        challenges_file_path=challenges_file_path,
        solutions_file_path=solutions_file_path,
        submission_file_path=submission_file_path,
        analysis_json_dir_path=analysis_json_dir_path,
    )
    sampler = BucketedBatchSampler(
        dataset, batch_size=batch_size, bucket_size=bucket_size, drop_last=False
    )

    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
    return loader
