"""Module for training on ARC data with leave-one-out data generation and training loop."""
random_seed = 42
torch.manual_seed(random_seed)
random.seed(random_seed)

import os
import random
import shutil
from prepare_leave_one_out import generate_leave_one_out_data
from arc_data_loader import get_arc_loader
import torch
from tqdm import tqdm
from run_grpo import run_training

data_loader_params = {
    "batch_size": 16,
    "bucket_size": 128,
}
augmentation_params = {
    "aug_count": 128,
    "max_spatial_transforms": 6,
}

def cleanup_leave_one_out_data(dir_to_remove):
    """Clean up leave-one-out data directory."""
    if os.path.exists(dir_to_remove):
        shutil.rmtree(dir_to_remove)
        print(f"Removed directory: {dir_to_remove}")
    else:
        print(f"Directory {dir_to_remove} does not exist.")


def main():
    """Main training loop."""
    # Load data
    challenges_file_path = "./data/arc-prize-2024/arc-agi_evaluation_challenges.json"
    solutions_file_path = "./data/arc-prize-2024/arc-agi_evaluation_solutions.json"
    leave_one_out_dir = "./data/leave_one_out_data/"

    # Output paths
    results_dir = "./train_results"
    logs_root_dir = "./train_logs"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_root_dir, exist_ok=True)

    # Select a random 100 base keys. For reproducibility, these were from my run.
    base_keys = [
        "292dd178",
        "1990f7a8",
        "8ee62060",
        "0692e18c",
        "ce039d91",
        "b0722778",
        "a04b2602",
        "2072aba6",
        "d37a1ef5",
        "14754a24",
        "cb227835",
        "da2b0fe3",
        "73c3b0d8",
        "0c9aba6e",
        "e9c9d9a1",
        "84db8fc4",
        "d56f2372",
        "45bbe264",
        "af24b4cc",
        "e0fb7511",
        "e133d23d",
        "ecaa0ec1",
        "27a77e38",
        "0becf7df",
        "ef26cbf6",
        "1e81d6f9",
        "97239e3d",
        "833dafe3",
        "7d18a6fb",
        "59341089",
        "4f537728",
        "d2acf2cb",
        "7e02026e",
        "8cb8642d",
        "5ffb2104",
        "3194b014",
        "ce8d95cc",
        "b7fb29bc",
        "66e6c45b",
        "0e671a1a",
        "0bb8deee",
        "11e1fe23",
        "ba9d41b8",
        "1c56ad9f",
        "66f2d22f",
        "2753e76c",
        "3f23242b",
        "68b67ca3",
        "332efdb3",
        "be03b35f",
        "27f8ce4f",
        "ca8de6ea",
        "5289ad53",
        "c3202e5a",
        "50f325b5",
        "7bb29440",
        "3979b1a8",
        "7953d61e",
        "c658a4bd",
        "456873bc",
        "94414823",
        "8719f442",
        "79369cc6",
        "070dd51e",
        "7c8af763",
        "4852f2fa",
        "281123b4",
        "d5c634a2",
        "62ab2642",
        "b942fd60",
        "ed98d772",
        "506d28a5",
        "60a26a3e",
        "dd2401ed",
        "aa18de87",
        "ccd554ac",
        "0c786b71",
        "e1baa8a4",
        "5d2a5c43",
        "bf699163",
        "642d658d",
        "d47aa2ff",
        "e69241bd",
        "17b80ad2",
        "e345f17b",
        "1d398264",
        "358ba94e",
        "992798f6",
        "477d2879",
        "08573cc6",
        "1da012fc",
        "48f8583b",
        "9110e3c5",
        "c8b7cc0f",
        "1a2e2828",
        "12422b43",
        "5207a7b5",
        "1a6449f1",
        "9c56f360",
        "6f473927",
    ]

    for base_key in tqdm(base_keys, desc="Training loop"):
        # Cleanup stale data for old key
        cleanup_leave_one_out_data(leave_one_out_dir)
        generate_leave_one_out_data(
            base_key,
            leave_one_out_dir,
            challenges_file_path,
            augmentation_params["aug_count"],
            augmentation_params["max_spatial_transforms"],
        )

        # Create new data loaders
        train_loader = get_arc_loader(
            split="train",
            batch_size=data_loader_params["batch_size"],
            bucket_size=data_loader_params["bucket_size"],
            challenges_file_path=challenges_file_path,
            solutions_file_path=solutions_file_path,
            leave_one_out_dir_path=leave_one_out_dir,
        )
        val_loader = get_arc_loader(
            split="val",
            batch_size=data_loader_params["batch_size"],
            bucket_size=data_loader_params["bucket_size"],
            challenges_file_path=challenges_file_path,
            solutions_file_path=solutions_file_path,
            leave_one_out_dir_path=leave_one_out_dir,
        )

        outfile = os.path.join(results_dir, f"{base_key}.json")
        log_dir = os.path.join(logs_root_dir, base_key)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        print(f"Training {base_key}...")
        run_training(outfile, log_dir, train_loader, val_loader)
        print(f"Training complete! Outputs saved to {outfile}")


if __name__ == "__main__":
    main()
