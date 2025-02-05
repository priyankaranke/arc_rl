import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from model import ParamPolicyValueTransformer
from arc_environment import PuzzleState
from arc_data_loader import get_arc_loader
import random
from arc_naive_policy import compute_naive_policy_and_done
import torch.optim.lr_scheduler as lr_scheduler
import csv


def do_training_epoch(
    net,
    optimizer,
    device,
    scaler,
    global_step,
    writer,
    train_loader,
    max_grad_norm,
    log_interval,
    epoch_index,
    save_interval,
    total_epochs,
):
    """
    Performs one epoch of training on the DataLoader 'train_loader'.
    Incorporates MCTS or naive targets to produce a policy distribution and value target.
    Logs memory usage, throughput, and losses to TensorBoard.
    """

    net.train()
    epoch_loss = 0.0
    epoch_samples = 0
    start_time = time.perf_counter()

    # Temporary while testing each puzzle
    puzzle_id = None

    for batch_idx, batch in enumerate(train_loader):
        # Each 'batch' is a list of puzzle dicts from your custom data loader
        if not batch:
            continue

        # We'll store the training targets (policy distributions & values) for the entire batch
        target_rows = []  # [B, row_dim]
        target_cols = []  # [B, col_dim]
        target_colors = []  # [B, num_colors]
        done_targets = []  # [B]
        puzzle_batch = []  # [B]

        for puzzle_data in batch:
            puzzle_id = puzzle_data["puzzle_id"]
            puzzle_state = PuzzleState(
                puzzle_id=puzzle_data["puzzle_id"],
                demo_input=puzzle_data["demo_input"],
                current_guess=puzzle_data["current_guess"],
                correct_output=puzzle_data["correct_output"],
                num_colors=net.num_colors,
            )
            row_vec, col_vec, color_vec, done_fraction = compute_naive_policy_and_done(
                puzzle_state,
                max_rows=net.max_rows,
                max_cols=net.max_cols,
                device=device,
            )
            target_rows.append(row_vec)
            target_cols.append(col_vec)
            target_colors.append(color_vec)
            done_targets.append(done_fraction)
            puzzle_batch.append(puzzle_data)

        target_rows = torch.stack(target_rows, dim=0)  # [B, row_dim]
        target_cols = torch.stack(target_cols, dim=0)  # [B, col_dim]
        target_colors = torch.stack(target_colors, dim=0)  # [B, num_colors]

        # For done_targets, shape = [B]. We'll convert to a 2-dimensional distribution
        # i.e. done_labels[i,0] = 1 - done_fraction, done_labels[i,1] = done_fraction
        B = len(done_targets)
        done_labels = torch.zeros(B, 2, device=device, dtype=torch.float)
        for i, df in enumerate(done_targets):
            done_labels[i, 0] = 1.0 - df
            done_labels[i, 1] = df

        with autocast(device_type="cuda", enabled=True):
            outputs = net.forward(puzzle_batch)

            row_log_probs = torch.log_softmax(outputs["row_logits"], dim=-1)
            col_log_probs = torch.log_softmax(outputs["col_logits"], dim=-1)
            color_log_probs = torch.log_softmax(outputs["color_logits"], dim=-1)
            done_log_probs = torch.log_softmax(outputs["done_logits"], dim=-1)

            loss_row = -torch.sum(target_rows * row_log_probs, dim=-1).mean()
            loss_col = -torch.sum(target_cols * col_log_probs, dim=-1).mean()
            loss_color = -torch.sum(target_colors * color_log_probs, dim=-1).mean()
            loss_done = -torch.sum(done_labels * done_log_probs, dim=-1).mean()

            loss = loss_row + loss_col + loss_color + loss_done

        # Backprop
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # Log
        global_step += 1
        epoch_loss += loss.item() * B
        epoch_samples += B

        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.perf_counter() - start_time
            samples_per_sec = epoch_samples / elapsed if elapsed > 0 else 0

            writer.add_scalar("train/throughput", samples_per_sec, global_step)

            writer.add_scalar("train/loss_row", loss_row.item(), global_step)
            writer.add_scalar("train/loss_col", loss_col.item(), global_step)
            writer.add_scalar("train/loss_color", loss_color.item(), global_step)
            writer.add_scalar("train/loss_done", loss_done.item(), global_step)
            writer.add_scalar("train/step_loss", loss.item(), global_step)

            print(
                f"[Step {global_step}] loss={loss.item():.4f} "
                f"throughput={samples_per_sec:.2f} samples/s "
            )

    if epoch_samples > 0:
        avg_loss = epoch_loss / epoch_samples
    else:
        avg_loss = 0.0

    writer.add_scalar("train/loss", avg_loss, global_step)
    print(f"[End of epoch {epoch_index}] train loss={avg_loss:.4f}")

    # At the end of training or at desired intervals, save the model
    if (epoch_index + 1) % save_interval == 0 or epoch_index == total_epochs - 1:
        save_model(
            net, optimizer, global_step, epoch_index, save_dir=f"models/{puzzle_id}"
        )

    return global_step


def save_model(model, optimizer, global_step, epoch_index, save_dir="models"):
    """
    Saves the model and optimizer state dictionaries.
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(
        save_dir, f"model_epoch_{epoch_index}_step_{global_step}.pth"
    )
    optimizer_path = os.path.join(
        save_dir, f"optimizer_epoch_{epoch_index}_step_{global_step}.pth"
    )

    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)

    print(f"Model saved to {model_path}")
    print(f"Optimizer saved to {optimizer_path}")


@torch.no_grad()
def do_validation_naive(
    net,
    writer,
    global_step,
    epoch_index,
    val_loader,
    device,
    trace_dir="inference_policy",
):
    os.makedirs(trace_dir, exist_ok=True)
    net.eval()

    total_loss = 0.0
    total_samples = 0
    total_already_correct = 0

    for batch_idx, batch in enumerate(val_loader):
        if not batch:
            continue

        puzzle_batch = []
        puzzle_ids = []
        row_targets_list = []
        col_targets_list = []
        color_targets_list = []
        done_targets_list = []

        # Build puzzle_batch and naive targets
        for puzzle_data in batch:
            puzzle_id = puzzle_data["puzzle_id"]
            puzzle_ids.append(puzzle_id)

            # Construct puzzle state to compute naive distribution
            puzzle_state = PuzzleState(
                puzzle_id=puzzle_data["puzzle_id"],
                demo_input=puzzle_data["demo_input"],
                current_guess=puzzle_data["current_guess"],
                correct_output=puzzle_data["correct_output"],
                num_colors=net.num_colors,
            )
            total_already_correct += puzzle_state.is_terminal()
            row_vec, col_vec, color_vec, done_frac = compute_naive_policy_and_done(
                puzzle_state,
                max_rows=net.max_rows,
                max_cols=net.max_cols,
                device=device,
            )
            # Convert done_frac -> shape=[2]: [p(notDone), p(done)]
            done_vec = torch.tensor([1.0 - done_frac, done_frac], device=device)

            puzzle_batch.append(puzzle_data)
            row_targets_list.append(row_vec)
            col_targets_list.append(col_vec)
            color_targets_list.append(color_vec)
            done_targets_list.append(done_vec)

        B = len(puzzle_batch)
        if B == 0:
            continue

        # Stack naive targets
        row_targets = torch.stack(row_targets_list, dim=0)  # [B, max_rows]
        col_targets = torch.stack(col_targets_list, dim=0)  # [B, max_cols]
        color_targets = torch.stack(color_targets_list, dim=0)  # [B, num_colors]
        done_targets = torch.stack(done_targets_list, dim=0)  # [B, 2]

        # Forward pass
        with autocast(device_type="cuda", enabled=True):
            outputs = net.forward(
                puzzle_batch
            )  # row_logits, col_logits, color_logits, done_logits

            row_log_probs = F.log_softmax(
                outputs["row_logits"], dim=-1
            )  # [B, max_rows]
            col_log_probs = F.log_softmax(
                outputs["col_logits"], dim=-1
            )  # [B, max_cols]
            color_log_probs = F.log_softmax(
                outputs["color_logits"], dim=-1
            )  # [B, num_colors]
            done_log_probs = F.log_softmax(outputs["done_logits"], dim=-1)  # [B, 2]

            # Cross-entropy = - sum(target * log_probs)
            loss_row = -(row_targets * row_log_probs).sum(dim=-1).mean()
            loss_col = -(col_targets * col_log_probs).sum(dim=-1).mean()
            loss_color = -(color_targets * color_log_probs).sum(dim=-1).mean()
            loss_done = -(done_targets * done_log_probs).sum(dim=-1).mean()

            batch_loss = loss_row + loss_col + loss_color + loss_done

        # Accumulate
        total_loss += batch_loss.item() * B
        total_samples += B

        # Only log for the last puzzle in this batch
        # i.e. i = B-1
        i = B - 1
        last_puzzle_id = puzzle_ids[i]

        # Create puzzle-specific directory
        puzzle_dir = os.path.join(trace_dir, last_puzzle_id)
        os.makedirs(puzzle_dir, exist_ok=True)

        # Write CSV in puzzle directory
        csv_path = os.path.join(
            puzzle_dir, f"epoch_{epoch_index}_batch_{batch_idx}.csv"
        )
        with open(csv_path, "w", newline="") as f_out:
            writer_csv = csv.writer(f_out)

            # Single header row
            writer_csv.writerow(
                [
                    "puzzle_id",
                    "demo_input",
                    "current_guess",
                    "correct_output",
                    "row_logits",
                    "row_targets",
                    "col_logits",
                    "col_targets",
                    "color_logits",
                    "color_targets",
                    "done_softmax",
                    "done_targets",
                    "loss_row",
                    "loss_col",
                    "loss_color",
                    "loss_done",
                ]
            )

            # puzzle_data for the last puzzle
            p_data = puzzle_batch[i]
            # Convert Tensors -> Python lists for CSV
            demo_input_list = p_data["demo_input"].cpu().numpy().tolist()
            current_guess_list = p_data["current_guess"].cpu().numpy().tolist()
            correct_output_list = p_data["correct_output"].cpu().numpy().tolist()

            # Full row/col/color/done arrays
            row_logits_full = row_log_probs[i].detach().cpu().numpy().tolist()
            col_logits_full = col_log_probs[i].detach().cpu().numpy().tolist()
            color_logits_full = color_log_probs[i].detach().cpu().numpy().tolist()
            done_softmax_i = torch.softmax(outputs["done_logits"][i], dim=-1)
            done_softmax_list = done_softmax_i.detach().cpu().numpy().tolist()

            row_targets_full = row_targets[i].cpu().numpy().tolist()
            col_targets_full = col_targets[i].cpu().numpy().tolist()
            color_targets_full = color_targets[i].cpu().numpy().tolist()
            done_targets_full = done_targets[i].cpu().numpy().tolist()

            writer_csv.writerow(
                [
                    last_puzzle_id,
                    demo_input_list,
                    current_guess_list,
                    correct_output_list,
                    row_logits_full,
                    row_targets_full,
                    col_logits_full,
                    col_targets_full,
                    color_logits_full,
                    color_targets_full,
                    done_softmax_list,
                    done_targets_full,
                    float(loss_row.item()),
                    float(loss_col.item()),
                    float(loss_color.item()),
                    float(loss_done.item()),
                ]
            )

    # Final average
    if total_samples > 0:
        avg_loss = total_loss / total_samples
    else:
        avg_loss = 0.0

    # Log cross-entropy to TensorBoard
    writer.add_scalar("val/loss_xent", avg_loss, global_step)
    writer.add_scalar(
        "val/total_already_correct_pct",
        total_already_correct / total_samples,
        global_step,
    )
    print(
        f"[Validation Xent] total_samples={total_samples}, loss_xent={avg_loss:.4f}, total_already_correct_pct={total_already_correct / total_samples:.4f}"
    )


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(47)

    # 1) Setup logging
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    base_path = "/home/ubuntu/arc-rl-x64-filesys/data/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aug_files = os.listdir(base_path + "architects_fork/kaggle/temp/aug_demo_analysis/")
    puzzle_id_to_index = {
        os.path.splitext(puzzle_id)[0]: i for i, puzzle_id in enumerate(aug_files)
    }

    # 2) Create model
    net = ParamPolicyValueTransformer(
        hidden_dim=512,
        n_heads=16,
        n_layers=16,
        max_rows=32,
        max_cols=32,
        num_colors=11,  # 10 puzzle colors + 1 padding dummy color
        max_puzzles=512,
        puzzle_embed_dim=256,
        row_embed_dim=128,
        col_embed_dim=128,
        input_color_embed_dim=256,
        guess_color_embed_dim=256,
        puzzle_id_to_index=puzzle_id_to_index,
    ).to(device)

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,} parameters")

    # 3) Create an optimizer
    epochs = 20
    optimizer = optim.AdamW(net.parameters(), lr=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = GradScaler()

    challenges_file_path = os.path.join(
        base_path, "arc-prize-2024/arc-agi_evaluation_challenges.json"
    )
    solutions_file_path = os.path.join(
        base_path, "arc-prize-2024/arc-agi_evaluation_solutions.json"
    )
    submission_file_path = os.path.join(base_path, "submission.json")
    analysis_json_dir_path = os.path.join(base_path, "aug_demo_analysis/")

    # 4) Create DataLoaders
    train_loader = get_arc_loader(
        split="train",
        batch_size=512,
        bucket_size=1024,
        challenges_file_path=challenges_file_path,
        solutions_file_path=solutions_file_path,
        submission_file_path=submission_file_path,
        analysis_json_dir_path=analysis_json_dir_path,
    )
    val_loader = get_arc_loader(
        split="val",
        batch_size=1024,
        bucket_size=4096,
        challenges_file_path=challenges_file_path,
        solutions_file_path=solutions_file_path,
        submission_file_path=submission_file_path,
        analysis_json_dir_path=analysis_json_dir_path,
    )

    # 5) Train
    global_step = 0
    for epoch in tqdm(range(epochs), desc="Epoch"):
        print(f"=== EPOCH {epoch} ===")
        do_validation_naive(
            net=net,
            writer=writer,
            global_step=global_step,
            epoch_index=epoch,
            val_loader=val_loader,
            device=device,
        )
        global_step = do_training_epoch(
            net=net,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            global_step=global_step,
            writer=writer,
            train_loader=train_loader,
            max_grad_norm=1.0,
            log_interval=8,
            epoch_index=epoch,
            save_interval=50,
            total_epochs=epochs,
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current learning rate: {current_lr:.6f}")
    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
