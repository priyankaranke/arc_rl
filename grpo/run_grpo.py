"""Module for running GRPO training."""

import os
import time
import json

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

from unsloth import FastLanguageModel
from peft import LoraConfig, get_peft_model
from prompt_generator import convert_batch_to_prompts
from reward import environment_reward_fn

generation_params = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_seq_length": 4096,
    "generation_buffer": 1024,
    "clip_grad_norm": 1.0,
}
training_params = {
    "clip_epsilon": 0.2,
    "beta_kl": 0.02,
    "lr_initial": 1e-4,
    "lr_min": 1e-5,
    "group_size": 32,
    "old_policy_update_interval": 2,
    "high_confidence_score_threshold": 3.0,
    "min_samples_for_high_confidence": 1,
}
lora_params = {
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "lora_bias": "none",
    "lora_use_rslora": True
}
logging_params = {
    "validation_interval": 1,
    "train_log_interval": 1,
}


def compute_ppo_loss(old_policy_chosen_logps, new_policy_chosen_logps,
                     advantage_tok, mask):
    """Compute PPO loss."""
    # PPO Ratio
    ratio_tok = torch.exp(new_policy_chosen_logps - old_policy_chosen_logps)
    unclipped = ratio_tok * advantage_tok
    clipped = torch.clamp(
        ratio_tok,
        1 - training_params["clip_epsilon"],
        1 + training_params["clip_epsilon"]
    ) * advantage_tok
    ppo_loss_per_tok = -torch.min(unclipped, clipped)  # [B, new_len]

    # Normalize
    ppo_loss_seq = ppo_loss_per_tok.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    loss_ppo = ppo_loss_seq.mean()
    return loss_ppo


def training_pass(model, old_policy, tokenizer, optimizer, scaler, sample, device):
    """Perform a single training pass and return logging statistics."""
    start_time = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    group = [sample] * training_params["group_size"]

    # Generate group size completions for each sample using frozen old policy
    prompts = convert_batch_to_prompts(
        group, tokenizer, generation_params["max_seq_length"],
        generation_params["generation_buffer"]
    )
    tokenized_prompts = tokenizer(
        prompts, return_token_type_ids=False, return_tensors="pt",
        padding=True, padding_side="left"
    ).to(device)

    with torch.no_grad(), autocast(device_type=device.type, dtype=torch.bfloat16):
        old_policy_out = old_policy.generate(
            **tokenized_prompts,
            return_dict_in_generate=True,
            output_logits=True,
            min_new_tokens=1,
            max_new_tokens=generation_params["generation_buffer"],
            do_sample=True,
            top_p=generation_params["top_p"],
            temperature=generation_params["temperature"],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    sequences = old_policy_out.sequences.detach().clone()  # [B, prompt_len + new_len]
    old_policy_logits = torch.stack(old_policy_out.logits, dim=1)  # [B, new_len, vocab]
    prompt_len = tokenized_prompts["input_ids"].shape[1]

    # Convert newly generated portion to text for rewards
    completions_text = tokenizer.batch_decode(
        sequences[:, prompt_len:], skip_special_tokens=True
    )

    # Compute reward from environment
    rewards_list = environment_reward_fn(
        completions_text, [sample["withheld_output"]] * len(completions_text)
    )
    rewards = torch.tensor(rewards_list, dtype=torch.float, device=device)

    # Slice out the newly generated portion from sequences: [B, new_len]
    old_gen_tokens = sequences[:, prompt_len:]

    # Build a pad mask (1.0 for real tokens, 0.0 for pad)
    mask = (old_gen_tokens != tokenizer.pad_token_id).float()

    # Gather old-policy logprobs
    old_policy_log_probs = F.log_softmax(old_policy_logits, dim=-1).detach()
    old_policy_chosen_logps = torch.gather(
        old_policy_log_probs, dim=2, index=old_gen_tokens.unsqueeze(2)
    ).squeeze(2)
    old_policy_chosen_logps = old_policy_chosen_logps * mask

    # Now, for the old-policy completions, compute the new-policy logprobs
    with autocast(device_type=device.type, dtype=torch.bfloat16):
        attn_mask = (sequences != tokenizer.pad_token_id).long().to(device)
        new_out = model(input_ids=sequences, attention_mask=attn_mask)
        # Exclude the last token because it is the logits for the next token
        new_policy_logits = new_out.logits[:, prompt_len-1:-1, :]

    new_policy_log_probs = F.log_softmax(new_policy_logits, dim=-1)
    new_policy_chosen_logps = torch.gather(
        new_policy_log_probs, dim=2, index=old_gen_tokens.unsqueeze(2)
    ).squeeze(2)
    new_policy_chosen_logps = new_policy_chosen_logps * mask

    # Advantage ~ reward - baseline
    advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    advantage_tok = advantage.unsqueeze(1).expand_as(new_policy_chosen_logps) * mask

    kl_loss = compute_kl_loss(old_policy_log_probs, new_policy_log_probs, mask)
    ppo_loss = compute_ppo_loss(
        old_policy_chosen_logps, new_policy_chosen_logps, advantage_tok, mask
    )
    loss = ppo_loss + training_params["beta_kl"] * kl_loss

    # Backprop
    scaled_loss = loss
    scaler.scale(scaled_loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(
        model.parameters(), generation_params["clip_grad_norm"]
    )
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    elapsed = time.perf_counter() - start_time
    samples_per_sec = len(group) / elapsed if elapsed > 0 else 0
    return {
        "loss": loss.item(),
        "loss_ppo": ppo_loss.item(),
        "kl": kl_loss,
        "reward_avg": rewards.mean().item(),
        "advantage_max": advantage.max().item(),
        "advantage_min": advantage.min().item(),
        "throughput": samples_per_sec,
    }


def normalize_key(key, log_stats):
    """Normalize a key over log statistics."""
    return sum(info[key] for info in log_stats) / len(log_stats)


def log_train_stats(writer, log_stats, batch_idx):
    """Log training statistics to TensorBoard."""
    loss_avg = normalize_key("loss", log_stats)
    reward_avg = normalize_key("reward_avg", log_stats)
    loss_ppo_avg = normalize_key("loss_ppo", log_stats)
    kl_loss_avg = normalize_key("kl", log_stats)
    advantage_max = normalize_key("advantage_max", log_stats)
    advantage_min = normalize_key("advantage_min", log_stats)
    samples_per_sec = normalize_key("throughput", log_stats)

    print(
        f"Batch {batch_idx} => loss={loss_avg:.4f}, reward={reward_avg:.3f}, "
        f"loss_ppo={loss_ppo_avg:.4f}, kl={kl_loss_avg:.4f}, advantage_max={advantage_max:.3f}, "
        f"advantage_min={advantage_min:.3f}, throughput={samples_per_sec:.2f} samples/s"
    )
    writer.add_scalar("losses/loss", loss_avg, batch_idx)
    writer.add_scalar("losses/loss_ppo", loss_ppo_avg, batch_idx)
    writer.add_scalar("losses/kl", kl_loss_avg, batch_idx)
    writer.add_scalar("losses/reward_avg", reward_avg, batch_idx)
    writer.add_scalar("losses/advantage_max", advantage_max, batch_idx)
    writer.add_scalar("losses/advantage_min", advantage_min, batch_idx)
    writer.add_scalar("losses/throughput", samples_per_sec, batch_idx)


@torch.no_grad()
def validation_loop(model, tokenizer, val_loader, writer, device, batch_idx):
    """Run validation loop and return completions text and rewards."""
    batch_rewards = []
    batch_completions_text = []
    for batch in val_loader:
        batch_prompts = convert_batch_to_prompts(
            batch, tokenizer, generation_params["max_seq_length"],
            generation_params["generation_buffer"]
        )
        tokenized_prompts = tokenizer(
            batch_prompts, return_token_type_ids=False, return_tensors="pt",
            padding=True, padding_side="left"
        ).to(device)

        with autocast(device_type=device.type, dtype=torch.bfloat16):
            out = model.generate(
                **tokenized_prompts,
                min_new_tokens=1,
                max_new_tokens=generation_params["generation_buffer"],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                use_cache=True,
            )
        prompt_len = tokenized_prompts["input_ids"].shape[1]
        completions_text = tokenizer.batch_decode(
            out.sequences[:, prompt_len:], skip_special_tokens=True
        )
        batch_completions_text.extend(completions_text)

        expected_outputs = [i["withheld_output"] for i in batch]
        rewards_list = environment_reward_fn(completions_text, expected_outputs)
        batch_rewards.extend(rewards_list)

    rewards = torch.tensor(batch_rewards, dtype=torch.float, device=device)
    print(f"Eval step {batch_idx} => Reward: {rewards.mean().item():.3f}")
    writer.add_scalar("val/reward_mean", rewards.mean().item(), batch_idx)
    return batch_completions_text, rewards


def compute_kl_loss(old_log_probs, new_log_probs, attention_mask=None):
    """Compute KL divergence loss."""
    old_probs = old_log_probs.exp()  # [B, T, V]
    kl_per_token = (old_probs * (old_log_probs - new_log_probs)).sum(dim=-1)  # [B, T]

    if attention_mask is not None:
        kl_per_token = kl_per_token * attention_mask
        denom = attention_mask.sum()
    else:
        denom = kl_per_token.numel()

    kl_value = kl_per_token.sum() / (denom + 1e-8)
    return kl_value


def model_statistics(model):
    """Print model parameter statistics."""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        numel = param.numel()
        all_params += numel
        if param.requires_grad:
            trainable_params += numel

    print(f"Total parameters: {all_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {100 * trainable_params/all_params:.2f}%")


def create_model_and_tokenizer(model_path, device, mode="inference"):
    """Create model and tokenizer from a pretrained model."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        local_files_only=True,
        dtype=None,
        load_in_4bit=True,
        max_seq_length=generation_params["max_seq_length"],
    )
    if mode == "inference":
        FastLanguageModel.for_inference(model)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    elif mode == "training":
        FastLanguageModel.for_training(model)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    model.to(device)
    # Apply LoRA to the model
    lora_config = LoraConfig(
        r=lora_params["lora_r"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
            "down_proj", "embed_tokens", "lm_head"
        ],
        lora_alpha=lora_params["lora_alpha"],
        lora_dropout=lora_params["lora_dropout"],
        bias=lora_params["lora_bias"],
        use_rslora=lora_params["lora_use_rslora"],
    )
    model = get_peft_model(model, lora_config)
    tokenizer.padding_side = "left"

    return model, tokenizer


def run_training(outfile, log_dir, train_loader, val_loader):
    """Run training loop."""
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set CUDA device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Load model & tokenizer
    merged_model_path = "./models/merged_model"
    print(f"Loading inital finetuned model from {merged_model_path}...")
    model, tokenizer = create_model_and_tokenizer(
        merged_model_path, device, mode="training"
    )
    print("Inital finetuned model loaded successfully.")
    # Copy for inference that will be updated every old_policy_update_interval
    old_policy, _ = create_model_and_tokenizer(
        merged_model_path, device, mode="inference"
    )

    model_statistics(model)

    optimizer = optim.AdamW(model.parameters(), lr=training_params["lr_initial"])
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader),
        eta_min=training_params["lr_min"]
    )
    scaler = GradScaler()
    rewards = []
    high_confidence_threshold_reached_early = False
    train_predictions = {}

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx % training_params["old_policy_update_interval"] == 0 and batch_idx > 0:
            old_policy_state = model.state_dict()
            old_policy.load_state_dict(old_policy_state, strict=False)

        if batch_idx % logging_params["validation_interval"] == 0:
            pred_completions_text, pred_rewards = validation_loop(
                old_policy, tokenizer, val_loader, writer, device, batch_idx
            )
            if batch_idx == 0:
                train_predictions["initial"] = {
                    "completions_text": pred_completions_text,
                    "pred_rewards": pred_rewards.tolist(),
                }
            if high_confidence_threshold_reached_early or batch_idx == len(train_loader) - 1:
                train_predictions["final"] = {
                    "completions_text": pred_completions_text,
                    "pred_rewards": pred_rewards.tolist(),
                }

        if high_confidence_threshold_reached_early:
            break

        log_stats = []
        for sample in batch:
            log_stat = training_pass(model, old_policy, tokenizer, optimizer, scaler, sample, device)
            log_stats.append(log_stat)

            rewards.append(log_stat["reward_avg"])
            reward_count = len(rewards)
            reward_avg = sum(rewards) / reward_count
            if reward_count >= training_params["min_samples_for_high_confidence"] and \
               reward_avg >= training_params["high_confidence_score_threshold"]:
                high_confidence_threshold_reached_early = True
                break

        if batch_idx % logging_params["train_log_interval"] == 0:
            log_train_stats(writer, log_stats, batch_idx)
            log_stats = []

        scheduler.step()

    output = {
        "high_confidence_threshold_reached_early": high_confidence_threshold_reached_early,
        "train_predictions": train_predictions,
    }
    with open(outfile, "w") as f:
        json.dump(output, f, indent=4)

    writer.close()
