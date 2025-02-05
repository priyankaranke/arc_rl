"""Merge LoRA adapter for the test time finetuning into the initial finetuned model."""
import torch
from peft import PeftModel
from unsloth import FastLanguageModel

# 1) Load the initial finetuned model
MODEL_PATH = "./models/initial_finetuned_model"
print("Loading initial finetuned model from", MODEL_PATH)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    local_files_only=True,
    load_in_4bit=False,
    dtype=torch.torch.float32,
    max_seq_length=8192,  # Use the max_seq_length from the base model config.
)

# 2) Load the LoRA adapter on top
ADAPTER_PATH = "./models/test_time_adapter"
print("Loading LoRA adapter from", ADAPTER_PATH)
peft_model = PeftModel.from_pretrained(model, ADAPTER_PATH)

# 3) Merge the LoRA weights into the base model
print("Merging LoRA weights into the base model...")
peft_model = peft_model.merge_and_unload()

# 4) Save the merged model
MERGED_OUTPUT_PATH = "./models/merged_model"
print("Saving merged model to", MERGED_OUTPUT_PATH)
peft_model.save_pretrained(MERGED_OUTPUT_PATH)
tokenizer.save_pretrained(MERGED_OUTPUT_PATH)

print("Done merging and saving!")
