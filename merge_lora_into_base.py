import torch
from peft import PeftModel
from unsloth import FastLanguageModel

# 1) Load the base model
base_model_path = "./base_model"
print("Loading base model from", base_model_path)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_path,
    local_files_only=True,
    load_in_4bit=False,
    dtype=torch.torch.float32,
    max_seq_length=8192, # Use the max_seq_length from the base model config.
)

# 2) Load the LoRA adapter on top
lora_model_path = "./finetuned_model"
print("Loading LoRA adapter from", lora_model_path)
peft_model = PeftModel.from_pretrained(model, lora_model_path)

# 3) Merge the LoRA weights into the base model
print("Merging LoRA weights into the base model...")
peft_model = peft_model.merge_and_unload()  

# 4) Save the merged model
merged_output_path = "./merged_model"
print("Saving merged model to", merged_output_path)
peft_model.save_pretrained(merged_output_path)
tokenizer.save_pretrained(merged_output_path)

print("Done merging and saving!")
