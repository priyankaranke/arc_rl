from huggingface_hub import snapshot_download

# Model after the first round of finetuning on training set
hf_model_id = "paranke/Mistral-NeMo-Minitron-8B-arc-training"
model_local_path = "./models/initial_finetuned_model"

# Adapter from test time finetuning on demonstration pairs of the public evaluation set
hf_adapter_id = "paranke/Mistral-NeMo-Minitron-8B-arc-eval-adapter"
adapter_local_path = "./models/test_time_adapter"

# Download the model and adapter
snapshot_download(
    repo_id=hf_model_id,
    local_dir=model_local_path,
    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.pdf"],
)
print(f"Model downloaded to {model_local_path}.")

snapshot_download(
    repo_id=hf_adapter_id,
    local_dir=adapter_local_path,
    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.pdf"],
)
print(f"Adapter downloaded to {adapter_local_path}.")
