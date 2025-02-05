"""Download model and adapter from Hugging Face Hub."""

from huggingface_hub import snapshot_download

# Model after the first round of finetuning on training set
HF_MODEL_ID = "paranke/Mistral-NeMo-Minitron-8B-arc-training"
MODEL_LOCAL_PATH = "./models/initial_finetuned_model"

# Adapter from test time finetuning on demonstration pairs of the public evaluation set
HF_ADAPTER_ID = "paranke/Mistral-NeMo-Minitron-8B-arc-eval-adapter"
ADAPTER_LOCAL_PATH = "./models/test_time_adapter"

# Download the model and adapter
snapshot_download(
    repo_id=HF_MODEL_ID,
    local_dir=MODEL_LOCAL_PATH,
    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.pdf"],
)
print(f"Model downloaded to {MODEL_LOCAL_PATH}.")

snapshot_download(
    repo_id=HF_ADAPTER_ID,
    local_dir=ADAPTER_LOCAL_PATH,
    ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.pdf"],
)
print(f"Adapter downloaded to {ADAPTER_LOCAL_PATH}.")
