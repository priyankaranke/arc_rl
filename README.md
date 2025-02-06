## An RL approach to solving ARC prize 2024
An approach to ARC-AGI using a custom Group Relative Policy Optimization (GRPO) implementation with a few modifications suited to ARC. This approach achieves 51% on a random subset of 100 
evaluation tasks from the ARC-AGI public evaluation dataset.

Please look at the [report](report.pdf) for a detailed explanation of the approach and results, as well as the [results sheet](https://docs.google.com/spreadsheets/d/1HA3Hcw2kc_Hie4cdTndIp9tajs0wxeRuElRkvt5oT08/edit?gid=1108968747#gid=1108968747) for the raw table of results.

### How to run it locally
1. Setup environment
```bash

git clone https://github.com/priyankaranke/arc_rl.git

cd arc_rl/grpo
python -m venv grpo_env
source grpo_env/bin/activate
pip install -r requirements.txt
```

2. Download and merge initial finetuned model and adapter
```python
python model_prep/download_model.py
python model_prep/merge_adapter_into_model.py
```

3. Begin training
```python
python train.py
```

Results will be written to `train_results/` and detailed train logs will be written to `train_logs/` for each puzzle.

### Understanding the codebase
`grpo/` contains the approach covered in the report and used to achieve the tabled results.
- `train.py` is the main entrypoint which orchestrates the training loop.
- `model_prep/` downloads the [initial finetuned model](paranke/Mistral-NeMo-Minitron-8B-arc-training) and merges the [adapter](paranke/Mistral-NeMo-Minitron-8B-arc-eval-adapter) to flatten the initial finetuning steps into a single model and prepare it for the GRPO training.
- `prepare_leave_one_out_data.py` processes raw puzzles into an augmented leave-one-out dataset for training.
- `arc_data_loader.py` contains the code for loading the leave-one-out dataset into a PyTorch DataLoader, handling the batching and bucketing of data.
- `run_grpo.py` contains the heart of the GRPO training loop, as well as performs evaluations on the test inputs.
- `prompt_generator.py` is a utility that handles converting the dataset into prompts for the LLM.
- `reward.py` parses the LLM output and computes the reward function used in the training loop.

`mcts/` is not used in the script, and is more a record of my previous attempts that led to the GRPO approach.
