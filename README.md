## An RL approach to solving ARC prize 2024

### How to run it locally
#### Setup environment
```bash

git clone https://github.com/priyankaranke/arc_rl.git

cd arc_rl/grpo
python -m venv grpo_env
source grpo_env/bin/activate
pip install -r requirements.txt
```

#### Download and merge initial finetuned model and adapter
```python
python model_prep/download_model.py
python model_prep/merge_adapter_into_model.py
```

#### Begin training
```python
python train.py
```

### Understanding the codebase

### Results