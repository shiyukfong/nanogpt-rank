# config for training GPT-2 (124M) with AdamW head + Muon blocks
# launch: torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2_muon.py

# import time 

# main optimizer selection
optimizer_name = 'muon_rank'
learning_rate = 0.0036
muon_lr_scale = 0.1  # Muon uses 0.1x the main lr internally
muon_momentum = 0.95
muon_backend = 'newtonschulz5'
muon_backend_steps = 7500
rank_analysis_mode = 'x' 
# wandb_log = False

# 491,520 tokens per iteration
batch_size = 12 * 2
block_size = 1024
gradient_accumulation_steps = 4 * 5


max_iters = 7500
lr_decay_iters = 7500
warmup_iters = 0 # how many steps to warm up for
warmdown_iters = 4000 # how many steps before cosine decay starts
min_lr = 0.00036
grad_clip = 0.0

# eval stuff
eval_interval = 75
eval_iters = 10
log_interval = 1

# regularization
weight_decay = 0

# data
dataset = 'fineweb-10b'
