# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
# learning_rate = 0.0018
learning_rate = 0.0036
min_lr = 0.00036
# min_lr = 0.00018
# wandb_log = False
# wandb_project = 'fineweb'
# wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M
# 589,824 tokens per iteration
batch_size = 10 * 4
block_size = 1024
gradient_accumulation_steps = 4 * 3

# max_iters = 10000
max_iters = 7500
lr_decay_iters = 7500
warmdown_iters = 4000 # how many steps before cosine decay starts
# lr_decay_iters = 10000
warmup_iters = 200 # how many steps to warm up for
# warmdown_iters = 5000 # how many steps before cosine decay starts

# eval stuff
eval_interval = 125
eval_iters = 100
log_interval = 10

# weight decay
weight_decay = 1e-1
dataset = 'fineweb-10b'
