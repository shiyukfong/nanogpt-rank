torchrun --standalone --nproc_per_node=4 --tee 3 --log_dir ./tmp/torchrun-logs-muon-rank  train.py config/train_gpt2_muon_rank.py > "nanogpt_muon_rank.log" 2>&1 
python plot_ranks.py "nanogpt_muon_rank.log"


# Alternative: run without torchrun for single GPU
# python train.py config/train_gpt2_muon_rank.py > "nanogpt_muon_rank.log" 2>&1
# python plot_ranks.py "nanogpt_muon_rank.log"
