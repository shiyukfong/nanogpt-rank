# saves the fineweb-10B dataset to binary files for training
# downloads pre-tokenized GPT-2 tokens from huggingface to avoid re-tokenization
# usage: python data/fineweb-10b/prepare.py [num_chunks]
#   num_chunks: number of 100M token chunks to download (default: 103 for full 10B)
#   examples:
#     python data/fineweb-10b/prepare.py 9    # download 900M tokens (9 chunks)
#     python data/fineweb-10b/prepare.py      # download full 10B tokens (103 chunks)

import os
import sys
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# fineweb10B shards store a 256-int32 header (1024 bytes) before the uint16 tokens
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4
HEADER_TOKENS = HEADER_BYTES // 2 # 512 tokens per shard
MAGIC = 20240520
VERSION = 1

# Parse command-line arguments
num_chunks = 103  # full fineweb10B. Each chunk is 100M tokens
if len(sys.argv) >= 2:
    num_chunks = int(sys.argv[1])
    print(f"Downloading {num_chunks} chunks (~{num_chunks * 100}M tokens)")
else:
    print(f"Downloading full dataset: {num_chunks} chunks (~10B tokens)")

# Create output directory
data_dir = os.path.dirname(__file__)
cache_dir = os.path.join(data_dir, 'fineweb10B')
os.makedirs(cache_dir, exist_ok=True)


def read_header(path):
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(HEADER_BYTES), dtype=np.int32)
    magic, version, ntok = header[:3]
    if magic != MAGIC:
        raise ValueError(f"Unexpected magic number in {path}: {magic}")
    if version != VERSION:
        raise ValueError(f"Unsupported version {version} in {path}")
    return int(ntok)

def download_file(fname):
    """Download a file from huggingface if not already cached."""
    local_path = os.path.join(cache_dir, fname)
    if not os.path.exists(local_path):
        print(f"Downloading {fname}...")
        hf_hub_download(
            repo_id="kjj0/fineweb10B-gpt2",
            filename=fname,
            repo_type="dataset",
            local_dir=cache_dir
        )
    return local_path

if __name__ == '__main__':
    # Download validation file
    print("Downloading validation data...")
    val_file = download_file("fineweb_val_%06d.bin" % 0)
    
    # Download training chunks
    print(f"Downloading {num_chunks} training chunks...")
    train_files = []
    for i in tqdm(range(1, num_chunks + 1), desc="Downloading train chunks"):
        fname = f"fineweb_train_{i:06d}.bin"
        fpath = download_file(fname)
        train_files.append(fpath)
    
    # Concatenate training chunks into single train.bin
    print("Concatenating training chunks into train.bin...")
    train_output = os.path.join(data_dir, 'train.bin')
    
    # Calculate total size
    total_tokens = 0
    for fpath in train_files:
        ntok = read_header(fpath)
        expected_bytes = HEADER_BYTES + ntok * 2
        actual_bytes = os.path.getsize(fpath)
        if expected_bytes != actual_bytes:
            raise ValueError(f"{fpath} size mismatch: expected {expected_bytes} bytes (ntok={ntok}), got {actual_bytes}")
        total_tokens += ntok
    
    print(f"Total training tokens: {total_tokens:,} (~{total_tokens / 1e9:.2f}B)")
    
    # Create memory-mapped output file
    dtype = np.uint16
    train_arr = np.memmap(train_output, dtype=dtype, mode='w+', shape=(total_tokens,))
    
    # Copy chunks sequentially
    idx = 0
    for fpath in tqdm(train_files, desc="Writing train.bin"):
        ntok = read_header(fpath)
        # skip header (1024 bytes) and only copy the uint16 token payload
        chunk = np.memmap(fpath, dtype=dtype, mode='r', offset=HEADER_BYTES, shape=(ntok,))
        train_arr[idx:idx + ntok] = chunk[:]
        idx += ntok
        del chunk  # Free memory
    
    train_arr.flush()
    print(f"Saved train.bin ({os.path.getsize(train_output) / 1e9:.2f} GB)")
    
    # Copy validation file to val.bin
    print("Copying validation data to val.bin...")
    val_output = os.path.join(data_dir, 'val.bin')
    val_tokens = read_header(val_file)
    val_chunk = np.memmap(val_file, dtype=dtype, mode='r', offset=HEADER_BYTES, shape=(val_tokens,))
    val_arr = np.memmap(val_output, dtype=dtype, mode='w+', shape=(val_tokens,))
    val_arr[:] = val_chunk[:]
    val_arr.flush()
    print(f"Saved val.bin ({os.path.getsize(val_output) / 1e6:.2f} MB, {val_tokens:,} tokens)")
    
    print("\nDataset preparation complete!")
    print(f"  train.bin: {total_tokens:,} tokens (~{total_tokens / 1e9:.2f}B)")
    print(f"  val.bin: {val_tokens:,} tokens (~{val_tokens / 1e6:.2f}M)")
    print(f"\nTo train, run: python train.py config/train_gpt2.py --dataset=fineweb-10b")