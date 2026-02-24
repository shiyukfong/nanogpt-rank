"""
Adapted from https://github.com/karpathy/nanoGPT and https://github.com/KellerJordan/modded-nanogpt.
Full definition of a GPT Language Model, with Rotary Embedding added, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from typing import List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from optimizer import Muon, Muon_Rank

class Rotary(torch.nn.Module):
    """
    Adapted from https://github.com/KellerJordan/modded-nanogpt.
    """

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / (2 * config.n_layer)**0.5)

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

@dataclass
class GPTConfig:
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_block: int = 12
    n_embd: int = 768
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        return logits, loss

    # def crop_block_size(self, block_size):
    #     # model surgery to decrease the block size if necessary
    #     # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
    #     # but want to use a smaller block size for some smaller, simpler model
    #     assert block_size <= self.config.block_size
    #     self.config.block_size = block_size
    #     self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
    #     for block in self.transformer.h:
    #         if hasattr(block.attn, 'bias'):
    #             block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
        optimizer_name: str = 'adamw',
        muon_momentum: float = 0.95,
        muon_lr_scale: float = 0.1,
        sgd_momentum: float = 0.9,
        muon_backend: str = 'newtonschulz5',
        muon_backend_steps: int = 5,
        rank_analysis_mode: str = 'x'
    ) -> List[torch.optim.Optimizer]:
        """Return a list of optimizers (no additional lr scaling)."""
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # helper to build AdamW groups with weight decay only on 2D params
        def build_adamw(params):
            decay = [p for p in params if p.dim() >= 2]
            nodecay = [p for p in params if p.dim() < 2]
            print(f"num decayed parameter tensors: {len(decay)}, with {sum(p.numel() for p in decay):,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay)}, with {sum(p.numel() for p in nodecay):,} parameters")
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            print(f"using fused AdamW: {use_fused}")
            return torch.optim.AdamW(
                [
                    {'params': decay, 'weight_decay': weight_decay},
                    {'params': nodecay, 'weight_decay': 0.0},
                ],
                lr=learning_rate,
                betas=betas,
                **extra_args,
            )

        optimizer_name = optimizer_name.lower()

        if optimizer_name == 'adamw':
            optimizer = build_adamw(list(param_dict.values()))
            return [optimizer]

        if optimizer_name == 'sgd':
            decay = [p for p in param_dict.values() if p.dim() >= 2]
            nodecay = [p for p in param_dict.values() if p.dim() < 2]
            print(f"num decayed parameter tensors: {len(decay)}, with {sum(p.numel() for p in decay):,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay)}, with {sum(p.numel() for p in nodecay):,} parameters")
            optimizer = torch.optim.SGD(
                [
                    {'params': decay, 'weight_decay': weight_decay},
                    {'params': nodecay, 'weight_decay': 0.0},
                ],
                lr=learning_rate,
                momentum=sgd_momentum,
            )
            return [optimizer]

        if 'muon' in optimizer_name or 'dion' in optimizer_name:
            if optimizer_name == "muon_attn":
                # Collect c_attn parameters from all attention layers for Muon_mini
                attn_named_params = []
                attn_param_ids = set()
                for i, block in enumerate(self.transformer.h):
                    for name, param in block.attn.c_attn.named_parameters():
                        attn_named_params.append((f"h.{i}.attn.c_attn.{name}", param))
                        attn_param_ids.add(id(param))
                
                # AdamW gets everything else
                adamw_params = [p for n, p in self.named_parameters()
                                if id(p) not in attn_param_ids]
                
                if len(attn_named_params) == 0:
                    raise ValueError("muon_attn selected but no c_attn parameters found.")
                
                adamw_opt = build_adamw(adamw_params)
                muon_opt = Muon_mini(
                    attn_named_params,
                    n_head=self.config.n_head,
                    lr=learning_rate * muon_lr_scale,
                    momentum=muon_momentum,
                    backend=muon_backend,
                    backend_steps=muon_backend_steps,
                )
                return [adamw_opt, muon_opt]
            # Collect ids of transformer block params for Muon without consuming
            # a single-use generator; we'll call `.parameters()` again when
            # constructing the Muon optimizer so we only iterate once for ids
            # and avoid exhausting a generator that we still need to pass on.
            muon_param_ids = {id(p) for p in self.transformer.h.parameters()}

            # AdamW gets everything else (excluding Muon params)
            adamw_params = [p for n, p in self.named_parameters()
                            if id(p) not in muon_param_ids]

            if len(muon_param_ids) == 0:
                raise ValueError("Muon optimizer selected but no 2D block parameters were found.")

            adamw_opt = build_adamw(adamw_params)
            if optimizer_name == 'muon':
                muon_opt = Muon(
                    # pass a fresh parameters() generator to the Muon optimizer
                    self.transformer.h.named_parameters(),
                    lr=learning_rate * muon_lr_scale,
                    momentum=muon_momentum,
                    backend=muon_backend,
                    backend_steps=muon_backend_steps,
                )
            elif optimizer_name == 'muon_rank':
                muon_opt = Muon_Rank(
                    # pass a fresh parameters() generator to the Muon optimizer
                    self.transformer.h.named_modules(),
                    lr=learning_rate * muon_lr_scale,
                    momentum=muon_momentum,
                    backend=muon_backend,
                    backend_steps=5,
                    n_iter=muon_backend_steps,
                    mode=rank_analysis_mode
                )
            else:
                raise ValueError(f"Unknown Muon optimizer variant: {optimizer_name}")
            return [adamw_opt, muon_opt]

        raise ValueError(f"Unknown optimizer_name: {optimizer_name}")

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head
        T = self.transformer.h[0].attn.rotary.seq_len_cached
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        # flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        flops_promised = 362.05e12 # L40S GPU bfloat16 peak flops is 362.05 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
