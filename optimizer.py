import os
import torch

def zeropower_via_svd(G, **kwargs):
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    return U @ Vh


@torch.compile
def zeropower_via_newtonschulz5(G, **kwargs):
    """
    Adapted from https://github.com/KellerJordan/modded-nanogpt.

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    steps = kwargs.get('steps', 10)
    eps = kwargs.get('eps', 1e-7)
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


def zeropower_via_newtonschulz5_batch(G, **kwargs):
    """
    Batched version of the above, with matrices of shape (..., m, n). 
    """
    steps = kwargs.get('steps', 10)
    eps = kwargs.get('eps', 1e-7)
    assert len(G.shape) == 3
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() / (G.norm(dim=(-2, -1), keepdim=True) + eps) # ensure top singular value <= 1
    if G.size(1) > G.size(2):
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(1) > G.size(2):
        X = X.mT
    return X.to(G.dtype)


zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)


class Muon(torch.optim.Optimizer):
    """
    Adapted from https://github.com/KellerJordan/modded-nanogpt.

    Muon: MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. 
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, named_params, lr=3e-4, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        self.param_map = {}
        params = self.get_param_map(named_params)
        super().__init__(params, defaults)
        self.n_step = 0
    
    def get_param_map(self, named_params):
        params = []
        for name, param in named_params:
            params.append(param)
            self.param_map[param] = name
            print(name)
        return params

    def step(self):
        self.n_step += 1
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            for p in group['params']:
                if int(os.environ.get('RANK', 0)) == 0 and False:
                    print(f"optimizing {self.param_map[p]}..", flush=True)
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)

                use_manifold = (group['backend'] == 'manifold' or group['backend'] == 'submanifold')
                if use_manifold:
                    if 'O' not in state:
                        state['O'] = torch.zeros_like(p.data)

                if g.size(0) == 3 * g.size(1): # (nanoGPT specific) split grouped QKV parameters
                    if use_manifold:
                        O_full = state['O']
                        chunks_g = g.split(g.size(1))
                        chunks_O = O_full.split(g.size(1))
                        g = torch.cat([zeropower_backend(g1, O=O1) for g1, O1 in zip(chunks_g, chunks_O)])
                        state['O'] = (g) 
                    else:
                        g = torch.cat([zeropower_backend(g1, steps=group['backend_steps']) for g1 in g.split(g.size(1))])
                    scale = g.size(1)**0.5
                else:
                    if use_manifold:
                        g = zeropower_backend(g, O=state['O'])
                        state['O'] = (g) 
                        g = g[:p.data.size(0), :p.data.size(1)]
                    else:
                        g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1))**0.5 # scale to have update.square().mean() == 1
                p.data.add_(g, alpha=-lr * scale)
        

class Muon_Rank(torch.optim.Optimizer):
    """
    The same as Muon, but with the addition of forward and backward hooks to capture covariance statistics of the inputs and gradients, and print out their effective rank at certain iterations during training. This is intended to be used as a diagnostic tool to understand the rank dynamics of the activations and gradients of a certain model during training. It is not necessarily intended to be used as a general-purpose optimizer, and may have some overhead due to the hooks and rank analysis. The 'mode' argument controls whether to capture the covariance of the input activations ('x') or the output gradients ('g').
    """
    def __init__(self, named_modules, lr=3e-4, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5, n_iter=7500, mode='x'):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        self.module_map = {}
        self.check_rank = True
        self.n_iter = n_iter
        self.iter = 0
        self.mode = mode
        self.rank_iters = [int(self.n_iter * 0.01), int(self.n_iter * 0.25), int(self.n_iter * 0.5), int(self.n_iter * 0.75), int(self.n_iter * 1.0)]
        params = self.register_module_hooks(named_modules)
        super().__init__(params, defaults)

    
    def register_module_hooks(self, named_modules):
        """
        Register forward and backward hooks on modules to capture covariance statistics.
        
        Args:
            named_modules: Iterable of (name, nn.Module) tuples (typically nn.Linear layers)
        
        Returns:
            params: List of parameters extracted from the modules
        """
        params = []
        for name, mod in named_modules:
            if isinstance(mod, torch.nn.Linear):
                params.append(mod.weight)
                self.module_map[mod.weight] = name
                self._register_hooks(mod)
        return params
        
    @staticmethod
    def analyze_rank(cov, threshold=0.95):
        """
        cov: Covariance matrix (2D)
        threshold: Energy threshold to consider 'effective rank'
        """
        cov = cov.to(torch.float32) # ensure numerical stability for SVD
        
        # 1. Compute SVD
        S_squared = torch.linalg.eigvalsh(cov).flip(dims=[0])  # Get eigenvalues (squared singular values) in descending order
        
        # 3. Cumulative energy (how much variance is explained by top k?)
        cumulative_energy = torch.cumsum(S_squared, dim=0) / S_squared.sum()
        
        # 4. Find effective rank (number of components needed to hit threshold)
        effective_rank = torch.searchsorted(cumulative_energy, threshold).item() + 1
        
        return effective_rank
    
    def _update_state(self, param, key, new_val):
        """Update covariance statistics using ema"""
        param_state = self.state[param]
        if key not in param_state:
            param_state[key] = new_val
        else:
            param_state[key].add_(new_val)

    def _register_hooks(self, module):
        """Register hooks to capture input activations and gradient outputs."""
        if self.mode == 'x':
            # Forward hook: capture input activations covariance
            @torch._dynamo.disable
            @torch.no_grad()
            def forward_hook(mod, inputs, output):
                if not mod.training:
                    return
                
                if self.check_rank:
                    x = inputs[0]
                    if x.dim() > 2: 
                        x = x.view(-1, x.size(-1))
                    B = x.size(0)
                    cov_x = (x.t() @ x) / B

                    self._update_state(mod.weight, 'cov_x', cov_x.detach())
            
            # Backward hook: do nothing
            @torch._dynamo.disable
            @torch.no_grad()
            def backward_hook(mod, grad_input, grad_output):
                return
        elif self.mode == 'g':
            # Forward hook: do nothing
            @torch._dynamo.disable
            @torch.no_grad()
            def forward_hook(mod, inputs, output):
                return
            
            # Backward hook: capture output gradients covariance
            @torch._dynamo.disable
            @torch.no_grad()
            def backward_hook(mod, grad_input, grad_output):
                if not mod.training:
                    return
                
                if self.check_rank:
                    g = grad_output[0]
                    if g.dim() > 2: 
                        g = g.view(-1, g.size(-1))
                    B = g.size(0)
                    cov_g = (g.t() @ g) / B

                    self._update_state(mod.weight, 'cov', cov_g.detach())
        else:
            raise ValueError(f"Invalid mode {self.mode} for Muon_Rank. Must be 'x' or 'g'.")
        
        module.register_forward_hook(forward_hook)
        module.register_full_backward_hook(backward_hook)

    def step(self):
        self.iter += 1
        if self.iter in self.rank_iters:
            self.check_rank = True
        else:
            self.check_rank = False
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)

                if g.size(0) == 3 * g.size(1): # (nanoGPT specific) split grouped QKV parameters 
                    g = torch.cat([zeropower_backend(g1, steps=group['backend_steps']) for g1 in g.split(g.size(1))])
                    scale = g.size(1)**0.5
                else:
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1))**0.5 # scale to have update.square().mean() == 1
                p.data.add_(g, alpha=-lr * scale)

                if int(os.environ.get('RANK', 0)) == 0 and 'cov' in state:
                    cov = state['cov']
                    rank = self.analyze_rank(cov)
                    print(f"[{self.module_map[p]}] Effective rank of {self.mode} covariance: {rank}/{cov.size(0)}", flush=True)
                    del state['cov'] # free memory
