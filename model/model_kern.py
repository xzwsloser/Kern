import torch
import math
from torch import cuda, nn

# RMSNorm
# x \in R^{B \times d}
# y = \frac {x}{\sqrt {\frac {1}{d} \sum_i x_i^2 + \epsilon}} * \gamma
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.dim:
            raise ValueError(f"RMSNorm input must be vector with {self.dim} dims, but got {x.shape[-1]}")
        dtype = x.dtype
        x = x.float()
        norm = x / (torch.sqrt(x.pow(2).mean(dim=-1,keepdim=True) + self.eps))
        y = (self.weight * norm).to(dtype)
        return y

# LayerNorm
# x \in R^{B \times d}
# y = \frac {x - E[x]}{\sqrt {Var[x] + \epsilon}} * \gamma + \beta
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.dim: 
            raise ValueError(f"LayerNorm input must be vector with {self.dim} dims, but got {x.shape[-1]}")
        dtype = x.dtype
        x = x.float()
        norm = (x - x.mean(dim=-1,keepdim=True)) / (torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps))
        y = (self.weight * norm + self.bias).to(dtype)
        return y

# RoPE
# \theta_t = [t base^{-2i/d}], i = [0, 2, ..., dim-2]
# RoPE(x) = [x_0, x_1,  ... x_{d-2},  x_{d-1}]*[cos(m\theta_0), cos(m\theta_0), ..., cos(m\theta_{d/2-1}, cos(m\theta_{d/2-1})] +
#           [-x_1, x_0, ..., -x_{d-1}, x_{d-2}]*[sin(m\theta_0), sin(m\theta_0), ..., sin(m\theta_{d/2-1}, sin(m\theta_{d/2-1})]
class RoPE(nn.Module):
    freqs_cos: torch.Tensor
    freqs_sin: torch.Tensor
    def __init__(self, dim: int, rope_base=1e6, yarn_params: dict = {}):
        super().__init__()
        # compute freqs
        # freqs = [ base^{-2i/d} ], i = [0, 2, ..., dim-2]
        freqs = 1.0 / torch.pow(rope_base, torch.arange(0, dim, 2)[:dim//2] / dim)
        if yarn_params is not None:
            # orign_len: seq len use in model train
            # max_seq_len: current seq len or seq len expected to expand to
            # beta_fast: max rotatioins for low freqs dim, > beta_fast, freqs keep
            # beat_slow: min rotatioins for high freqs dim, < beta_slow, freqs / factor 
            #            beta_slow < beta < beta_fast, freqs keep linear scale
            # factor: max_seq_len / origin_len
            origin_len, seq_len, beta_fast, beta_slow, factor = (
                yarn_params.get('orign_len', 2048),
                yarn_params.get('seq_len', 32*1024),
                yarn_params.get('beta_fast', 32.0),
                yarn_params.get('beta_slow', 1.0),
                yarn_params.get('factor', 16))

            # need to expand seq len
            if origin_len < seq_len:
                # low, high is edge dim for beta_slow and beta_fast
                low, high = self._find_correction_range(dim, rope_base, origin_len, beta_slow, beta_fast)
                # linear scale factor
                # linear_factor = (i - low) / (high - low + eps)
                linear_func = (torch.arange(dim//2,device=freqs.device) - low).float() / (high - low + 0.001)
                # ramp factor:
                #   for linear_factor < 0 -> 0
                #   for linear_factor > 1 -> 1
                #   for linear_factor \in [0, 1] -> keep
                ramp_func = torch.clamp(linear_func, 0, 1)
                # if linear_factor is 1 -> rotations few(close to beta_slow)  -> high freqs -> div factor 
                # if linear_factor is 0 -> rotations lot(close to beta fast)  -> low freqs -> keep
                freqs = (1 - ramp_func + ramp_func/factor) * freqs

        # t refer to seq len [1, 2, 3, ..., seq_len-1]
        t = torch.arange(seq_len)        
        # freqs = [ t* base^{-2i/d} ], i = [0, 2, ..., dim-2]
        freqs = torch.outer(t, freqs)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        freqs_cos = torch.cat([cos, cos], dim=-1)
        freqs_sin = torch.cat([sin, sin], dim=-1)
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

    def _find_correction_dim(self, dim, base, seq_len, beta):
        # for beta, get corrent dim
        # Note: L is the max content size in model train !
        # L/(2 \pi base^{2i/d}) = beta i = [0, 1, ..., dim//2-1]
        # i = (d * log(L/(2 \pi * beta))) / (2 * log(base))
        current_dim = (dim * math.log(seq_len / (2 * math.pi * beta))) / (2 * math.log(base))
        return current_dim

    def _find_correction_range(self, dim, base, seq_len, beta_slow, beta_fast):
        # find edge dim for beta_slow and beta_fast
        low = max(math.floor(self._find_correction_dim(dim, base, seq_len, beta_fast)), 0)
        high = min(math.ceil(self._find_correction_dim(dim, base, seq_len, beta_slow)), dim//2-1)
        return low, high

    def forward(self, x: torch.Tensor):
        # x -> [batch_size, seq_len, h, dim]
        # not need [x_0, x_1], ..., [x_{d-2}, x_{d-1}]
        # every two dim is fine 
        # e.g: [x_0, x_{d/2}], [x_1, x_{d/2+1}], ..., [x_{d/2-1}, x_{d-1}]
        assert x.dim == 4 and x.shape[1] == self.freqs_cos.size(0)
        dim = x.shape[-1]
        rotate_half_x = torch.cat((-x[..., dim//2:], x[..., :dim//2]),dim=-1)
        # freqs_cos -> [seq_len, 1, dim]
        x = (x * self.freqs_cos.unsqueeze(1) + rotate_half_x * self.freqs_sin.unsqueeze(1)).to(x.dtype) 
        return x

