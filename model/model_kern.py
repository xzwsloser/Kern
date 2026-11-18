import torch
import math
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedConfig

class KernConfig(PreTrainedConfig): 
        model_type = "kern" 
def __init__(self, hidden_size=768, num_layers=8, use_moe=False, **kwargs):
        # model arch
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get('dropout', 0.0)

        # FFN Parameters
        self.ffn_hidden_size = kwargs.get('ffn_hidden_size', math.ceil((self.hidden_size * 8) / (3 * 64)) * 64)
        self.activation      = kwargs.get('activation', 'silu')

        # RMSNorm Parameters
        self.rms_norm_eps = kwargs.get('rms_norm_eps', 1e-6)

        # Attention Parameters
        self.num_attention_heads = kwargs.get('num_attention_heads', 8)
        self.num_kv_heads        = kwargs.get('num_kv_heads', 4)
        self.head_dim            = kwargs.get('head_dim', self.hidden_size // self.num_attention_heads)
        self.is_causal           = kwargs.get('is_causal', True)
        # use flash attention for train and prefill
        self.flash_attention     = kwargs.get('flash_attention', True)

        # RoPE Parameters
        self.rope_base          = kwargs.get('rope_base', 1e6)
        self.use_yarn_scale     = kwargs.get('use_yarn_scale', True)
        self.seq_len            = kwargs.get('seq_len', 32*1024)
        self.yarn_params = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "origin_len": 2048,
        } if self.use_yarn_scale else None
        
        # MoE Parameters
        self.num_experts = kwargs.get('num_experts', 4)
        self.num_experts_topk = kwargs.get('num_experts_topk', 1)
        self.moe_ffn_hidden_size = kwargs.get('moe_ffn_hidden_size', self.ffn_hidden_size)
        self.norm_topk_score = kwargs.get('norm_topk_score', True)
        self.route_aux_loss_coef = kwargs.get('route_aux_loss_coef', 5e-4)

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
    def __init__(self, dim: int, config: KernConfig):
        super().__init__()
        rope_base = config.get('rope_base', 1e6)
        yarn_params = config.get('yarn_params', None)
        seq_len = config.get('seq_len', 32*1024)
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
            origin_len, beta_fast, beta_slow, factor = (
                yarn_params.get('orign_len', 2048),
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

class Attention(nn.Module):
    def __init__(self, config: KernConfig):
        super().__init__()
        self.num_kv_heads =  config.num_attention_heads if config.num_kv_heads is None else config.num_kv_heads
        # n_local means params on per gpu
        self.n_local_heads = config.num_attention_heads
        self.n_query_per_head = self.n_local_heads // self.num_kv_heads
        self.head_dim      = config.head_dim
        self.is_causal     = config.is_causal
        # W_q, W_k, W_k, W_o
        self.q_proj = nn.Linear(config.hidden_size, self.n_local_heads*self.head_dim,bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads*self.head_dim,bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads*self.head_dim,bias=False)
        self.o_proj = nn.Linear(self.n_local_heads*self.head_dim, config.hidden_size,bias=False)

        # q, k norm
        self.q_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.result_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # flash attention
        self.flash_attention = config.flash_attention
    def forward(self, x, position_embedding: nn.Module, kv_cache=None, use_cache=False, attention_mask=None):
        # x -> [batch_size, seq_len, hidden_size]
        batch_size, seq_len, _ = x.shape
        # q -> [batch_size, seq_len, heads*head_dim]
        q = self.q_proj(x)
        # k -> [batch_size, seq_len, kv_heads*head_dim]
        k = self.k_proj(x)
        # v -> [batch_size, seq_len, kv_heads*head_dim]
        v = self.v_proj(x)

        # q -> [batch_size, seq_len, heads, head_dim]
        q = q.view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        # k, v -> [batch_size, seq_len, kv_heads, head_dim]
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Q, K Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q = position_embedding(q)
        k = position_embedding(k)

        # use kv cache
        if use_cache and kv_cache is not None:
            # kv_cache[0] -> k_cache -> [batch_size, seq_len_past, kv_heads, head_dim]
            # kv_cache[1] -> v_cache -> [batch_size, seq_len_past, kv_heads, head_dim]

            # k, v -> [batch_size, seq_len_past+seq_len, kv_heads, head_dim]
            k = torch.cat([kv_cache[0], k], dim=1)
            v = torch.cat([kv_cache[1], v], dim=1)

        kv_cache_updated = (k, v) if use_cache else None

        # q     -> [batch_size, head, seq_len, head_dim]
        # k, v  -> [batch_size, head, seq_len_total, head_dim] 
        q = q.transpose(1, 2)
        k = self.repeat_kv(k, self.n_query_per_head).transpose(1, 2)
        v = self.repeat_kv(v, self.n_query_per_head).transpose(1, 2)

        # flash attention
        if self.flash_attention and (not self.is_causal or kv_cache is None) and (attention_mask is None or torch.full(attention_mask, 1)):
            # output -> [batch_size, head, seq_len, seq_len_total]
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            # attention_score -> [batch_size, head, seq_len, seq_len_total]
            attention_score = q @ k.mT / math.sqrt(self.head_dim)
            if self.is_causal:
                # causal mask
                attention_score[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float('-inf'), device=attention_score.device).triu(1)
            # attention_mask -> [B, seq_len_total] -> tokens need to drop
            if attention_mask is not None:
                attention_score = attention_score.masked_fill(attention_mask[:, None, None, :] == 1, float('-inf'))

            # output -> [batch_size, head, seq_len, head_dim]
            output = self.attn_dropout(F.softmax(attention_score, dim=-1).type_as(q)) @ v

        output = output.transpose(1, 2).view(batch_size, seq_len, -1)
        output = self.result_dropout(self.o_proj(output))

        return output, kv_cache_updated

    def repeat_kv(self, x: torch.Tensor, n_rep: int):
        # x -> [batch_size, seq_len, num_kv_heads, head_dim]
        batch_size, seq_len, num_kv_heads, head_dim = x.shape
        return x.unsqueeze(3).expand((batch_size, seq_len, num_kv_heads, n_rep, head_dim)).reshape(batch_size, seq_len, num_kv_heads*n_rep, head_dim)

# Silu(x) = x \sigma(x) = \frac {x}{1 + e^{-x}}
class Silu(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.silu(x)

# Relu(x) = max(0, x)
class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.relu(x)

# FFN(x) = W_{down} (Silu(W_{up}x) * (W_{gated}x))
class FFN(nn.Module):
    def __init__(self, config: KernConfig, ffn_hidden_size: int = 0):
        super().__init__()
        if ffn_hidden_size <= 0:
            self.ffn_hidden_size = config.ffn_hidden_size
        else:
            self.ffn_hidden_size = ffn_hidden_size
        self.up_proj = nn.Linear(config.hidden_size, self.ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(self.ffn_hidden_size, config.hidden_size, bias=False)
        self.gated     = nn.Linear(config.hidden_size, self.ffn_hidden_size, bias=False)
        if config.activation == 'silu':
            self.act_fn = Silu()
        else:
            self.act_fn = ReLU()
    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)) * self.gated(x)) 


class MoEFFN(nn.Module):
    def __init__(self, config: KernConfig):
        super().__init__()
        self.num_experts = config.num_experts        
        self.num_experts_topk = config.num_experts_topk
        self.ffn_hidden_size = config.moe_ffn_hidden_size
        self.norm_topk_score = config.norm_topk_score
        self.route_aux_loss_coef = config.route_aux_loss_coef

        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([ FFN(config, self.ffn_hidden_size) for _ in range (self.num_experts) ])

    def forward(self, x: torch.Tensor):    
        batch_size, seq_len, _ = x.shape
        # x_flat -> [T, hidden_size]
        x_flat = x.reshape(batch_size*seq_len, -1)
        # scores -> [T, num_experts]
        scores = F.softmax(self.router(x_flat), dim=-1)
        # weights -> [T, topk]
        # token_idx -> [T, topk]
        weights, token_idxs  = torch.topk(scores, self.num_experts_topk, dim=-1, sorted=False)

        if self.norm_topk_score:
            if self.num_experts_topk > 1:
                weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-20)
            else:
                # num_experts_topk = 1
                # top1 -> [T, 1]
                top1,_ = torch.topk(F.softmax(self.router(x_flat.detach()), dim=-1), 1, dim=-1, sorted=False)
                # for forward  ->  weights = 1.0
                # for backward ->  only grad for router
                weights = top1 - top1.detach() + 1.0

        # output -> [batch_size*seq_len, hidden_size]
        output = torch.zeros_like(x_flat)

        for idx, expert in enumerate(self.experts):
            mask = (token_idxs == idx)
            if mask.any():
                # mask -> [T, topk]
                # get all token selected this experts
                token_idx = mask.any(dim=-1).nonzero().flatten()
                # [token_selected, 1]
                token_weight = weights[mask].reshape(-1, 1)
                output.index_add_(0, token_idx, (expert(x_flat[token_idx]) * token_weight)).to(output.dtype)
            elif self.training:
                # if training, there exist expert with no token select
                output[0, 0] += 0 * sum([p.sum() for p in expert.parameters()])

        # aux_loss
        if self.training and self.route_aux_loss_coef > 0:
            # load -> [T, topk, num_experts_topk]
            #      -> [topk, num_experts_topk] 
            load = F.one_hot(token_idxs, self.num_experts_topk).float().mean(dim=0)
            # scores -> [T, num_experts]
            #        -> [num_experts]
            self.aux_loss = (load * scores.mean(dim=0)).sum() * self.num_experts_topk * self.route_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        return output.view(batch_size, seq_len, -1)
