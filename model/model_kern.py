import torch
import math
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

class KernConfig(PreTrainedConfig): 
    model_type = "kern" 
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        # model arch
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get('dropout', 0.0)

        # vocab Parameters
        self.vocab_size = kwargs.get('vocab_size', 6400)
        self.bos_token_id = kwargs.get('bos_token_id', 1)
        self.end_token_id = kwargs.get('end_token_id', 2)

        # input embedding and output embedding share weights
        self.tie_word_embedding = kwargs.get('tie_word_embeddings', True)

        # FFN Parameters
        self.ffn_hidden_size = kwargs.get('ffn_hidden_size', math.ceil((self.hidden_size * math.pi) / (64)) * 64)
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
def precompute_freq_cis(dim: int, config: KernConfig):
    rope_base = getattr(config, 'rope_base', 1e6)
    yarn_params = getattr(config, 'yarn_params', None)
    seq_len = getattr(config, 'seq_len', 32*1024)
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
            yarn_params.get('origin_len', 2048),
            yarn_params.get('beta_fast', 32.0),
            yarn_params.get('beta_slow', 1.0),
            yarn_params.get('factor', 16))

        # need to expand seq len
        if origin_len < seq_len:
            # low, high is edge dim for beta_slow and beta_fast
            low, high = find_correction_range(dim, rope_base, origin_len, beta_slow, beta_fast)
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
    return freqs_cos, freqs_sin

def find_correction_dim(dim, base, seq_len, beta):
    # for beta, get corrent dim
    # Note: L is the max content size in model train !
    # L/(2 \pi base^{2i/d}) = beta i = [0, 1, ..., dim//2-1]
    # i = (d * log(L/(2 \pi * beta))) / (2 * log(base))
    current_dim = (dim * math.log(seq_len / (2 * math.pi * beta))) / (2 * math.log(base))
    return current_dim

def find_correction_range(dim, base, seq_len, beta_slow, beta_fast):
    # find edge dim for beta_slow and beta_fast
    low = max(math.floor(find_correction_dim(dim, base, seq_len, beta_fast)), 0)
    high = min(math.ceil(find_correction_dim(dim, base, seq_len, beta_slow)), dim//2-1)
    return low, high

def apply_rotate_embed(x: torch.Tensor, freqs_cos, freqs_sin):
    # x -> [batch_size, seq_len, h, dim]
    # not need [x_0, x_1], ..., [x_{d-2}, x_{d-1}]
    # every two dim is fine 
    # e.g: [x_0, x_{d/2}], [x_1, x_{d/2+1}], ..., [x_{d/2-1}, x_{d-1}]
    assert x.dim() == 4 and x.shape[1] == freqs_cos.size(0)
    dim = x.shape[-1]
    rotate_half_x = torch.cat((-x[..., dim//2:], x[..., :dim//2]),dim=-1)
    # freqs_cos -> [seq_len, 1, dim]
    x = (x * freqs_cos.unsqueeze(1) + rotate_half_x * freqs_sin.unsqueeze(1)).to(x.dtype) 
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
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

        # dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.result_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        # flash attention
        self.flash_attention = config.flash_attention
    def forward(self, x, position_embedding, kv_cache=None, use_cache=False, attention_mask=None):
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
        freqs_cos, freqs_sin = position_embedding
        q = apply_rotate_embed(q, freqs_cos, freqs_sin)
        k = apply_rotate_embed(k, freqs_cos, freqs_sin)

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
        if self.flash_attention and (not self.is_causal or kv_cache is None) and (attention_mask is None or torch.all(attention_mask == 1)):
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


# h_t = \sum_{i=1}^N (g_{i,t} FFN_i(u_t))
# g_{i,t} = s_{i,t} if s_{i,t} \in Topk(s_{i,t},k)
#         = 0 else    
# s_t = p_t = softmax(Linear(d, N))
# f_i = \frac {1}{T} number of token select expert i 
# p_i = \frac {1}{T} p(x)
# aux_loss = \alpha N \sum_{i=1}^N f_i p_i
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
                output.index_add_(0, token_idx, (expert(x_flat[token_idx]) * token_weight).to(output.dtype))
            elif self.training:
                # if training, there exist expert with no token select
                output[0, 0] += 0 * sum([p.sum() for p in expert.parameters()])

        # aux_loss
        if self.training and self.route_aux_loss_coef > 0:
            # token_idxs -> [T, topk], with values in [0, num_experts)
            # load -> [num_experts], the fraction of routed slots sent to each expert
            load = F.one_hot(token_idxs, num_classes=self.num_experts).float().mean(dim=0)
            # scores -> [T, num_experts] -> [num_experts]
            # The standard auxiliary loss is N * sum(load_i * importance_i).
            self.aux_loss = (
                load * scores.mean(dim=0)
            ).sum() * self.num_experts * self.route_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        return output.view(batch_size, seq_len, -1)

# KernBlock:
# y_1 = x + Attn(RMS(x))
# y = y_1 + MLP(RMS(y_1))
class KernBlock(nn.Module):
    def __init__(self, layer_id: int, config: KernConfig):
        super().__init__()
        self.layer_id = layer_id
        self.attn_layer = Attention(config)
        self.pre_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp_layer = FFN(config) if not config.use_moe else MoEFFN(config) 
    def forward(self, hidden_states, position_embedding, kv_cache=None, use_cache=False, attention_mask=None):
        raw_state = hidden_states
        hidden_states, present_kv_cache = self.attn_layer(
            self.pre_norm(hidden_states), position_embedding, 
            kv_cache, use_cache, attention_mask
        )
        hidden_states += raw_state
        hidden_states = hidden_states + self.mlp_layer(
            self.post_norm(hidden_states)
        )

        return hidden_states, present_kv_cache

# KernModel:
# embedding -> num_hidden_layers * KernBlock -> norm
class KernModel(nn.Module):
    def __init__(self, config: KernConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embedding_layer = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([KernBlock(l, config) for l in range (self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freq_cis(config.head_dim, config)
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)
    def forward(self, input_ids, attention_mask=None, kv_cache=None, use_cache=False, **kwargs):
        _, seq_len = input_ids.shape         
        if hasattr(kv_cache, 'layers'): 
            kv_cache = None
        kv_cache = kv_cache or [None] * self.num_hidden_layers
        start_pos = kv_cache[0][0].shape[1] if kv_cache[0] is not None else 0
        hidden_state = self.dropout(self.embedding_layer(input_ids))
        # during meta-device init, recompute rope buffers
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freq_cis(self.config.head_dim, self.config)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_state.device), freqs_sin.to(hidden_state.device)

        position_embedding = (self.freqs_cos[start_pos:start_pos+seq_len], self.freqs_sin[start_pos:start_pos+seq_len])
        presents_kv_cache = []
        for layer, past_kv_cache in zip(self.layers, kv_cache):
            hidden_state, present = layer(
                hidden_state,
                position_embedding,
                kv_cache=past_kv_cache,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents_kv_cache.append(present)

        hidden_state = self.norm(hidden_state)
        # whether use moe or not use moe, aux_loss is a tensor
        aux_loss = sum([l.mlp_layer.aux_loss for l in self.layers if isinstance(l.mlp_layer, MoEFFN)], hidden_state.new_zeros(1).squeeze())
        return hidden_state, presents_kv_cache, aux_loss

class KernCausalLM(PreTrainedModel, GenerationMixin):
    config_class = KernConfig
    _tied_weights_keys = {'lm_head.weight': 'model.embedding_layer.weight'}
    def __init__(self, config: KernConfig):
        self.config = config or KernConfig()
        super().__init__(self.config)
        self.model = KernModel(config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embedding:
            self.model.embedding_layer.weight = self.lm_head.weight
        # init module model need
        self.post_init()

    def forward(self, input_ids, attention_mask=None, kv_cache=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_state, past_kv_cache, aux_loss = self.model(
            input_ids,
            attention_mask,
            kv_cache,
            use_cache,
            **kwargs
        )

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # hidden_state -> [batch_size, seq_len, hidden_size]
        # slice_indices -> hidden_state[:, -k:, :]
        # logits -> [batch_size, k, vocab_size]
        logits = self.lm_head(hidden_state[:, slice_indices, :])
        loss = None
        if labels is not None:
            # target: predict next token
            # input, label -> [a_0, a_1, ..., a_n]
            # output -> [a_1, a_2, ..., a_{n+1}]
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            # x -> [batch_size*seq_len, vocab_size]
            # ignore_index -> index not in compute
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100) 
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_kv_cache, hidden_states=hidden_state)
        
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        # input_ids -> [batch_size, seq_len, 1] -> [batch_size*num_return_sequences, seq_len, 1]
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        # attention_mask -> [batch_size, seq_len, 1] -> [batch_size*num_return_sequences, seq_len, 1]
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        # past_kv_cache -> [layer, 2, batch_size, seq_len_past, hidden_size]
        past_key_values = kwargs.pop("past_key_values", None)
        # finished -> [batch_size*num_return_sequences]
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        if streamer:
           streamer.put(input_ids.cpu())

        for _ in range(max_new_tokens):
            # First Time -> Prefill Stage
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            # input_ids -> [batch_size*num_return_sequences, seq_len_past, 1]
            # inputs_ids >> 1
            # outputs(MoeCausalLMOutputWithPast) -> [batch_size*num_return_sequences, seq_len_past, vocab_size]
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            # attention_mask -> [batch_size*num_return_sequences, seq_len_past+1, 1]
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            # logtis -> new token -> [batch_size*num_return_sequences, 1, vocab_size]
            logits = outputs.logits[:, -1, :] / temperature
            # repetition_penalty -> decrease the times token repeat genreate
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    # find token exist in sequence
                    # [unique_token_idx, 1]
                    seen = torch.unique(input_ids[i])
                    # scores -> [unique_token_idx]
                    score = logits[i, seen]
                    logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)

            # select topk token
            if top_k > 0: 
                # topk -> [batch_size*num_return_sequences, 1, vocab_size]
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')

            # top_p
            if top_p < 1.0:
                # sorted_logits (score)  -> [batch_size*num_return_sequences, 1, vocab_size]
                # sorted_indices (index) -> [batch_size*num_return_sequences, 1, vocab_size]
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                # torch.cumsum([A, B, C]) -> [A, A+B, A+B+C]
                # mask (score mask) -> [batch_size*num_return_sequences, 1, vocab_size]
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                # if mask -> [false, false, true, true, true] we should leave first 3 token
                # so mask -> [false, flase, false, true, true]
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                # find element index to drop
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

            # sample next token
            # logits -> [batch_size*num_return_sequences, vocab_size]
            # torch.multinomial: sample using the prob
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)

            if eos_token_id is not None: 
                # if a seq is end, the next token is <EOS>
                # torch.where(condition/mask, x, y)
                next_token = torch.where(
                    finished.unsqueeze(-1), 
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token
                )

            # Decode Stage
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None

            if streamer:
                streamer.put(next_token.cpu())

            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break

        if streamer:
            streamer.end()

        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids

