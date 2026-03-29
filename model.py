"""
Full definition of a GPT Language Model with optional modern architecture improvements.

Architectural options (all backwards-compatible via GPTConfig flags):
  - use_rmsnorm:  Replace LayerNorm with RMSNorm (no bias, more stable)
  - use_rope:     Replace absolute positional embeddings with Rotary (RoPE)
  - use_swiglu:   Replace GELU MLP with SwiGLU FFN (better perplexity)
  - use_qk_norm:  Apply RMSNorm to Q/K before attention (training stability)

All flags default to False → vanilla nanoGPT behaviour.
Set all four to True for a modern LLaMA-style transformer with QK-Norm.

References:
  [1] GPT-2: https://github.com/openai/gpt-2
  [2] LLaMA: Touvron et al. 2023 (RMSNorm, RoPE, SwiGLU)
  [3] RoFormer: Su et al. 2021 (Rotary Positional Embedding)
  [4] GLU Variants: Shazeer 2020 (SwiGLU)
  [5] QK-Norm: Team et al. 2024, Gemma 2 (Query-Key Normalization)
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch built-in doesn't support bias=False cleanly)."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (no mean-centring, no bias).

    Used in LLaMA/Gemma/Mistral. Faster than LayerNorm, numerically stable.
    """

    def __init__(self, ndim, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical stability, then cast back
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight).to(x.dtype)


def _make_norm(ndim, bias, use_rmsnorm):
    """Factory: return either RMSNorm or LayerNorm based on config."""
    if use_rmsnorm:
        return RMSNorm(ndim)
    return LayerNorm(ndim, bias)


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin tables.

    Applied inside CausalSelfAttention when config.use_rope=True.
    Buffers (cos_cache, sin_cache) are included in state_dict → resume-safe.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=True)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)         # (T, D)
        self.register_buffer('cos_cache', emb.cos()[None, None, :, :], persistent=True)
        self.register_buffer('sin_cache', emb.sin()[None, None, :, :], persistent=True)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                seq_len: int):
        """Apply rotary embeddings to query and key tensors.

        Args:
            q, k: (B, n_head, T, head_dim)
            seq_len: current sequence length T
        """
        cos = self.cos_cache[:, :, :seq_len, :]  # (1, 1, T, D)
        sin = self.sin_cache[:, :, :seq_len, :]
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional RoPE and QK-Norm.

    When config.use_rope=True:
      - Positional information comes from RoPE applied to Q/K
      - The wpe positional embedding in the parent GPT is disabled
    When config.use_rope=False:
      - Standard learned absolute positional embeddings (via wpe in GPT)

    When config.use_qk_norm=True:
      - RMSNorm is applied to Q and K *per head* before attention score
        computation. This stabilises training by preventing attention logit
        explosion, enabling higher learning rates. Used in Gemma 2, Cohere
        Command-R, and Chameleon (Meta, 2024).
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.use_rope = config.use_rope
        self.use_qk_norm = config.use_qk_norm

        # QKV projection (combined, same as original nanoGPT)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE module (only created when use_rope=True; stored in state_dict).
        # Cache at least 2048 positions so inference with longer prompts or
        # Task 4's block_size=512 never raises an index error.
        if config.use_rope:
            rope_max_len = max(config.block_size, 2048)
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len=rope_max_len)

        # QK-Norm: per-head RMSNorm on Q and K (Gemma 2 / Cohere style)
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Flash Attention (PyTorch >= 2.0)
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # QKV projection
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply QK-Norm (before RoPE, as per Gemma 2 — normalise in head space)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply rotary embeddings to Q and K (if enabled)
        if self.use_rope:
            q, k = self.rope(q, k, T)

        # Attention
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ---------------------------------------------------------------------------
# Feed-Forward Networks
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Standard GPT-2 GELU MLP."""

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward: FFN(x) = (SiLU(gate(x)) ⊙ up(x)) × down.

    Parameter-count: uses 8/3 × n_embd for hidden dim, which gives the same
    total parameter count as the standard 4× GELU MLP:
        MLP:    2 × (n_embd × 4·n_embd)     = 8 × n_embd²
        SwiGLU: 3 × (n_embd × 8/3·n_embd)   = 8 × n_embd²
    For n_embd=384: hidden=1024. For n_embd=768: hidden=2048.
    This keeps ALL configs ≤ 32M parameters (4× would push C and E to ~33.6M).

    State-dict keys: gate_proj, up_proj, down_proj  (different from MLP's c_fc/c_proj)
    """

    def __init__(self, config):
        super().__init__()
        # 8/3 × n_embd gives identical param count to standard 4× GELU MLP.
        # Round up to nearest multiple of 64 for CUDA efficiency.
        hidden = int(8 / 3 * config.n_embd)
        hidden = (hidden + 63) // 64 * 64
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=False)
        self.up_proj   = nn.Linear(config.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Pre-norm transformer block. Norm type and FFN type are config-driven."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = _make_norm(config.n_embd, config.bias, config.use_rmsnorm)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = _make_norm(config.n_embd, config.bias, config.use_rmsnorm)
        self.mlp  = SwiGLUMLP(config) if config.use_swiglu else MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# GPT Config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int   = 1024
    vocab_size: int   = 50304  # GPT-2 vocab padded to nearest mult of 64
    n_layer:    int   = 12
    n_head:     int   = 12
    n_embd:     int   = 768
    dropout:    float = 0.0
    bias:       bool  = True   # LayerNorm / Linear bias

    # ---------- Modern architecture flags (default=False → vanilla nanoGPT) ----------
    use_rmsnorm: bool = False   # RMSNorm instead of LayerNorm (no bias needed)
    use_rope:    bool = False   # Rotary positional embeddings instead of learned wpe
    use_swiglu:  bool = False   # SwiGLU FFN instead of GELU MLP
    use_qk_norm: bool = False   # QK-Norm: RMSNorm on Q/K before attention (Gemma 2)

    # ---------- Training enhancements ----------
    use_gradient_checkpointing: bool = False  # Trade compute for memory


# ---------------------------------------------------------------------------
# Full GPT model
# ---------------------------------------------------------------------------

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Build transformer sub-modules
        # Note: wpe is ONLY created when use_rope=False (absolute positional embeddings)
        transformer_dict = dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = _make_norm(config.n_embd, config.bias, config.use_rmsnorm),
        )
        if not config.use_rope:
            transformer_dict['wpe'] = nn.Embedding(config.block_size, config.n_embd)

        self.transformer = nn.ModuleDict(transformer_dict)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: input token embedding == output LM head projection
        self.transformer.wte.weight = self.lm_head.weight

        # Initialise weights
        self.apply(self._init_weights)
        # Special scaled init for residual stream projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.use_rope:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None,
                label_smoothing: float = 0.0):
        """Forward pass with optional label smoothing.

        Args:
            idx:             (B, T) input token indices
            targets:         (B, T) target token indices (or None for inference)
            label_smoothing: smoothing factor for cross-entropy loss (0.0 = none)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Sequence length {t} > block_size {self.config.block_size}"

        # Token embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)

        if self.config.use_rope:
            # RoPE: positional info is injected inside attention via Q/K rotation
            x = self.transformer.drop(tok_emb)
        else:
            # Absolute positional embeddings
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)

        # Forward through transformer blocks (with optional gradient checkpointing)
        if self.config.use_gradient_checkpointing and self.training:
            for block in self.transformer.h:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False
                )
        else:
            for block in self.transformer.h:
                x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                label_smoothing=label_smoothing
            )
        else:
            # Inference: only forward the last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """Reduce block_size (useful when loading a model trained with a larger context)."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if not self.config.use_rope:
            self.transformer.wpe.weight = nn.Parameter(
                self.transformer.wpe.weight[:block_size]
            )
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """Load GPT-2 pretrained weights. Only works with vanilla config (no modern flags)."""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),   # 124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config_args['bias'] = True
        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys()
                      if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), \
            f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # Weight-decay on 2-D params (matmul weights), not on 1-D (biases, norms)
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay   = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, "
              f"with {num_decay:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, "
              f"with {num_nodecay:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model flops utilisation (MFU) as fraction of A100 bfloat16 peak FLOPS."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token   = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd  = flops_per_token * T
        flops_per_iter    = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved    = flops_per_iter * (1.0 / dt)
        flops_promised    = 312e12  # A100 bfloat16 peak FLOPS
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None,
                 top_p: float = 1.0,
                 repetition_penalty: float = 1.0,
                 stop_token: int = 50256) -> torch.Tensor:
        """Autoregressive generation with temperature, top-k, nucleus (top-p),
        and repetition penalty sampling.

        Args:
            idx:                (B, T) seed token indices
            max_new_tokens:     how many tokens to generate
            temperature:        > 1 = more random, < 1 = more deterministic
            top_k:              keep only the top-k logits (None = no filter)
            top_p:              nucleus sampling probability mass (1.0 = no filter)
            repetition_penalty: penalise already-seen tokens (1.0 = no penalty)
            stop_token:         stop generation immediately when this token id is
                                produced (e.g. 50256 for GPT-2 <|endoftext|>).
                                None = no early stop, run full max_new_tokens.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                            else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Repetition penalty: divide logits of already-generated tokens
            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    prev_tokens = idx[b].unique()
                    # For tokens with positive logits, divide; negative, multiply
                    # This preserves relative ordering (Keskar et al., 2019)
                    pos_mask = logits[b, prev_tokens] > 0
                    logits[b, prev_tokens[pos_mask]] /= repetition_penalty
                    logits[b, prev_tokens[~pos_mask]] *= repetition_penalty

            logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-Inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs_sorted = sorted_logits.softmax(dim=-1)
                cumulative_probs = probs_sorted.cumsum(dim=-1)
                # >= matches standard HuggingFace nucleus sampling:
                # the boundary token whose cumulative prob first meets/exceeds
                # top_p is excluded (keeps the nucleus strictly within top_p mass)
                sorted_remove = (cumulative_probs - probs_sorted) >= top_p
                sorted_logits[sorted_remove] = float('-Inf')
                logits.scatter_(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            # Early stop: halt when stop_token is produced (e.g. EOT)
            if stop_token is not None and (idx_next == stop_token).any():
                break

        return idx
