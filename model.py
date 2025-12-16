"""
Adapted from https://github.com/karpathy/nanoGPT, which is (c) 2022 Andrej Karpathy

Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

Converted to JAX/Flax/Haliax for TPU/XLA training.
"""

import math
import inspect
from dataclasses import dataclass
from typing import Tuple, Optional, List

import jax
import jax.numpy as jnp
from jax import lax
import flax.linen as nn
from flax import struct

import haliax as hax
from haliax import Axis, NamedArray
from haliax.nn import LayerNorm as HaxLayerNorm

from loguru import logger
from utils import plot


# Define axis names for Haliax
@struct.dataclass
class AxesConfig:
    Batch: Axis
    Pos: Axis
    MaxPos: Axis
    Embed: Axis
    Heads: Axis
    HeadDim: Axis
    Vocab: Axis
    Mlp: Axis


class RotaryPosEncoding:
    """RoPE implementation by Róbert Csordás, converted to JAX"""
    def __init__(self, d_model: int, base=10000, seq_dim: int = 1):
        if d_model % 2 != 0:
            raise ValueError("RoPE can only be used with an even number of dimensions")

        self.d_model = d_model
        self.base = base
        self.seq_dim = seq_dim

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (jnp.arange(0, d_model, 2, dtype=jnp.float32) / d_model))
        self.inv_freq = inv_freq

        # Cache for sin/cos
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def rotate_half(self, x: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return jnp.concatenate((-x2, x1), axis=-1)

    def apply_rot(self, x: jnp.ndarray, sinp: jnp.ndarray, cosp: jnp.ndarray,
                  seq_dim: int, offset: int) -> jnp.ndarray:
        sin = lax.dynamic_slice_in_dim(sinp, offset, x.shape[seq_dim], axis=seq_dim)
        cos = lax.dynamic_slice_in_dim(cosp, offset, x.shape[seq_dim], axis=seq_dim)
        return (x * cos) + (self.rotate_half(x) * sin)

    def apply_rotary_pos_emb(self, q: jnp.ndarray, k: jnp.ndarray, sin: jnp.ndarray,
                            cos: jnp.ndarray, seq_dim: int, offset: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.apply_rot(q, sin, cos, seq_dim, offset), self.apply_rot(k, sin, cos, seq_dim, 0)

    def get(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        seq_len = x.shape[self.seq_dim]
        enable_cache = t is None

        if (not enable_cache) or (seq_len > self.seq_len_cached):
            self.seq_len_cached = seq_len
            if t is None:
                t = jnp.arange(x.shape[self.seq_dim], dtype=jnp.float32)

            t = t.astype(self.inv_freq.dtype)

            freqs = jnp.einsum("...i,j->...ij", t, self.inv_freq)
            emb = jnp.concatenate((freqs, freqs), axis=-1)

            tgt_shape = [1] * x.ndim
            tgt_shape[self.seq_dim] = seq_len
            tgt_shape[-1] = x.shape[-1]

            # support batch
            tgt_shape[0] = -1

            cos = jnp.cos(emb).reshape(tgt_shape)
            sin = jnp.sin(emb).reshape(tgt_shape)

            if enable_cache:
                self.cos_cached = cos
                self.sin_cached = sin
            else:
                return sin, cos

        return self.sin_cached, self.cos_cached

    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, pos_offset: int = 0,
                t: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        sin, cos = self.get(k, t)
        return self.apply_rotary_pos_emb(q, k, sin, cos, self.seq_dim, pos_offset)


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. JAX/Flax version"""
    ndim: int
    bias: bool

    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', nn.initializers.ones, (self.ndim,))
        bias = self.param('bias', nn.initializers.zeros, (self.ndim,)) if self.bias else None

        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + 1e-5)

        if bias is not None:
            return weight * x_norm + bias
        else:
            return weight * x_norm


class CausalSelfAttention(nn.Module):
    config: any

    def setup(self):
        assert self.config.n_embd % self.config.n_head == 0

        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd
        self.dropout_rate = self.config.dropout

        # Initialize RoPE
        self.rope = RotaryPosEncoding(self.config.n_embd // self.config.n_head, seq_dim=-2)

        # Cache for causal masks
        self.cached_causal_mask = {}

    @nn.compact
    def __call__(self, x, cumulative_scores, token_index, padding_mask=None, deterministic=False):
        B, T, C = x.shape

        # QKV projection
        c_attn_weight = self.param('c_attn_weight',
                                   nn.initializers.normal(stddev=0.02),
                                   (C, 3 * C))
        c_attn_bias = self.param('c_attn_bias',
                                nn.initializers.zeros,
                                (3 * C,)) if self.config.bias else None

        qkv = x @ c_attn_weight
        if c_attn_bias is not None:
            qkv = qkv + c_attn_bias

        q, k, v = jnp.split(qkv, 3, axis=2)

        # Reshape for multi-head attention
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)  # (B, nh, T, hs)

        # Apply RoPE with fractional rotations based on forks
        # Compute token counts for fractional rotations
        block_size = self.config.block_size
        token_counts = jnp.zeros((*token_index.shape[:-1], block_size), dtype=token_index.dtype)
        token_counts = token_counts.at[..., token_index].add(1)

        partial_rotations = jnp.cumsum(
            jnp.take_along_axis(1 / (token_counts + 1e-10), token_index, axis=-1),
            axis=-1
        )
        q, k = self.rope(q, k, t=partial_rotations)

        # Fork-specific: use last channel for cumulative scores
        if "fork" in self.config.plan:
            q = q.at[:, :, :, -1].set(jnp.ones_like(q[:, :, :, -1]))
            k = k.at[:, :, :, -1].set(jnp.repeat(cumulative_scores[:, None, :], k.shape[1], axis=1))

        # Build causal attention mask
        mask_key = (q.shape[-2], k.shape[-2])
        if mask_key not in self.cached_causal_mask:
            causal_mask = jnp.tril(jnp.ones((q.shape[-2], k.shape[-2]), dtype=jnp.bool_))
            causal_mask = jnp.where(causal_mask,
                                   jnp.zeros_like(causal_mask, dtype=jnp.float32),
                                   jnp.full_like(causal_mask, float("-inf"), dtype=jnp.float32))
            self.cached_causal_mask[mask_key] = causal_mask

        causal_mask = self.cached_causal_mask[mask_key]
        causal_mask = jnp.repeat(causal_mask[None, :, :], B, axis=0)
        mask = causal_mask

        # Add padding mask if necessary
        if padding_mask is not None:
            padding_mask = jnp.take_along_axis(padding_mask, token_index, axis=1)
            padding_mask_fl = padding_mask.astype(jnp.float32)
            padding_mask_outer = padding_mask_fl[:, :, None] @ padding_mask_fl[:, None, :]
            padding_mask_additive = jnp.where(
                padding_mask_outer.astype(jnp.bool_),
                jnp.zeros_like(padding_mask_outer, dtype=jnp.float32),
                jnp.full_like(padding_mask_outer, float("-inf"), dtype=jnp.float32)
            )
            mask = mask + padding_mask_additive

        # Attenuate v values with cumulative scores
        v = jnp.einsum("bnlh,bl->bnlh", v, jnp.exp(cumulative_scores))

        # Scaled dot-product attention
        mask_4d = jnp.repeat(mask[:, None, :, :], self.n_head, axis=1)

        # Manual attention computation
        scale = 1.0 / jnp.sqrt(q.shape[-1])
        attn_weights = jnp.einsum("bnqd,bnkd->bnqk", q, k) * scale
        attn_weights = attn_weights + mask_4d
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        if not deterministic:
            attn_weights = nn.Dropout(rate=self.dropout_rate)(attn_weights, deterministic=False)

        y = jnp.einsum("bnqk,bnkd->bnqd", attn_weights, v)

        # Re-assemble all head outputs
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)

        # Output projection
        c_proj_weight = self.param('c_proj_weight',
                                   nn.initializers.normal(stddev=0.02),
                                   (C, C))
        c_proj_bias = self.param('c_proj_bias',
                                nn.initializers.zeros,
                                (C,)) if self.config.bias else None

        y = y @ c_proj_weight
        if c_proj_bias is not None:
            y = y + c_proj_bias

        if not deterministic:
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=False)

        return y


class MLP(nn.Module):
    config: any

    @nn.compact
    def __call__(self, x, deterministic=False):
        C = self.config.n_embd

        # First linear layer
        c_fc_weight = self.param('c_fc_weight',
                                nn.initializers.normal(stddev=0.02),
                                (C, 4 * C))
        c_fc_bias = self.param('c_fc_bias',
                              nn.initializers.zeros,
                              (4 * C,)) if self.config.bias else None

        x = x @ c_fc_weight
        if c_fc_bias is not None:
            x = x + c_fc_bias
        x = jax.nn.gelu(x)

        # Second linear layer
        c_proj_weight = self.param('c_proj_weight',
                                   nn.initializers.normal(stddev=0.02),
                                   (4 * C, C))
        c_proj_bias = self.param('c_proj_bias',
                                nn.initializers.zeros,
                                (C,)) if self.config.bias else None

        x = x @ c_proj_weight
        if c_proj_bias is not None:
            x = x + c_proj_bias

        if not deterministic:
            x = nn.Dropout(rate=self.config.dropout)(x, deterministic=False)

        return x


class Block(nn.Module):
    config: any

    def setup(self):
        self.ln_1 = LayerNorm(self.config.n_embd, bias=self.config.bias)
        self.attn = CausalSelfAttention(self.config)
        self.ln_2 = LayerNorm(self.config.n_embd, bias=self.config.bias)
        self.mlp = MLP(self.config)

    def __call__(self, x, cumulative_scores, token_index, padding_mask=None,
                layer_num=None, deterministic=False):
        exponentiated_scores = jnp.exp(cumulative_scores)
        x = x + jnp.einsum("bl,blh->blh", exponentiated_scores,
                          self.attn(self.ln_1(x), cumulative_scores, token_index,
                                   padding_mask=padding_mask, deterministic=deterministic))
        x = x + jnp.einsum("bl,blh->blh", exponentiated_scores,
                          self.mlp(self.ln_2(x), deterministic=deterministic))
        return x


class ForkingBlock(nn.Module):
    config: any

    def setup(self):
        self.ln_1 = LayerNorm(self.config.n_embd, bias=self.config.bias)
        self.attn = CausalSelfAttention(self.config)
        self.ln_2 = LayerNorm(self.config.n_embd, bias=self.config.bias)
        self.mlp = MLP(self.config)

        self.forking_ln = LayerNorm(self.config.n_embd, bias=self.config.bias)

    @staticmethod
    def clipped_logsigmoid(x, min_val=-20.0):
        logsigmoid = -jax.nn.softplus(-x)
        return jnp.clip(logsigmoid, a_min=min_val)

    @nn.compact
    def fork(self, x, cumulative_scores, token_index):
        """Top-k forking implementation"""
        batch_size = cumulative_scores.shape[0]

        # Forking score projection
        forking_score_weight = self.param('forking_score_weight',
                                         nn.initializers.normal(stddev=0.02),
                                         (self.config.n_embd, 2))

        # Compute current layer's forking scores
        curr_layer_forking_score = self.forking_ln(x) @ forking_score_weight

        # Update cumulative scores (in log space)
        forking_scores_cum = (
            self.clipped_logsigmoid(curr_layer_forking_score) +
            cumulative_scores[:, :, None]
        ).reshape(batch_size, -1)
        forking_scores_cum_for_topk = forking_scores_cum

        # Copy token index twice (for original and fork)
        forked_token_index = (
            jnp.repeat(token_index[:, :, None], 2, axis=-1)
            .reshape(batch_size, -1)
        )

        # Mark rightmost token of each original token with +inf (always keep)
        rolled = jnp.roll(forked_token_index, -1, axis=-1)
        is_rightmost = rolled != forked_token_index
        forking_scores_cum_for_topk = jnp.where(
            is_rightmost,
            float("+inf"),
            forking_scores_cum_for_topk
        )

        # Perform top-k selection
        k = min(self.config.max_block_size, forking_scores_cum_for_topk.shape[-1])
        top_k_values, top_k_indices = lax.top_k(forking_scores_cum_for_topk, k)
        top_k_indices = jnp.sort(top_k_indices, axis=-1)

        # Gather based on indices that survived
        orig_indices = top_k_indices // 2  # (batch_size, k)
        is_fork = (top_k_indices % 2) == 0  # 0: fork, 1: orig

        # Gather x values
        batch_indices = jnp.arange(batch_size)[:, None]
        x_to_consider = x[batch_indices, orig_indices, :]  # (batch_size, k, n_embd)

        # Add fork embedding
        fork_embedding = self.param('fork_embedding',
                                   lambda rng, shape: jax.random.normal(rng, shape) * (1 / jnp.sqrt(self.config.n_embd)),
                                   (self.config.n_embd,))

        x_to_consider = x_to_consider + (is_fork[:, :, None].astype(x.dtype) * fork_embedding)

        # Gather cumulative scores and token indices
        new_cumulative_scores = jnp.take_along_axis(forking_scores_cum, top_k_indices, axis=-1)
        new_token_indices = jnp.take_along_axis(forked_token_index, orig_indices, axis=-1)

        return x_to_consider, new_cumulative_scores, new_token_indices

    def __call__(self, x, cumulative_scores, token_index, padding_mask=None,
                layer_num=None, deterministic=False):
        # Fork first
        x, cumulative_scores, token_index = self.fork(x, cumulative_scores, token_index)

        if not self.config.compiled:
            # Plotting (host callback for non-jitted mode)
            plot(
                "forking",
                cumulative_scores=cumulative_scores,
                token_index=token_index,
                key=layer_num
            )

        # Then normal block forward
        exponentiated_scores = jnp.exp(cumulative_scores)
        x = x + jnp.einsum("bl,blh->blh", exponentiated_scores,
                          self.attn(self.ln_1(x), cumulative_scores, token_index,
                                   padding_mask=padding_mask, deterministic=deterministic))
        x = x + jnp.einsum("bl,blh->blh", exponentiated_scores,
                          self.mlp(self.ln_2(x), deterministic=deterministic))

        return x, cumulative_scores, token_index


class Thoughtbubbles(nn.Module):
    config: any

    def setup(self):
        assert self.config.vocab_size is not None
        assert self.config.block_size is not None

        # Token embeddings
        self.wte = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.n_embd,
            embedding_init=nn.initializers.normal(stddev=0.02)
        )

        # Transformer blocks
        self.blocks = [
            ForkingBlock(self.config) if layer == "fork" else Block(self.config)
            for layer in self.config.plan
        ]

        # Final layer norm
        self.ln_f = LayerNorm(self.config.n_embd, bias=self.config.bias)

        # Language model head (weight tied with embeddings)
        # Note: In Flax, weight tying is handled differently

    @nn.compact
    def __call__(self, idx, targets=None, padding_mask=None, deterministic=False):
        device = idx.device if hasattr(idx, 'device') else None
        b, t = idx.shape
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )

        # Token embeddings
        tok_emb = self.wte(idx)  # (b, t, n_embd)

        # Dropout on embeddings
        if not deterministic:
            tok_emb = nn.Dropout(rate=self.config.dropout)(tok_emb, deterministic=False)

        x = tok_emb

        # Initialize cumulative scores and token index
        cumulative_scores = jnp.zeros((b, t), dtype=x.dtype)
        token_index = jnp.tile(jnp.arange(t), (b, 1))

        # Transformer blocks
        for indx, (plan, block) in enumerate(zip(self.config.plan, self.blocks)):
            if "fork" in plan:
                x, cumulative_scores, token_index = block(
                    x, cumulative_scores, token_index,
                    padding_mask=padding_mask,
                    layer_num=indx,
                    deterministic=deterministic
                )
            else:
                x = block(
                    x, cumulative_scores, token_index,
                    padding_mask=padding_mask,
                    layer_num=indx,
                    deterministic=deterministic
                )

        # Final layer norm
        x = self.ln_f(x)

        # Language model head
        lm_head_weight = self.param('lm_head_weight',
                                    nn.initializers.normal(stddev=0.02),
                                    (self.config.n_embd, self.config.vocab_size))

        # Compute logits based on sequence length and averaging method
        if x.shape[1] == self.config.block_size:
            logits = x @ lm_head_weight
        else:
            # Apply the selected averaging method
            if self.config.averaging_method == "logit":
                logits = self.logit_average(x @ lm_head_weight, cumulative_scores, token_index)
            elif self.config.averaging_method == "residual":
                logits = self.residual_average(x, cumulative_scores, token_index) @ lm_head_weight
            elif self.config.averaging_method == "rightmost":
                # Take only rightmost token of each original token
                rolled = jnp.roll(token_index, -1, axis=-1)
                is_rightmost = rolled != token_index
                rightmost_x = x[is_rightmost].reshape(b, t, -1)
                logits = rightmost_x @ lm_head_weight
            else:
                raise ValueError(f"Invalid averaging_method: {self.config.averaging_method}")

        # Compute loss if targets provided
        if targets is not None:
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)

            # Mask out ignore index (-1)
            mask = targets_flat != -1
            logits_masked = logits_flat[mask]
            targets_masked = targets_flat[mask]

            loss = -jnp.sum(
                jax.nn.log_softmax(logits_masked, axis=-1) *
                jax.nn.one_hot(targets_masked, self.config.vocab_size)
            ) / jnp.sum(mask)
        else:
            loss = None

        return logits, loss

    def residual_average(self, residuals, cumulative_scores, token_index):
        """Average residuals weighted by cumulative scores"""
        scaled_residuals = residuals * jnp.exp(cumulative_scores)[:, :, None]

        # Scatter-add to original token positions
        summed_residuals = jnp.zeros(
            (residuals.shape[0], self.config.block_size, self.config.n_embd),
            dtype=residuals.dtype
        )

        # Manual scatter-add implementation
        for b in range(residuals.shape[0]):
            for i in range(residuals.shape[1]):
                idx = token_index[b, i]
                summed_residuals = summed_residuals.at[b, idx].add(scaled_residuals[b, i])

        return summed_residuals

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model"""
        # This will be computed after initialization
        return 0  # Placeholder

    def configure_optimizers_adamw(self, weight_decay, learning_rate, betas, device_type):
        """Returns Optax optimizer for AdamW"""
        import optax

        # Create optimizer chain
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=learning_rate, b1=betas[0], b2=betas[1],
                       weight_decay=weight_decay)
        )

        logger.info(f"OPTIMIZER | using AdamW")
        return optimizer

    def configure_optimizers_muon(self, distributed=False):
        """Returns Optax optimizer combining Muon and AdamW"""
        from muon import create_muon_optax

        # This will be implemented in muon.py
        optimizer = create_muon_optax(self.config, distributed=distributed)

        logger.info(f"OPTIMIZER | using Muon {'distributed' if distributed else 'single device'}")
        return optimizer
