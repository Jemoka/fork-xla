"""
Adapted from https://github.com/karpathy/nanoGPT, which is (c) 2022 Andrej Karpathy

Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
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

from loguru import logger
from utils import plot

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

        return sin, cos

    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, pos_offset: int = 0,
                 t: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        sin, cos = self.get(k, t)
        return self.apply_rotary_pos_emb(q, k, sin, cos, self.seq_dim, pos_offset)


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. JAX/Flax version"""
    ndim: int
    bias: bool
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', nn.initializers.ones, (self.ndim,))
        bias = self.param('bias', nn.initializers.zeros, (self.ndim,)) if self.bias else None

        # Cast to float32 for numerical stability
        x_f32 = x.astype(jnp.float32)
        mean = jnp.mean(x_f32, axis=-1, keepdims=True)
        var = jnp.var(x_f32, axis=-1, keepdims=True)
        x_norm = (x_f32 - mean) / jnp.sqrt(var + 1e-5)

        # Cast back to compute dtype
        x_norm = x_norm.astype(self.dtype)
        weight_cast = weight.astype(self.dtype)

        if bias is not None:
            bias_cast = bias.astype(self.dtype)
            return weight_cast * x_norm + bias_cast
        else:
            return weight_cast * x_norm


ATTN_DTYPE = jnp.bfloat16 


@jax.jit(static_argnums=0)
def causal_bias(max_block_size):
    neg = jnp.array(-jnp.inf, ATTN_DTYPE)
    m = jnp.tril(jnp.ones((max_block_size, max_block_size), dtype=jnp.bool_))
    return jnp.where(m, jnp.array(0, ATTN_DTYPE), neg)[None, None, :, :]

@jax.jit
def key_padding_bias(padding_mask):
    neg = jnp.array(-jnp.inf, ATTN_DTYPE)
    keep = padding_mask.astype(jnp.bool_)[:, None, None, :]  # (B,1,1,T)
    return jnp.where(keep, jnp.array(0, ATTN_DTYPE), neg)

class CausalSelfAttention(nn.Module):
    config: any

    def setup(self):
        assert self.config.n_embd % self.config.n_head == 0

        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd
        self.dropout_rate = self.config.dropout
        self.xavier = jax.nn.initializers.xavier_normal()

        # Initialize RoPE
        self.rope = RotaryPosEncoding(self.config.n_embd // self.config.n_head, seq_dim=-2)

        # QKV projection
        self.c_attn = nn.Dense(
            3 * self.n_embd,
            use_bias=self.config.bias,
            kernel_init=nn.with_partitioning(self.xavier, ('n_embd', 'n_attn')),
            param_dtype=jnp.float32,
            dtype=jnp.bfloat16
        )

        # Output projection
        self.c_proj = nn.Dense(
            self.n_embd,
            use_bias=self.config.bias,
            kernel_init=nn.with_partitioning(self.xavier, ('n_embd_out', 'n_embd')),
            param_dtype=jnp.float32,
            dtype=jnp.bfloat16
        )

    @nn.compact
    def __call__(self, x, cumulative_scores, token_index, padding_mask=None, deterministic=False):
        # sometimes peolpe do float32 attn, idk

        B, T, C = x.shape

        # QKV projection
        qkv = self.c_attn(x)
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

        mask = causal_bias(self.config.max_block_size)
        mask = mask[:,:,:q.shape[-2],:k.shape[-2]]
        
        if padding_mask is not None:
            mask = mask + key_padding_bias(padding_mask)

        # Attenuate v values with cumulative scores
        v = jnp.einsum("bnlh,bl->bnlh", v, jnp.exp(cumulative_scores))

        # Use JAX's optimized attention
        # expects (batch x seq x heads x hidden), so we permute back from (b, nH, T, hs)
        y = jax.nn.dot_product_attention(
            query=q.transpose(0, 2, 1, 3).astype(ATTN_DTYPE),
            key=k.transpose(0, 2, 1, 3).astype(ATTN_DTYPE),
            value=v.transpose(0, 2, 1, 3).astype(ATTN_DTYPE),
            bias=mask
        )

        if not deterministic:
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=False)

        # Re-assemble all head outputs
        y = y.reshape(B, T, C)

        # Output projection
        y = self.c_proj(y)

        if not deterministic:
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=False)

        return y

class MLP(nn.Module):
    config: any

    def setup(self):
        self.xavier = jax.nn.initializers.xavier_normal()

        self.c_fc = nn.Dense(
            4 * self.config.n_embd,
            use_bias=self.config.bias,
            kernel_init=nn.with_partitioning(self.xavier, ('n_embd', 'n_embd_ff')),
            param_dtype=jnp.float32,
            dtype=jnp.bfloat16
        )

        self.c_proj = nn.Dense(
            self.config.n_embd,
            use_bias=self.config.bias,
            kernel_init=nn.with_partitioning(self.xavier, ('n_embd_ff', 'n_embd')),
            param_dtype=jnp.float32,
            dtype=jnp.bfloat16
        )

    @nn.compact
    def __call__(self, x, deterministic=False):
        x = self.c_fc(x)
        x = jax.nn.gelu(x)
        x = self.c_proj(x)

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
        return x, cumulative_scores, token_index


class ForkingBlock(nn.Module):
    config: any

    def setup(self):
        self.xavier = jax.nn.initializers.xavier_normal()

        self.ln_1 = LayerNorm(self.config.n_embd, bias=self.config.bias)
        self.attn = CausalSelfAttention(self.config)
        self.ln_2 = LayerNorm(self.config.n_embd, bias=self.config.bias)
        self.mlp = MLP(self.config)

        self.forking_ln = LayerNorm(self.config.n_embd, bias=self.config.bias)

        # Forking score projection
        self.forking_score = nn.Dense(
            2,
            use_bias=False,
            kernel_init=nn.with_partitioning(self.xavier, ('n_embd', 'n_fork')),
            param_dtype=jnp.float32,
            dtype=jnp.bfloat16
        )

    @staticmethod
    def clipped_logsigmoid(x, min_val=-20.0):
        logsigmoid = -jax.nn.softplus(-x)
        return jnp.clip(logsigmoid, a_min=min_val)

    @nn.compact
    def fork(self, x, cumulative_scores, token_index):
        """Top-k forking implementation"""

        batch_size = cumulative_scores.shape[0] # (batch_size, k)

        # Compute current layer's forking scores
        curr_layer_forking_score = self.forking_score(self.forking_ln(x)) # (batch_size, k, 2)

        # Update cumulative scores (in log space)
        forking_scores_cum = (
            self.clipped_logsigmoid(curr_layer_forking_score) +
            cumulative_scores[:, :, None] # (batch_size, k, 1)
        ).reshape(batch_size, -1) # (batch_size, 2k)
        forking_scores_cum_for_topk = forking_scores_cum

        # Copy token index twice (for original and fork)
        forked_token_index = token_index.repeat(2, axis=-1)

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
        _, top_k_indices = lax.top_k(forking_scores_cum_for_topk, k)
        top_k_indices = jnp.sort(top_k_indices, axis=-1)

        # Gather based on indices that survived
        orig_indices = top_k_indices // 2  # (batch_size, k)
        is_fork = (top_k_indices % 2) == 0  # 0: fork, 1: orig (batch_size, k)

        # Gather x values
        batch_indices = jnp.arange(batch_size)[:, None]
        x_to_consider = x[batch_indices, orig_indices, :]  # (batch_size, k, n_embd)

        # Add fork embedding
        fork_embedding = self.param(
            'fork_embedding',
            nn.with_partitioning(
                lambda rng, shape: jax.random.normal(rng, shape) * (1 / jnp.sqrt(self.config.n_embd)),
                ('n_embd',)
            ),
            (self.config.n_embd,)
        ).astype(jnp.bfloat16)

        x_to_consider = x_to_consider + (is_fork[:, :, None].astype(x.dtype) * fork_embedding)

        # Gather cumulative scores and token indices
        new_cumulative_scores = jnp.take_along_axis(forking_scores_cum, top_k_indices, axis=-1)
        new_token_indices = jnp.take_along_axis(token_index, orig_indices, axis=-1)

        return x_to_consider, new_cumulative_scores, new_token_indices

    def __call__(self, x, cumulative_scores, token_index, padding_mask=None,
                 layer_num=None, deterministic=False):
        # Fork first
        x, cumulative_scores, token_index = self.fork(x, cumulative_scores, token_index)

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

        # Transformer blocks
        self.blocks = [
            ForkingBlock(self.config) if layer == "fork" else Block(self.config)
            for layer in self.config.plan
        ]

        # Final layer norm
        self.ln_f = LayerNorm(self.config.n_embd, bias=self.config.bias)

    @nn.compact
    def __call__(self, idx, targets=None, padding_mask=None, deterministic=False):
        """
        padding_mask should be True at (b,j) for batch `b` and token `j` if (b,j) is *KEPT*
        i.e. True = not padding
        """
        device = idx.device if hasattr(idx, 'device') else None
        b, t = idx.shape
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )

        # Shared token embedding table: (vocab, d_model)
        wte = self.param(
            "wte",
            nn.with_partitioning(
                nn.initializers.normal(stddev=0.02),
                ('vocab', 'n_embd')
            ),
            (self.config.vocab_size, self.config.n_embd),
            jnp.float32
        )

        # Token embeddings
        tok_emb = jnp.take(wte, idx, axis=0).astype(jnp.bfloat16)  # (b, t, n_embd)

        # Dropout on embeddings
        if not deterministic:
            tok_emb = nn.Dropout(rate=self.config.dropout)(tok_emb, deterministic=False)

        x = tok_emb

        # Initialize cumulative scores and token index
        cumulative_scores = jnp.zeros((b, t), dtype=x.dtype)
        token_index = jnp.tile(jnp.arange(t), (b, 1))

        # Transformer blocks
        for indx, (plan, block) in enumerate(zip(self.config.plan, self.blocks)):
            x, cumulative_scores, token_index = block(
                x, cumulative_scores, token_index,
                padding_mask=padding_mask,
                layer_num=indx,
                deterministic=deterministic
            )


        # Final layer norm
        # jax.debug.print("{}", x[0].mean())
        # jax.debug.print("{} {}", x[0].mean(), x.dtype)
        # chickens = self.param('chickens', nn.initializers.ones, (self.config.n_embd,))
        # mean = jnp.mean(x, axis=-1, keepdims=True)
        # var = jnp.var(x, axis=-1, keepdims=True)
        # x_norm = (x - mean) / jnp.sqrt(var + 1e-5)

        # jax.debug.print("{} {}", x[0].astype(jnp.float32).sum(), x.dtype)
        x = self.ln_f(x)
        # jax.debug.print("{} {}", x[0].mean(), x.dtype)

        # Compute logits based on sequence length and averaging method
        if x.shape[1] == idx.shape[-1]:
            logits = jnp.einsum("blh,vh->blv", x, wte.astype(jnp.bfloat16))
        else:
            # Apply the selected averaging method
            if self.config.averaging_method == "logit":
                raise ValueError("Didn't implement this; nah.")
            elif self.config.averaging_method == "residual":
                logits = jnp.einsum("blh,vh->blv", self.residual_average(idx.shape[-1], x, cumulative_scores, token_index), wte.astype(jnp.bfloat16))
            elif self.config.averaging_method == "rightmost":
                # Take only rightmost token of each original token
                try:
                    rolled = jnp.roll(token_index, -1, axis=-1)
                    is_rightmost = rolled != token_index
                    rightmost_x = x[is_rightmost].reshape(b, t, -1)
                    logits = jnp.einsum("blh,vh->blv", rightmost_x, wte.astype(jnp.bfloat16))
                except:
                    raise ValueError(f"averaging_method: {self.config.averaging_method} can't be jitted in the way we have implemented it since rightmost shape is not guaranteed; you shouldn't be using it anyway except for debugging; for details see error above.")
            else:
                raise ValueError(f"Invalid averaging_method: {self.config.averaging_method}")

        # Compute loss if targets provided
        if targets is not None:
            # Cast logits to float32 for numerical stability
            logits_f32 = logits.astype(jnp.float32)
            logits_flat = logits_f32.reshape(-1, logits_f32.shape[-1])
            targets_flat = targets.reshape(-1)

            # Mask out ignore index (-1) AND padding tokens
            mask = targets_flat != -1
            logits_masked = logits_flat
            targets_masked = jnp.where(mask, targets_flat, 0)

            # Compute cross entropy in float32
            loss = -jnp.sum(
                (jax.nn.log_softmax(logits_masked, axis=-1) *
                 jax.nn.one_hot(targets_masked, self.config.vocab_size))*mask[:,None]
            ) / mask.sum().clip(min=1)
        else:
            loss = None

        return logits, loss

    def residual_average(self, input_size, residuals, cumulative_scores, token_index):
        """Average residuals weighted by cumulative scores"""

        # residuals = jax.random.normal(jax.random.PRNGKey(8), (8,128,512))
        # cumulative_scores = jnp.zeros((8,128))
        # token_index = jnp.arange(64).repeat(2)[None].repeat(8, axis=0)

        scaled_residuals = residuals * jnp.exp(cumulative_scores)[:, :, None]
        # jax.debug.print(str(scaled_residuals.shape))

        # Scatter-add to original token positions
        summed_residuals = jnp.zeros(
            (residuals.shape[0], input_size, self.config.n_embd),
            dtype=residuals.dtype
        )
        B, T = token_index.shape
        b = jnp.arange(B)[:, None] # (b, 1)

        return summed_residuals.at[b, token_index].add(scaled_residuals)

    def configure_optimizers_adamw(self, weight_decay, learning_rate, betas, device_type):
        """Returns Optax optimizer for AdamW"""
        import optax

        # Create optimizer chain
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=learning_rate, b1=betas[0], b2=betas[1],
                        weight_decay=weight_decay)
        )

        return optimizer

    def configure_optimizers_muon(self, distributed=False):
        """Returns Optax optimizer combining Muon and AdamW"""
        from muon import create_muon_optax

        # This will be implemented in muon.py
        optimizer = create_muon_optax(self.config, distributed=distributed)

        logger.info(f"OPTIMIZER | using Muon {'distributed' if distributed else 'single device'}")
        return optimizer

# map logical axes to device axes
SHARDING_PLAN = (
    ('n_embd', None), 
    ('n_embd_out', 'shard'), 
    ('n_embd_ff', 'shard'), # for an upproj, we shard along input axis
    ('n_attn', 'shard'),
    ('n_fork', None),
    ('vocab', None),
)

    
