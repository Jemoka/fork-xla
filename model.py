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

import torch
import torch.nn as nn
from torch.nn import functional as F

from loguru import logger
from utils import plot

from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

class RotaryPosEncoding(torch.nn.Module):
    """RoPE implementation by Róbert Csordás"""
    def __init__(self, d_model: int, base=10000, seq_dim: int = 1):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError("RoPE can only be used with an even number of dimensions")

        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None
        self.seq_dim = seq_dim

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat(
            (-x2, x1), dim=x1.ndim - 1
        )  # dim=-1 triggers a bug in torch < 1.8.0

    def apply_rot(self, x: torch.Tensor, sinp: torch.Tensor, cosp: torch.Tensor, seq_dim: int, offset: int) -> torch.Tensor:
        sin = sinp.narrow(seq_dim, offset, x.shape[seq_dim])
        cos = cosp.narrow(seq_dim, offset, x.shape[seq_dim])
        return (x * cos) + (self.rotate_half(x) * sin)

    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor,
                             seq_dim: int, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply_rot(q, sin, cos, seq_dim, offset), self.apply_rot(k, sin, cos, seq_dim, 0)

    def get(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[self.seq_dim]
        enable_cache = t is None

        if (not enable_cache) or (seq_len > self.seq_len_cached):
            self.seq_len_cached = seq_len
            if t is None:
                t = torch.arange(x.shape[self.seq_dim], device=x.device)

            t = t.type_as(self.inv_freq)

            freqs = torch.einsum("...i,j->...ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            tgt_shape = [1] * x.ndim
            tgt_shape[self.seq_dim] = seq_len
            tgt_shape[-1] = x.shape[-1]

            # support batch.
            tgt_shape[0] = -1

            cos = emb.cos().view(*tgt_shape)
            sin = emb.sin().view(*tgt_shape)

            if enable_cache:
                self.cos_cached = cos
                self.sin_cached = sin
            else:
                return sin, cos

        return self.sin_cached, self.cos_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                pos_offset: int=0,
                t: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        sin, cos = self.get(k, t)
        return self.apply_rotary_pos_emb(q, k, sin, cos, self.seq_dim, pos_offset)

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # rotational position embedding
        self.rope = RotaryPosEncoding(config.n_embd//config.n_head, seq_dim = -2)
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        assert hasattr(torch.nn.functional, "scaled_dot_product_attention"), "Flash Attention requires PyTorch >= 2.0"
        self.config = config

        # we cache the computed attention masks; we don't need to save this
        # Map: (int,int) -> torch.Tensor
        # q.size(-2), k.size(-2) ->
        #          torch.ones(q.size(-2), k.size(-2), dtype=torch.bool).tril(diagonal=0)
        self.__cached_casual_mask = {}

    def compute_channels(
            self, x, cumulative_scores,
            token_index, padding_mask=None
    ):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # apply position embeddings: RoPE with fractional rotations
        # based on the number of forks
        # number of forks for each token, ^-1 of which is the
        # "moves" that each fork will contribute from 0
        token_counts = torch.scatter_add(
            torch.zeros(
                *token_index.shape[:-1], self.config.block_size,
                dtype=token_index.dtype,
                device=token_index.device,
            ),
            -1,
            token_index,
            torch.ones_like(token_index)
        )
        # token_counts = scatter_add(torch.ones_like(token_index), token_index, -1)
        partial_rotations = torch.cumsum(
            torch.gather(
                (1/token_counts),
                -1,
                token_index
            ),
            dim=-1
        )
        q,k = self.rope(q,k,t=partial_rotations)

        # stick an extra row of 1s to the end of q (this is for cumulative score addition pre softmax)
        # yes we do loose a channel, but sadly the alternative is to launch a whole new kernel cell just for
        # padding the next 15 elements with blanks (i.e. to have dims be (... x 129)), which is bad juju
        if "fork" in self.config.plan:
            q[:,:,:, -1] = torch.ones_like(q[:,:,:, -1])
            k[:,:,:, -1] = cumulative_scores.unsqueeze(-2).repeat(1, k.size(1), 1)

        # build causal attention mask if not already cached
        casual_mask = self.__cached_casual_mask.get((q.size(-2), k.size(-2)))
        if casual_mask is None:
            casual_mask = torch.ones(
                q.size(-2),
                k.size(-2),
            ).tril(diagonal=0).to(q.device).bool()
            casual_mask = torch.where(
                casual_mask,
                torch.zeros_like(casual_mask, dtype=torch.bfloat16),
                torch.full_like(casual_mask, float("-inf"), dtype=torch.bfloat16)
            )
            self.__cached_casual_mask[(q.size(-2), k.size(-2))] = casual_mask
        casual_mask = casual_mask.unsqueeze(0).repeat(x.size(0), 1, 1)
        mask = casual_mask

        # add additional padding mask, if necessary
        if padding_mask is not None:
            padding_mask = padding_mask.gather(1, token_index)
            padding_mask_fl = (padding_mask).float()
            padding_mask_outer = (
                padding_mask_fl.unsqueeze(-1) @
                padding_mask_fl.unsqueeze(-2)
            )
            padding_mask_additive = torch.where(
                padding_mask_outer.bool(),
                torch.zeros_like(padding_mask_outer, dtype=torch.bfloat16),
                torch.full_like(padding_mask_outer, float("-inf"), dtype=torch.bfloat16)
            )
            mask += padding_mask_additive

        return q,k,v,mask

    def forward(self, x, cumulative_scores, token_index, padding_mask=None):
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        q,k,v,mask = self.compute_channels(
            x,
            cumulative_scores,
            token_index,
            padding_mask
        )

        # attenuate v values with cumulative scores, so that tokens that are about to be
        # killed cannot be attended to
        v = torch.einsum("bnlh,bl -> bnlh", v, cumulative_scores.exp())

        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask.unsqueeze(-3).repeat(1,self.n_head,1,1).to(q.dtype),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False, # since we are giving a custom mask
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection

        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, cumulative_scores, token_index, padding_mask=None, layer_num=None):
        exponentiated_scores = cumulative_scores.exp()
        x = x + torch.einsum("bl,blh -> blh", exponentiated_scores, self.attn(
            self.ln_1(x), cumulative_scores, token_index, padding_mask=padding_mask
        ))
        x = x + torch.einsum("bl,blh -> blh", exponentiated_scores, self.mlp(self.ln_2(x)))
        return x


class ForkingBlock(Block):
    def __init__(self, config):
        super().__init__(config)

        self.forking_ln = LayerNorm(config.n_embd, bias=config.bias)
        self.forking_score = nn.Linear(config.n_embd, 2, bias=False)
        self.fork_embedding = nn.Parameter(torch.rand(config.n_embd)*
                                           (1/math.sqrt(config.n_embd)))
        self.config = config

    @staticmethod
    def gumbel_sigmoid_noise(logits: torch.Tensor) -> torch.Tensor:
        eps = 3e-4 if logits.dtype == torch.float16 else 1e-10
        uniform = logits.new_empty([2] + list(logits.shape)).uniform_(eps, 1 - eps)

        noise = -(uniform[1].log() / uniform[0].log() + eps).log()
        return noise
    # self.gumbel_sigmoid_noise = gumbel_sigmoid_noise

    @staticmethod
    def clipped_logsigmoid(x, min_val=-20.0):
        logsigmoid = -F.softplus(-x)
        return torch.clamp(logsigmoid, min=min_val)
    # self.clipped_logsigmoid = clipped_logsigmoid

    def fork(self, x, cumulative_scores, token_index):
        return self._topk_fork(x, cumulative_scores, token_index)

    def _threshold_fork(self, x, cumulative_scores, token_index):
        """forking such that any element >0.5 is forked"""

        batch_size = len(cumulative_scores)

        # compute current layer's forking scores given residuals
        curr_layer_forking_score = self.forking_score(self.forking_ln(x))
        forking_scores_cum = (
            self.clipped_logsigmoid(curr_layer_forking_score) +
            cumulative_scores.unsqueeze(-1)
        ).view(batch_size, -1)
        forking_scores_cum_for_analysis = forking_scores_cum.clone()
        forked_token_index = (token_index
                              .unsqueeze(-1)
                              .repeat(*([1]*token_index.ndim),2)
                              .flatten(-2,-1))
        forking_scores_cum_for_analysis[
            torch.roll(forked_token_index, -1) != forked_token_index
        ] = float("+inf") # rightmost token of every token should be 1
        forking_judgement = (forking_scores_cum_for_analysis.exp() > 0.5)

        # set the forking scores of dropped tokens to -inf
        drop_template = torch.full_like(forking_scores_cum, float("-inf"))
        new_forking_scores = torch.where(
            forking_judgement,
            forking_scores_cum,
            drop_template
        ) # compute new cum scores, which is 0 for killed tokens

        # gather each token twice, the leftwards ones are forked
        exploded_indx = torch.arange(
            forking_scores_cum_for_analysis.shape[-1]
        ).to(forking_scores_cum_for_analysis.device).unsqueeze(0).repeat(
            forking_scores_cum_for_analysis.shape[0], 1
        )
        x_to_consider = torch.gather(
            x, -2, (exploded_indx//2).unsqueeze(-1).expand(-1, -1, x.size(-1))
        )  # (batch_size, k, n_embd)
        x_to_consider[:,::2,:] = x_to_consider[:,::2,:] + self.fork_embedding # add fork embeddings to forks
        new_token_indicies = (exploded_indx//2)

        # finally, optimizations where if an entire batch's one column
        # is -inf we nuke it as a whole
        keep_columns = (~(new_forking_scores == float("-inf")).all(dim=-2))
        x_to_consider = x_to_consider[:, keep_columns, :]
        new_token_indicies = new_token_indicies[:, keep_columns]
        new_forking_scores = forking_scores_cum[:, keep_columns]

        # TODO
        # - implement during attn + block update the -inf mechanism for <0.5
        # - implement pi controller

        # x = x_to_consider
        # cumulative_scores = new_forking_scores
        # token_index = new_token_indicies

        return x_to_consider, new_forking_scores, new_token_indicies


    def _topk_fork(self, x, cumulative_scores, token_index):
        """forking via top-k judgement"""
        batch_size = len(cumulative_scores)

        # compute current layer's forking scores given residuals
        curr_layer_forking_score = self.forking_score(self.forking_ln(x))

        # update cumulative scores
        forking_scores_cum = (
            self.clipped_logsigmoid(curr_layer_forking_score) +
            cumulative_scores.unsqueeze(-1)
        ).view(batch_size, -1)
        forking_scores_cum_for_topk = forking_scores_cum.clone()

        # we copy the index twice
        forked_token_index = (token_index
                              .unsqueeze(-1)
                              .repeat(*([1]*token_index.ndim),2)
                              .flatten(-2,-1))
        forking_scores_cum_for_topk[
            torch.roll(forked_token_index, -1) != forked_token_index
        ] = float("+inf") # rightmost token of every token should be 1

        # perform top k, but not as much if the block size is too shallow
        k = min(
            self.config.max_block_size,
            forking_scores_cum_for_topk.size(-1)
        )

        # compute existing and deleted tokens
        _, top_k_indices = torch.topk(forking_scores_cum_for_topk, k, dim=-1, sorted=True)
        top_k_indices = top_k_indices.sort().values

        # gather based on the indicies that survived
        orig_indices = top_k_indices // 2  # (batch_size, k)
        is_fork = (top_k_indices % 2) == 0  # (0: orig, 1: fork)

        x_to_consider = torch.gather(
            x, -2, orig_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        )  # (batch_size, k, n_embd)
        x_to_consider = x_to_consider + (
            is_fork.unsqueeze(-1).to(self.fork_embedding.dtype) * self.fork_embedding)

        # gather cumulative scores
        new_cumulative_scores = forking_scores_cum.gather(-1, top_k_indices) 
        new_token_indicies = token_index.gather(-1, orig_indices)

        # x = x_to_consider
        # cumulative_scores = new_cumulative_scores
        # token_index = new_token_indicies

        # return updated values
        return x_to_consider, new_cumulative_scores, new_token_indicies


    def forward(self, x, cumulative_scores, token_index, padding_mask=None, layer_num=None):
        # fork!
        x, cumulative_scores, token_index = self.fork(
            x,
            cumulative_scores,
            token_index
        )

        if not self.config.compiled:
            plot(
                "forking",
                cumulative_scores=cumulative_scores.detach(),
                token_index=token_index.detach(),
                key=layer_num
            )

        # and now a normal layer
        return (
            super().forward(x, cumulative_scores, token_index),
            cumulative_scores,
            token_index
        )

class Thoughtbubbles(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([
                    (
                        ForkingBlock(config)
                        if layer == "fork" else
                        Block(config)
                    ) for layer in config.plan
                ]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.plan = config.plan
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * len(config.plan))
                )

        # report number of parameters
        logger.info("MODEL | number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
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

    def forward(self, idx, targets=None, padding_mask=None):
        # mock this
        # idx = torch.randint(128,(9, self.config.block_size)).cuda()
        # b, t = idx.size()
        # pos = torch.arange(0, t, dtype=torch.long).cuda()  # shape (t)
        # tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        # x = self.transformer.drop(tok_emb)
        # cumulative_scores = torch.zeros_like(idx, dtype=x.dtype).to(x.device)
        # token_index = pos.repeat(b, 1)
        #######

        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)

        # compute cumulative scores and token index, which each forknig block modulates
        # we initialize cumulative scores to 1 (e^0 = 1)
        cumulative_scores = torch.zeros_like(idx, dtype=x.dtype).to(x.device)
        token_index = pos.repeat(b, 1)

        # transformer computation
        for indx, (plan, block) in enumerate(zip(self.plan, self.transformer.h)):
            if "fork" in plan:
                (
                    x,
                    cumulative_scores,
                    token_index
                ) = block(
                    x,
                    cumulative_scores,
                    token_index,
                    padding_mask=padding_mask,
                    layer_num=indx
                )
            else:
                x = block(
                    x,
                    cumulative_scores,
                    token_index,
                    padding_mask=padding_mask,
                    layer_num=indx
                )
        
        # normalization + logit average to decode the final logits
        x = self.transformer.ln_f(x)
        if x.size(-2) == self.config.block_size:
            logits = self.lm_head(x)
        else:
            # Apply the selected averaging method
            if self.config.averaging_method == "logit":
                logits = self.logit_average(self.lm_head(x), cumulative_scores, token_index)
            elif self.config.averaging_method == "residual":
                logits = self.lm_head(self.residual_average(x, cumulative_scores, token_index))
            elif self.config.averaging_method == "rightmost":
                logits = self.lm_head(x)[(torch.roll(token_index, -1) != token_index)].view(*tok_emb.shape[:-1], -1)
            else:
                raise ValueError(f"Invalid averaging_method: {self.config.averaging_method}. Must be one of: 'logit', 'residual', 'rightmost'")
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            loss = None

        return logits, loss

    def residual_average(self, residuals, cumulative_scores, token_index):
        # residuals: (batch_size, len, hidden_size)
        # token_index (batch_size, len)
        # cumulative_scores: (batch_size, len)

        # scatter_max, it turns out, is implemented in an extremely
        # wonky way. so we skip this and hope that we don't underflow

        ####### here's an implementation using scatter_max #######
        # # max for each group (result: (batch_size, vocab_size))
        # max_score, _ = scatter_max(cumulative_scores, token_index, dim=-1)
        # # brodcast into the original tokens
        # max_score_broadcasted = max_score.gather(
        #     -1, 
        #     token_index
        # ) # (batch_size, forked_len)
        # # subtract max score and exp, this is the logsumexp trick
        # shifted_cum_scores = (cumulative_scores - max_score_broadcasted).exp()
        # scaled_residuals = (residuals * shifted_cum_scores.unsqueeze(-1))
        
        # # Sum over each group
        # summed_residuals = scatter_add(scaled_residuals, token_index, -2)

        # # and add back the max score we subtracted
        # return (max_score.unsqueeze(-1).exp()*summed_residuals)
        ####### here's an implementation using scatter_max #######

        scaled_residuals = (residuals * cumulative_scores.exp().unsqueeze(-1))
        summed_residuals = torch.scatter_add(
            torch.zeros(
                *token_index.shape[:-1], self.config.block_size, self.config.n_embd,
                dtype=scaled_residuals.dtype,
                device=scaled_residuals.device,
            ),
            -2,
            token_index.unsqueeze(-1).repeat(1, 1, self.config.n_embd),
            scaled_residuals
        )

        return summed_residuals


    def configure_optimizers_adamw(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(
            f"MODEL | num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        logger.info(
            f"MODEL | num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        logger.info(f"OPTIMIZER | using fused AdamW: {use_fused}")

        return optimizer

    def configure_optimizers_muon(self, distributed=False):
        hidden_matrix_params = [p for n, p in self.transformer.h.named_parameters() if (p.ndim >= 2 and p.requires_grad and "wte" not in n)]
        embed_params = [p for n, p in self.named_parameters() if "wte" in n]
        scalar_params = [p for p in self.parameters() if p.ndim < 2]

        # sanity check that we got everything
        assert (set(hidden_matrix_params + embed_params + scalar_params) ==
                set(self.parameters())), "whoops, looks like we missed some parameters?"
        assert (len(hidden_matrix_params + embed_params + scalar_params) ==
                len(list(self.parameters()))), "whoops, looks like we double counted some parameters?"

        # talky talk
        logger.info(
            f"MODEL | num decayed AdamW parameter tensors: {len(embed_params)}, with {sum(p.numel() for p in embed_params):,} parameters"
        )
        logger.info(
            f"MODEL | num non-decayed AdamW parameter tensors: {len(scalar_params)}, with {sum(p.numel() for p in scalar_params):,} parameters"
        )
        logger.info(
            f"MODEL | num Muon parameter tensors: {len(hidden_matrix_params)}, with {sum(p.numel() for p in hidden_matrix_params):,} parameters"
        )

        # form adam groups 
        parameters = [
            dict(params=embed_params, lr=self.config.lr*self.config.adamw_embd_scale, weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2), use_muon=False),
            dict(params=scalar_params, lr=self.config.lr*self.config.adamw_scalar_scale,
                betas=(self.config.beta1, self.config.beta2), use_muon=False, weight_decay=0.0),
            dict(params=hidden_matrix_params, lr=self.config.lr*self.config.muon_scale,
                weight_decay=self.config.weight_decay, use_muon=True),

        ]

        logger.info(f"OPTIMIZER | using Muon {'distributed' if distributed else 'single device'}")

        return MuonWithAuxAdam(parameters) if distributed else SingleDeviceMuonWithAuxAdam(parameters)

