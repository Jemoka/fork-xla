from typing import Optional, Tuple

import torch
from torch_scatter import scatter_add as _ts_scatter_add
from torch_scatter import scatter_max as _ts_scatter_max


# ---- Shared helpers ---------------------------------------------------------

def _canonical_dim(dim: int, ndim: int) -> int:
    return dim + ndim if dim < 0 else dim


def _scatter_out_shape(
    src: torch.Tensor,
    dim: int,
    dim_size: Optional[int],
) -> Tuple[int, ...]:
    """Return the output shape for scatter ops: same as src, but shape[dim]=dim_size.

    If dim_size is None, the real op uses index.max()+1 (data-dependent), so in fake
    we create a dynamic SymInt via torch.library.get_ctx().new_dynamic_size().
    """
    dim = _canonical_dim(dim, src.dim())
    if dim_size is None:
        ctx = torch.library.get_ctx()
        dim_size = ctx.new_dynamic_size()
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    return tuple(out_shape)


# torch_scatter supports broadcasting index to src in some APIs; for backward we
# want the same behavior. If torch_scatter.utils.broadcast exists, use it.
try:
    from torch_scatter.utils import broadcast as _broadcast_index  # type: ignore
except Exception:
    # Fallback that matches the common/expected torch_scatter convention:
    # index is 1D and aligns to src along `dim`, broadcast across other dims.
    def _broadcast_index(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
        dim = _canonical_dim(dim, src.dim())
        view_shape = (1,) * dim + (-1,) + (1,) * (src.dim() - dim - 1)
        return index.view(view_shape).expand_as(src)


# ---- scatter_add custom op --------------------------------------------------

@torch.library.custom_op("thoughtbubbles::scatter_add", mutates_args=())
def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    # torch_scatter semantics: out.shape == src.shape except out.size(dim) == dim_size
    # (or inferred from index.max()+1 if dim_size is None).
    return _ts_scatter_add(src, index, dim=dim, dim_size=dim_size)


@scatter_add.register_fake
def _(src, index, dim: int = -1, dim_size: Optional[int] = None):
    out_shape = _scatter_out_shape(src, dim, dim_size)
    return src.new_empty(out_shape)


def _scatter_add_setup_context(ctx, inputs, output):
    src, index, dim, dim_size = inputs
    ctx.dim = dim
    # Save src for shape+dim canonicalization, and index for gather
    ctx.save_for_backward(src, index)


def _scatter_add_backward(ctx, grad_out):
    src, index = ctx.saved_tensors
    dim = _canonical_dim(ctx.dim, src.dim())
    index_b = _broadcast_index(index, src, dim)
    # d/dsrc scatter_add == gather(grad_out, index) along dim
    grad_src = grad_out.gather(dim, index_b)
    return grad_src, None, None, None


torch.library.register_autograd(
    "thoughtbubbles::scatter_add",
    _scatter_add_backward,
    setup_context=_scatter_add_setup_context,
)


# ---- scatter_max custom op --------------------------------------------------

@torch.library.custom_op("thoughtbubbles::scatter_max", mutates_args=())
def scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    dim_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # torch_scatter returns (out, argmax)
    return _ts_scatter_max(src, index, dim=dim, dim_size=dim_size)


@scatter_max.register_fake
def _(src, index, dim: int = -1, dim_size: Optional[int] = None):
    out_shape = _scatter_out_shape(src, dim, dim_size)
    out = src.new_empty(out_shape)
    argmax = src.new_empty(out_shape, dtype=torch.long)
    return out, argmax


def _scatter_max_setup_context(ctx, inputs, output):
    src, index, dim, dim_size = inputs
    out, argmax = output
    ctx.dim = dim
    ctx.src_shape = src.shape
    ctx.save_for_backward(argmax)


def _scatter_max_backward(ctx, grad_out, grad_argmax):
    (argmax,) = ctx.saved_tensors
    # Gradient flows only to the argmax locations.
    grad_src = grad_out.new_zeros(ctx.src_shape)
    dim = _canonical_dim(ctx.dim, grad_src.dim())

    # torch_scatter uses -1 in argmax for "empty" segments; mask those out
    # so we don't scatter into an invalid/last position.
    valid = argmax >= 0
    arg_safe = argmax.clamp(min=0)
    grad_out_safe = grad_out * valid.to(dtype=grad_out.dtype)

    grad_src.scatter_(dim, arg_safe.detach(), grad_out_safe)
    return grad_src, None, None, None


torch.library.register_autograd(
    "thoughtbubbles::scatter_max",
    _scatter_max_backward,
    setup_context=_scatter_max_setup_context,
)
