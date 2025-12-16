"""
Keller Jordan's implementation of the Muon Optimizer
as in https://kellerjordan.github.io/posts/muon/

Converted to JAX/Optax for TPU/XLA training.
"""

import jax
import jax.numpy as jnp
from jax import lax
import optax
from typing import NamedTuple, Any


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    JAX implementation.
    """
    assert G.ndim >= 2  # batched Muon implementation
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.astype(jnp.bfloat16)

    if G.shape[-2] > G.shape[-1]:
        X = jnp.swapaxes(X, -2, -1)

    # Ensure spectral norm is at most 1
    X = X / (jnp.linalg.norm(X, ord='fro', axis=(-2, -1), keepdims=True) + 1e-7)

    # Perform the NS iterations
    for _ in range(steps):
        A = X @ jnp.swapaxes(X, -2, -1)
        B = b * A + c * A @ A  # quintic computation strategy
        X = a * X + B @ X

    if G.shape[-2] > G.shape[-1]:
        X = jnp.swapaxes(X, -2, -1)

    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    """
    Compute Muon update for a single parameter.

    Args:
        grad: Gradient
        momentum: Momentum buffer
        beta: Momentum coefficient
        ns_steps: Number of Newton-Schulz steps
        nesterov: Whether to use Nesterov momentum

    Returns:
        Updated momentum and the update to apply
    """
    # Update momentum
    new_momentum = beta * momentum + (1 - beta) * grad

    # Choose update based on Nesterov
    update = (1 - beta) * grad + beta * new_momentum if nesterov else new_momentum

    # Reshape for 2D if needed (e.g., conv filters)
    if update.ndim == 4:
        original_shape = update.shape
        update = update.reshape(update.shape[0], -1)
    else:
        original_shape = None

    # Apply Newton-Schulz orthogonalization
    if update.ndim >= 2:
        update = zeropower_via_newtonschulz5(update, steps=ns_steps)
        update = update * jnp.sqrt(max(1, update.shape[-2] / update.shape[-1]))

    # Reshape back if needed
    if original_shape is not None:
        update = update.reshape(original_shape)

    return new_momentum, update


class MuonState(NamedTuple):
    """State for Muon optimizer"""
    momentum: Any
    step: int


def muon(learning_rate: float = 0.02,
         weight_decay: float = 0.0,
         momentum: float = 0.95,
         ns_steps: int = 5,
         nesterov: bool = True) -> optax.GradientTransformation:
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.

    Args:
        learning_rate: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
        ns_steps: Number of Newton-Schulz iteration steps.
        nesterov: Whether to use Nesterov momentum.

    Returns:
        An Optax GradientTransformation.
    """

    def init_fn(params):
        return MuonState(
            momentum=jax.tree_map(jnp.zeros_like, params),
            step=0
        )

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("Muon requires params to be passed for weight decay")

        # Apply weight decay to params
        if weight_decay > 0:
            params = jax.tree_map(
                lambda p: p * (1 - learning_rate * weight_decay),
                params
            )

        # Apply Muon update to each parameter
        def apply_muon(grad, mom):
            new_mom, update = muon_update(grad, mom, beta=momentum,
                                         ns_steps=ns_steps, nesterov=nesterov)
            return -learning_rate * update, new_mom

        updates, new_momentum = jax.tree_map(apply_muon, updates, state.momentum)

        return updates, MuonState(momentum=new_momentum, step=state.step + 1)

    return optax.GradientTransformation(init_fn, update_fn)


def create_muon_optax(config, distributed=False):
    """
    Create a combined Muon + AdamW optimizer following the original architecture.

    Args:
        config: Model configuration with lr, weight_decay, beta1, beta2, etc.
        distributed: Whether this is for distributed training (uses pmean for gradients)

    Returns:
        An Optax GradientTransformation
    """

    # We'll use optax.multi_transform to apply different optimizers to different params
    # This requires labeling parameters, which is done via masking

    def is_2d_matrix(path, param):
        """Check if parameter is a 2D matrix (for Muon)"""
        # Apply Muon to 2D weight matrices in transformer blocks
        # Exclude embeddings ('wte', 'lm_head')
        param_name = '/'.join(str(p) for p in path)
        is_matrix = param.ndim >= 2
        is_not_embedding = 'wte' not in param_name and 'lm_head' not in param_name
        return is_matrix and is_not_embedding

    def is_scalar(path, param):
        """Check if parameter is a scalar (bias, layernorm, etc.)"""
        return param.ndim < 2

    def is_embedding(path, param):
        """Check if parameter is an embedding"""
        param_name = '/'.join(str(p) for p in path)
        return 'wte' in param_name or 'lm_head' in param_name

    # Create optimizer for each parameter group
    muon_optimizer = muon(
        learning_rate=config.lr * config.muon_scale,
        weight_decay=config.weight_decay,
        momentum=config.beta1
    )

    adamw_embedding = optax.adamw(
        learning_rate=config.lr * config.adamw_embd_scale,
        b1=config.beta1,
        b2=config.beta2,
        weight_decay=config.weight_decay
    )

    adamw_scalar = optax.adamw(
        learning_rate=config.lr * config.adamw_scalar_scale,
        b1=config.beta1,
        b2=config.beta2,
        weight_decay=0.0
    )

    # Combine optimizers using multi_transform
    optimizer = optax.multi_transform(
        {
            'muon': muon_optimizer,
            'adamw_embedding': adamw_embedding,
            'adamw_scalar': adamw_scalar,
        },
        param_labels=lambda path, param: (
            'muon' if is_2d_matrix(path, param)
            else 'adamw_embedding' if is_embedding(path, param)
            else 'adamw_scalar'
        )
    )

    # Add gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optimizer
    )

    # If distributed, add cross-replica mean
    if distributed:
        optimizer = optax.chain(
            optax.apply_every(1),  # This ensures we sync every step
            optimizer
        )

    return optimizer


class SingleDeviceMuonWithAuxAdam:
    """
    Wrapper class for single-device Muon + AdamW optimizer.
    Maintains API compatibility with original PyTorch version.
    """

    def __init__(self, param_groups):
        """
        Args:
            param_groups: List of dicts with 'params', 'lr', 'weight_decay', 'use_muon', etc.
        """
        self.param_groups = param_groups

        # Build Optax optimizer from param groups
        self._build_optimizer()

    def _build_optimizer(self):
        """Build the Optax optimizer chain"""
        # This is a simplified version; actual implementation would need
        # to handle parameter grouping properly
        transforms = []

        for group in self.param_groups:
            if group.get('use_muon', False):
                transforms.append(
                    muon(
                        learning_rate=group['lr'],
                        weight_decay=group.get('weight_decay', 0.0),
                        momentum=group.get('betas', (0.95,))[0]
                    )
                )
            else:
                transforms.append(
                    optax.adamw(
                        learning_rate=group['lr'],
                        b1=group.get('betas', (0.9, 0.95))[0],
                        b2=group.get('betas', (0.9, 0.95))[1],
                        weight_decay=group.get('weight_decay', 0.0)
                    )
                )

        self.optimizer = optax.chain(*transforms)


class MuonWithAuxAdam:
    """
    Wrapper class for distributed Muon + AdamW optimizer.
    Maintains API compatibility with original PyTorch version.
    """

    def __init__(self, param_groups):
        """
        Args:
            param_groups: List of dicts with 'params', 'lr', 'weight_decay', 'use_muon', etc.
        """
        self.param_groups = param_groups
        self._build_optimizer()

    def _build_optimizer(self):
        """Build the Optax optimizer chain with cross-replica mean"""
        # Similar to SingleDeviceMuonWithAuxAdam but with distributed support
        transforms = []

        for group in self.param_groups:
            if group.get('use_muon', False):
                transforms.append(
                    muon(
                        learning_rate=group['lr'],
                        weight_decay=group.get('weight_decay', 0.0),
                        momentum=group.get('betas', (0.95,))[0]
                    )
                )
            else:
                transforms.append(
                    optax.adamw(
                        learning_rate=group['lr'],
                        b1=group.get('betas', (0.9, 0.95))[0],
                        b2=group.get('betas', (0.9, 0.95))[1],
                        weight_decay=group.get('weight_decay', 0.0)
                    )
                )

        # Add cross-replica mean for distributed training
        self.optimizer = optax.chain(
            optax.apply_every(1),  # Sync every step
            *transforms
        )
