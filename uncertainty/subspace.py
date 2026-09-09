"""A restricted parameter posterior and paired action-response covariance.

The basis defines additive weight directions, not activation noise. Coefficients
must stay fixed across every denoising step and both action conditions.
"""

import math
import torch


class WeightSubspace:
    """Install reproducible rank-one weight directions in attention and actions.

    This estimates uncertainty only in the chosen subspace. It is not a posterior
    over all weights. Zero coefficients exactly preserve the checkpoint function.
    """

    def __init__(self, model, rank=8, relative_scale=0.02, seed=0):
        if rank < 1 or relative_scale <= 0:
            raise ValueError('rank and relative_scale must be positive')
        self.rank = rank
        self.handles = []
        self.names = []
        self.coefficients = None
        self.reference_parameter = next(model.parameters())
        generator = torch.Generator().manual_seed(seed)
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not (name.startswith('action_encoder.') or
                    (name.startswith('unet.') and any(
                        name.endswith(s) for s in ('to_q', 'to_k', 'to_v', 'to_out.0')))):
                continue
            a = torch.randn(rank, module.in_features, generator=generator)
            a /= math.sqrt(module.in_features)
            b = torch.randn(module.out_features, rank, generator=generator)
            b *= (relative_scale * module.weight.detach().float().square().mean().sqrt().cpu()
                  * math.sqrt(module.in_features) / math.sqrt(rank))
            a = a.to(module.weight)
            b = b.to(module.weight)

            def hook(layer, inputs, output, a=a, b=b):
                if self.coefficients is None:
                    return output
                coeff = self.coefficients.to(output)
                return output + ((inputs[0] @ a.T) * coeff) @ b.T

            self.names.append(name)
            self.handles.append(module.register_forward_hook(hook))
        if not self.names:
            raise ValueError('No supported linear layers found')

    def set(self, coefficients=None):
        if coefficients is not None and coefficients.shape != (self.rank,):
            raise ValueError('Expected one coefficient per basis direction')
        # Transfer once per probe, avoiding a blocking host-to-device copy in
        # every attention layer during each denoising step.
        self.coefficients = None if coefficients is None else coefficients.to(self.reference_parameter)

    def close(self):
        self.set()
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def finite_jacobian(function, space, step=0.1):
    """Return derivatives with shape (rank, *output_shape) on CPU.

    The caller must reset all sampling randomness and scheduler state on each
    function call. Restoring coefficients also happens if inference fails.
    """
    if step <= 0:
        raise ValueError('step must be positive')
    rows = []
    try:
        for index in range(space.rank):
            direction = torch.zeros(space.rank)
            direction[index] = step
            space.set(direction)
            plus = function().detach().float().cpu()
            space.set(-direction)
            minus = function().detach().float().cpu()
            rows.append((plus - minus) / (2 * step))
    finally:
        space.set()
    return torch.stack(rows)


def posterior_covariance(gram, observation_variance, prior_precision=1.0):
    """Invert a GGN precision for the explicitly normalized pseudo-likelihood.

    gram is a sum over clips of J J^T / output_dimension. This treats each clip
    as one effective observation; it does not count correlated pixels or repeated
    diffusion timesteps as independent training examples.
    """
    if observation_variance <= 0 or prior_precision <= 0:
        raise ValueError('Variances and precisions must be positive')
    gram = gram.double()
    precision = gram / observation_variance
    precision += prior_precision * torch.eye(len(gram), dtype=gram.dtype, device=gram.device)
    return torch.cholesky_inverse(torch.linalg.cholesky(precision))


def variance_map(jacobian, covariance):
    """Project the full subspace covariance without discarding cross terms."""
    shape = jacobian.shape[1:]
    flat = jacobian.reshape(jacobian.shape[0], -1).double()
    value = (flat * (covariance.to(flat) @ flat)).sum(0)
    return value.clamp_min(0).reshape(shape).float()


def action_response_variance(actual_jacobian, hold_jacobian, covariance):
    """Var_theta[G_theta(a) - G_theta(hold)] at shared diffusion noise."""
    if actual_jacobian.shape != hold_jacobian.shape:
        raise ValueError('Action branches must have matching shapes')
    return variance_map(actual_jacobian - hold_jacobian, covariance)
