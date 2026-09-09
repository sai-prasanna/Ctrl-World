"""Check covariance identities, shared-noise cancellation, and weight hooks."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from uncertainty.subspace import (WeightSubspace, finite_jacobian,
                                  posterior_covariance, variance_map,
                                  action_response_variance)


def main():
    torch.manual_seed(1)
    j = torch.randn(3, 2, 4)
    covariance = posterior_covariance(torch.eye(3), 2.0)
    torch.testing.assert_close(covariance, torch.eye(3).double() * 2 / 3)
    torch.testing.assert_close(variance_map(j, covariance), j.square().sum(0) * 2 / 3)
    assert action_response_variance(j, j, covariance).count_nonzero() == 0
    # Shared appearance uncertainty cancels, while uncertain action effects remain.
    appearance = torch.randn_like(j)
    effect = torch.randn_like(j)
    torch.testing.assert_close(action_response_variance(appearance + effect, appearance,
                                                       covariance),
                               variance_map(effect, covariance))
    model = torch.nn.Module()
    model.action_encoder = torch.nn.Sequential(torch.nn.Linear(4, 2, bias=False))
    x = torch.randn(5, 4)
    original = model.action_encoder(x).detach().clone()
    weights = model.action_encoder[0].weight.detach().clone()
    space = WeightSubspace(model, rank=3, relative_scale=0.1)
    jac = finite_jacobian(lambda: model.action_encoder(x), space)
    coeff = torch.tensor([0.2, -0.1, 0.3])
    space.set(coeff)
    torch.testing.assert_close(model.action_encoder(x), original + torch.einsum('r,rij->ij', coeff, jac))
    space.set()
    assert torch.equal(original, model.action_encoder(x))
    assert torch.equal(weights, model.action_encoder[0].weight)
    try:
        finite_jacobian(lambda: (_ for _ in ()).throw(RuntimeError('expected')), space)
    except RuntimeError:
        pass
    assert space.coefficients is None
    space.close()
    # Correlated coefficients must retain off-diagonal covariance contributions.
    correlated = torch.tensor([[1.0, 0.5], [0.5, 2.0]], dtype=torch.float64)
    derivatives = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    explicit = derivatives.T.double() @ correlated @ derivatives.double()
    torch.testing.assert_close(variance_map(derivatives, correlated), explicit.diag().float())
    # Shared parameters across two steps can cancel their effects completely;
    # adding independent step variances would incorrectly give a positive value.
    step_one, step_two = torch.tensor([[2.0]]), torch.tensor([[-2.0]])
    assert variance_map(step_one + step_two, torch.ones(1, 1)).item() == 0
    assert (variance_map(step_one, torch.ones(1, 1)) +
            variance_map(step_two, torch.ones(1, 1))).item() == 8
    # Increasing data precision cannot increase the projected posterior variance.
    smaller = posterior_covariance(torch.eye(3) * 10, 2.0)
    assert (variance_map(j, smaller) <= variance_map(j, covariance)).all()
    print('PASS: covariance, action cancellation, linear finite differences, restoration, contraction')


if __name__ == '__main__':
    main()
