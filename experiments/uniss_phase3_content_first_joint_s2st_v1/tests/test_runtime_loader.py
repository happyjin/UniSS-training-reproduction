from contextlib import nullcontext

import torch
from torch import nn

from experiments.uniss_phase3_content_first_joint_s2st_v1.runtime.model_loader import (
    ContentFirstPolicyOverlay,
    IdentityRouteController,
)


class _WrappedLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(3, 2, bias=False)
        self.base.weight.data.zero_()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.target = _WrappedLinear()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.target(value)


def test_identity_route_is_a_noop() -> None:
    controller = IdentityRouteController()
    with controller.route(True):
        value = 7
    controller.close()
    assert value == 7


def test_policy_overlay_is_additive_and_route_scoped() -> None:
    model = _TinyModel()
    a = torch.tensor([[1.0, 0.0, 0.0]])
    b = torch.tensor([[2.0], [3.0]])
    controller = ContentFirstPolicyOverlay(
        model, {"target": (a, b)}, scale=0.5
    )
    value = torch.tensor([[4.0, 5.0, 6.0]])
    torch.testing.assert_close(model(value), torch.zeros(1, 2))
    with controller.route(True):
        torch.testing.assert_close(model(value), torch.tensor([[4.0, 6.0]]))
    torch.testing.assert_close(model(value), torch.zeros(1, 2))
    controller.close()
    with nullcontext():
        torch.testing.assert_close(model(value), torch.zeros(1, 2))
