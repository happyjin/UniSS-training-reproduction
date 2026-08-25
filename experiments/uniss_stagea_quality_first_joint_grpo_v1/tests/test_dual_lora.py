import torch
from torch import nn

from experiments.uniss_stagea_quality_first_joint_grpo_v1.model.dual_lora import (
    DualLoRAController,
)


def test_reference_snapshot_and_route_mask():
    linear = nn.Linear(4, 4, bias=False)
    controller = DualLoRAController()
    controller.add("layer", linear, rank=2, alpha=4.0, dropout=0.0)
    with torch.no_grad():
        controller.policy["layer"].lora_b.fill_(0.5)
    controller.snapshot_reference()
    assert bool(controller.reference_ready)
    assert controller.reference_anchor().item() == 0.0
    controller.set_active_mask(torch.tensor([True, False]))
    value = torch.ones(2, 4)
    with controller.use("disabled"):
        base = linear(value)
    with controller.use("policy"):
        adapted = linear(value)
    assert not torch.equal(adapted[0], base[0])
    assert torch.equal(adapted[1], base[1])
