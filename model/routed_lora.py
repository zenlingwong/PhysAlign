from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .projector import ProjectorBank


@dataclass(frozen=True)
class RoutedLoRAConfig:
    visible_rank: int
    complement_rank: int
    alpha: float = 8.0
    seed: int = 0

    @property
    def total_rank(self) -> int:
        return self.visible_rank + self.complement_rank

    def __post_init__(self) -> None:
        if self.visible_rank < 0 or self.complement_rank < 0 or self.total_rank < 1:
            raise ValueError("routed ranks must be non-negative with a positive total")
        if not torch.isfinite(torch.tensor(self.alpha)) or self.alpha <= 0.0:
            raise ValueError("alpha must be finite and positive")


class RoutedLoRALinear(nn.Module):
    def __init__(
        self,
        weight: Tensor,
        projector: ProjectorBank,
        config: RoutedLoRAConfig,
        bias: Tensor | None = None,
    ) -> None:
        super().__init__()
        if weight.ndim != 2 or weight.shape[1] != projector.input_dim:
            raise ValueError("weight and projector dimensions do not match")
        if config.visible_rank > projector.visible_dim or config.complement_rank > projector.complement_dim:
            raise ValueError("routed ranks exceed their projector dimensions")
        self.projector = projector
        self.config = config
        self.register_buffer("weight", weight.detach().clone(), persistent=True)
        self.register_buffer("bias", None if bias is None else bias.detach().clone(), persistent=True)
        output_dim = weight.shape[0]
        self.Bv = nn.Parameter(torch.zeros(output_dim, config.visible_rank, device=weight.device, dtype=weight.dtype))
        self.Cv = nn.Parameter(torch.empty(config.visible_rank, projector.visible_dim, device=weight.device, dtype=weight.dtype))
        self.Bc = nn.Parameter(torch.zeros(output_dim, config.complement_rank, device=weight.device, dtype=weight.dtype))
        self.Cc = nn.Parameter(torch.empty(config.complement_rank, projector.complement_dim, device=weight.device, dtype=weight.dtype))
        generator = torch.Generator(device=weight.device.type)
        generator.manual_seed(config.seed)
        bound_v = 1.0 / max(projector.visible_dim, 1) ** 0.5
        bound_c = 1.0 / max(projector.complement_dim, 1) ** 0.5
        with torch.no_grad():
            if config.visible_rank:
                self.Cv.uniform_(-bound_v, bound_v, generator=generator)
            if config.complement_rank:
                self.Cc.uniform_(-bound_c, bound_c, generator=generator)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        projector: ProjectorBank,
        config: RoutedLoRAConfig,
    ) -> "RoutedLoRALinear":
        return cls(linear.weight, projector, config, linear.bias)

    @property
    def scale(self) -> float:
        return self.config.alpha / self.config.total_rank

    def effective_delta_weight(self) -> Tensor:
        delta = self.weight.new_zeros(self.weight.shape)
        if self.config.visible_rank:
            qv = self.projector.basis("visible", device=self.Bv.device, dtype=self.Bv.dtype)
            delta = delta + self.Bv @ self.Cv @ qv.T
        if self.config.complement_rank:
            qc = self.projector.basis("complement", device=self.Bc.device, dtype=self.Bc.dtype)
            delta = delta + self.Bc @ self.Cc @ qc.T
        return self.scale * delta

    def forward(self, values: Tensor) -> Tensor:
        return F.linear(values, self.weight + self.effective_delta_weight(), self.bias)
