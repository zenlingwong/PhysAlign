from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ProjectorConfig:
    energy_threshold: float = 0.999
    mean_tolerance: float = 1.0e-10
    input_axis: int = -1

    def __post_init__(self) -> None:
        if not 0.0 < self.energy_threshold <= 1.0:
            raise ValueError("energy_threshold must lie in (0, 1]")
        if self.mean_tolerance < 0.0:
            raise ValueError("mean_tolerance must be non-negative")


def _flatten(activations: Tensor | Iterable[Tensor], axis: int) -> Tensor:
    batches = [activations] if isinstance(activations, Tensor) else list(activations)
    if not batches:
        raise ValueError("activation calibration requires at least one batch")
    rows: list[Tensor] = []
    for batch in batches:
        if batch.ndim < 2:
            raise ValueError("activation batches must have at least two dimensions")
        normalized_axis = axis if axis >= 0 else batch.ndim + axis
        if normalized_axis < 0 or normalized_axis >= batch.ndim:
            raise ValueError("input_axis is outside the activation tensor")
        value = batch.movedim(normalized_axis, -1).reshape(-1, batch.shape[normalized_axis])
        rows.append(value.to(dtype=torch.float64, device="cpu"))
    result = torch.cat(rows, dim=0)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("calibration activations must be finite")
    return result


def _canonical_signs(columns: Tensor) -> Tensor:
    if columns.shape[1] == 0:
        return columns
    pivots = columns.abs().argmax(dim=0)
    signs = columns.gather(0, pivots.unsqueeze(0)).squeeze(0).sign()
    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
    return columns * signs


def _complete_basis(visible: Tensor) -> tuple[Tensor, Tensor]:
    input_dim = visible.shape[0]
    if visible.shape[1] == input_dim:
        return _canonical_signs(visible), visible.new_zeros((input_dim, 0))
    seed = torch.cat((visible, torch.eye(input_dim, dtype=visible.dtype)), dim=1)
    complete, _ = torch.linalg.qr(seed, mode="complete")
    visible_dim = visible.shape[1]
    return (
        _canonical_signs(complete[:, :visible_dim]),
        _canonical_signs(complete[:, visible_dim:]),
    )


class ProjectorBank(nn.Module):
    def __init__(self, visible_basis: Tensor, complement_basis: Tensor) -> None:
        super().__init__()
        if visible_basis.ndim != 2 or complement_basis.ndim != 2:
            raise ValueError("projector bases must be matrices")
        if visible_basis.shape[0] != complement_basis.shape[0]:
            raise ValueError("projector bases must share the input dimension")
        combined = torch.cat((visible_basis, complement_basis), dim=1)
        if combined.shape[1] != combined.shape[0]:
            raise ValueError("projector bases must form a complete basis")
        identity = torch.eye(combined.shape[0], dtype=combined.dtype)
        if not torch.allclose(combined.T @ combined, identity, atol=1.0e-10, rtol=0.0):
            raise ValueError("projector bases must be orthonormal")
        self.register_buffer("visible_basis", visible_basis.detach().cpu().double(), persistent=True)
        self.register_buffer("complement_basis", complement_basis.detach().cpu().double(), persistent=True)

    @classmethod
    def calibrate(
        cls,
        activations: Tensor | Iterable[Tensor],
        config: ProjectorConfig = ProjectorConfig(),
    ) -> "ProjectorBank":
        samples = _flatten(activations, config.input_axis)
        second_moment = samples.T @ samples / samples.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues.index_select(0, order)
        eigenvectors = eigenvectors.index_select(1, order)
        total_energy = torch.clamp_min(eigenvalues, 0.0).sum()
        if float(total_energy) == 0.0:
            visible = eigenvectors[:, :0]
        else:
            target = config.energy_threshold * total_energy
            rank = int(torch.searchsorted(torch.clamp_min(eigenvalues, 0.0).cumsum(0), target).item()) + 1
            visible = eigenvectors[:, :rank]
        mean = samples.mean(dim=0)
        residual = mean - visible @ (visible.T @ mean)
        if float(torch.linalg.vector_norm(residual)) > config.mean_tolerance * max(float(torch.linalg.vector_norm(mean)), 1.0):
            visible = torch.cat((visible, (residual / torch.linalg.vector_norm(residual)).unsqueeze(1)), dim=1)
        visible, complement = _complete_basis(visible)
        return cls(visible, complement)

    @property
    def input_dim(self) -> int:
        return self.visible_basis.shape[0]

    @property
    def visible_dim(self) -> int:
        return self.visible_basis.shape[1]

    @property
    def complement_dim(self) -> int:
        return self.complement_basis.shape[1]

    def basis(self, route: str, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if route == "visible":
            return self.visible_basis.to(device=device, dtype=dtype)
        if route == "complement":
            return self.complement_basis.to(device=device, dtype=dtype)
        raise ValueError("route must be visible or complement")

    def allocate_rank(self, dense_gradient: Tensor, total_rank: int) -> tuple[int, int]:
        if dense_gradient.ndim != 2 or dense_gradient.shape[1] != self.input_dim:
            raise ValueError("dense_gradient must have the projector input dimension")
        if total_rank < 1 or total_rank > self.input_dim:
            raise ValueError("total_rank must lie in [1, input_dim]")
        gradient = dense_gradient.detach().to(dtype=torch.float64)
        scores = {
            "visible": torch.linalg.svdvals(gradient @ self.visible_basis.to(gradient)).square(),
            "complement": torch.linalg.svdvals(gradient @ self.complement_basis.to(gradient)).square(),
        }
        best: tuple[float, int, int] | None = None
        for visible_rank in range(total_rank + 1):
            complement_rank = total_rank - visible_rank
            if visible_rank > self.visible_dim or complement_rank > self.complement_dim:
                continue
            score = float(scores["visible"][:visible_rank].sum() + scores["complement"][:complement_rank].sum())
            candidate = (score, visible_rank, complement_rank)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            raise ValueError("total_rank cannot be allocated across the projector basis")
        return best[1], best[2]

    def project(self, values: Tensor, route: str) -> Tensor:
        basis = self.basis(route, device=values.device, dtype=values.dtype)
        return values @ basis
