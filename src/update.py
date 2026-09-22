from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer


@dataclass(frozen=True)
class ProjectionResult:
    direction: Tensor
    data_derivative: Tensor
    physics_derivative: Tensor
    correction_norm: Tensor


def _dot(left: Tensor, right: Tensor) -> Tensor:
    return torch.dot(left.double(), right.double())


def _validate(proposal: Tensor, data_gradient: Tensor, physics_gradient: Tensor) -> None:
    if proposal.ndim != 1 or proposal.shape != data_gradient.shape or proposal.shape != physics_gradient.shape:
        raise ValueError("projection inputs must be equal-shaped flat tensors")
    if not all(value.is_floating_point() for value in (proposal, data_gradient, physics_gradient)):
        raise ValueError("projection inputs must be floating-point tensors")
    if not all(bool(torch.isfinite(value).all()) for value in (proposal, data_gradient, physics_gradient)):
        raise ValueError("projection inputs must be finite")


def project_to_joint_cone(
    proposal: Tensor,
    data_gradient: Tensor,
    physics_gradient: Tensor,
) -> ProjectionResult:
    _validate(proposal, data_gradient, physics_gradient)
    data_norm = _dot(data_gradient, data_gradient)
    physics_norm = _dot(physics_gradient, physics_gradient)
    data_dot = _dot(data_gradient, proposal)
    physics_dot = _dot(physics_gradient, proposal)
    candidates: list[tuple[Tensor, Tensor]] = [(proposal, torch.zeros((), dtype=torch.float64, device=proposal.device))]

    if float(data_norm) > 0.0:
        coefficient = torch.clamp_min(-data_dot / data_norm, 0.0)
        candidates.append((proposal + coefficient.to(proposal.dtype) * data_gradient, coefficient))
    if float(physics_norm) > 0.0:
        coefficient = torch.clamp_min(-physics_dot / physics_norm, 0.0)
        candidates.append((proposal + coefficient.to(proposal.dtype) * physics_gradient, coefficient))

    if float(data_norm) > 0.0 and float(physics_norm) > 0.0:
        cross = _dot(data_gradient, physics_gradient)
        gram = torch.stack((torch.stack((data_norm, cross)), torch.stack((cross, physics_norm))))
        rhs = torch.stack((-data_dot, -physics_dot))
        determinant = torch.linalg.det(gram)
        if float(torch.abs(determinant)) > torch.finfo(torch.float64).eps:
            coefficients = torch.linalg.solve(gram, rhs)
            if bool(torch.all(coefficients >= 0.0)):
                direction = proposal + coefficients[0].to(proposal.dtype) * data_gradient + coefficients[1].to(proposal.dtype) * physics_gradient
                candidates.append((direction, torch.linalg.vector_norm(direction - proposal).double()))

    valid: list[tuple[Tensor, Tensor]] = []
    tolerance = 256.0 * torch.finfo(torch.float64).eps
    for direction, _ in candidates:
        if float(_dot(data_gradient, direction)) >= -tolerance and float(_dot(physics_gradient, direction)) >= -tolerance:
            valid.append((direction, torch.linalg.vector_norm(direction - proposal)))
    if not valid:
        raise FloatingPointError("joint cone projection produced no feasible candidate")
    direction, correction_norm = min(valid, key=lambda item: float(item[1]))
    return ProjectionResult(
        direction=direction,
        data_derivative=_dot(data_gradient, direction).to(proposal.dtype).detach(),
        physics_derivative=_dot(physics_gradient, direction).to(proposal.dtype).detach(),
        correction_norm=correction_norm.detach(),
    )


def _flatten(parameters: Iterable[Tensor]) -> Tensor:
    values = [parameter.detach().reshape(-1) for parameter in parameters]
    if not values:
        raise ValueError("at least one trainable parameter is required")
    return torch.cat(values)


def _write(parameters: Iterable[Tensor], values: Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.copy_(values[offset : offset + count].reshape_as(parameter))
            offset += count


def corrected_optimizer_step(
    optimizer: Optimizer,
    parameters: Iterable[Tensor],
    data_gradient: Tensor,
    physics_gradient: Tensor,
) -> ProjectionResult:
    parameter_list = list(parameters)
    before = _flatten(parameter_list)
    optimizer.step()
    after = _flatten(parameter_list)
    proposal = before - after
    result = project_to_joint_cone(proposal, data_gradient, physics_gradient)
    _write(parameter_list, before - result.direction)
    return result
