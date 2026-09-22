from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def normalized_state_loss(prediction: Tensor, target: Tensor, scale: Tensor | None = None) -> Tensor:
    error = prediction - target
    if scale is not None:
        error = error / scale
    return error.square().mean()


def observed_response_loss(
    operator: Callable[[object, Tensor], Tensor],
    current: object,
    prediction: Tensor,
    target: Tensor,
    scale: Tensor | float = 1.0,
) -> Tensor:
    predicted_response = operator(current, prediction)
    with torch.no_grad():
        observed_response = operator(current, target)
    return ((predicted_response - observed_response) / scale).square().mean()
