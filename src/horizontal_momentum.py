"""Differentiable horizontal-momentum response operator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class HorizontalMomentumConfig:
    """Fixed discretization and closure choices for a two-component flow field."""

    active_levels: tuple[float, ...] | None = (400.0, 500.0, 600.0, 700.0)
    timestep_seconds: float = 6.0 * 60.0 * 60.0
    earth_radius_m: float = 6_371_000.0
    earth_rotation_rad_s: float = 7.2921159e-5
    diffusion_number: float = 0.05
    acceleration_scale_m_s2: float = 1.0e-3
    zonal_acceleration_scales_m_s2: tuple[float, ...] | None = None
    meridional_acceleration_scales_m_s2: tuple[float, ...] | None = None
    level_weights: tuple[float, ...] | None = None
    zonal_equation_weight: float = 0.5
    meridional_equation_weight: float = 0.5
    include_geopotential_gradient: bool = True
    include_spherical_metric_curvature: bool = True
    residual_mode: str = "zero_residual"


@dataclass
class HorizontalMomentumResult:
    """Dimensionless loss components and native-unit diagnostics."""

    loss: torch.Tensor
    zonal: torch.Tensor
    meridional: torch.Tensor
    zonal_residual_energy_m2_s4: torch.Tensor
    meridional_residual_energy_m2_s4: torch.Tensor
    zonal_by_level: torch.Tensor
    meridional_by_level: torch.Tensor
    zonal_residual_energy_by_level_m2_s4: torch.Tensor
    meridional_residual_energy_by_level_m2_s4: torch.Tensor
    eddy_viscosity_m2_s: torch.Tensor
    prediction_zonal_residual_energy_m2_s4: torch.Tensor
    prediction_meridional_residual_energy_m2_s4: torch.Tensor
    prediction_zonal_residual_energy_by_level_m2_s4: torch.Tensor
    prediction_meridional_residual_energy_by_level_m2_s4: torch.Tensor
    target_zonal_residual_energy_m2_s4: torch.Tensor | None
    target_meridional_residual_energy_m2_s4: torch.Tensor | None
    target_zonal_residual_energy_by_level_m2_s4: torch.Tensor | None
    target_meridional_residual_energy_by_level_m2_s4: torch.Tensor | None


def _validate_config(config: HorizontalMomentumConfig) -> None:
    finite_positive = {
        "timestep_seconds": config.timestep_seconds,
        "earth_radius_m": config.earth_radius_m,
        "acceleration_scale_m_s2": config.acceleration_scale_m_s2,
    }
    for name, value in finite_positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(config.earth_rotation_rad_s) or config.earth_rotation_rad_s < 0.0:
        raise ValueError("earth_rotation_rad_s must be finite and nonnegative")
    if not math.isfinite(config.diffusion_number) or config.diffusion_number < 0.0:
        raise ValueError("diffusion_number must be finite and nonnegative")
    equation_weights = (config.zonal_equation_weight, config.meridional_equation_weight)
    if any(not math.isfinite(value) or value <= 0.0 for value in equation_weights):
        raise ValueError("horizontal-momentum equation weights must be finite and positive")
    if not math.isclose(sum(equation_weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("horizontal-momentum equation weights must sum to one")
    if config.residual_mode not in {"zero_residual", "target_residual_match"}:
        raise ValueError("residual_mode must be 'zero_residual' or 'target_residual_match'")


def _positive_vector(
    values: tuple[float, ...] | None,
    *,
    count: int,
    fallback: float,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    source = (fallback,) * count if values is None else values
    result = torch.as_tensor(source, dtype=torch.float32, device=device).flatten()
    if result.numel() != count or not bool(torch.isfinite(result).all()) or not bool((result > 0).all()):
        raise ValueError(f"{name} must contain one finite positive value per active level")
    return result


def _normalization_vectors(
    config: HorizontalMomentumConfig, *, level_count: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zonal_scales = _positive_vector(
        config.zonal_acceleration_scales_m_s2,
        count=level_count,
        fallback=config.acceleration_scale_m_s2,
        name="zonal_acceleration_scales_m_s2",
        device=device,
    )
    meridional_scales = _positive_vector(
        config.meridional_acceleration_scales_m_s2,
        count=level_count,
        fallback=config.acceleration_scale_m_s2,
        name="meridional_acceleration_scales_m_s2",
        device=device,
    )
    level_weights = _positive_vector(
        config.level_weights,
        count=level_count,
        fallback=1.0,
        name="level_weights",
        device=device,
    )
    return zonal_scales, meridional_scales, level_weights / level_weights.sum()


def _level_indices(
    levels: Sequence[float] | torch.Tensor,
    requested: tuple[float, ...] | None,
    *,
    level_count: int,
    device: torch.device,
) -> torch.Tensor:
    coordinate = torch.as_tensor(levels, device=device, dtype=torch.float32).flatten()
    if coordinate.numel() != level_count or not bool(torch.isfinite(coordinate).all()):
        raise ValueError("level coordinates must be finite and match the flow level dimension")
    if requested is None:
        return torch.arange(level_count, device=device)
    indices: list[int] = []
    for level in requested:
        matches = torch.nonzero(
            torch.isclose(coordinate, torch.tensor(level, device=device, dtype=coordinate.dtype)),
            as_tuple=False,
        )
        if matches.numel() != 1:
            raise ValueError(f"required level {level:g} is missing or ambiguous")
        indices.append(int(matches.item()))
    return torch.tensor(indices, device=device, dtype=torch.long)


def _coordinates(
    lat_degrees: Sequence[float] | torch.Tensor,
    lon_degrees: Sequence[float] | torch.Tensor,
    *,
    shape: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lat = torch.deg2rad(torch.as_tensor(lat_degrees, device=device, dtype=torch.float32).flatten())
    lon = torch.deg2rad(torch.as_tensor(lon_degrees, device=device, dtype=torch.float32).flatten())
    if lat.numel() != shape[0] or lon.numel() != shape[1]:
        raise ValueError("latitude and longitude coordinates must match the flow grid")
    lat_steps, lon_steps = torch.diff(lat), torch.diff(lon)
    if (
        not bool(torch.isfinite(lat).all())
        or not bool(torch.isfinite(lon).all())
        or not bool(torch.all(lat_steps > 0) or torch.all(lat_steps < 0))
        or not bool(torch.all(lon_steps > 0) or torch.all(lon_steps < 0))
    ):
        raise ValueError("latitude and longitude coordinates must be finite and strictly monotonic")
    cosine = torch.cos(lat[1:-1])
    if lat.numel() < 3 or lon.numel() < 3 or not bool(torch.all(cosine > 0)):
        raise ValueError("the interior grid must contain at least 3x3 non-polar points")
    weights = cosine.view(1, 1, 1, -1, 1)
    return lat, lon, weights


def _weighted_mean_by_level(value: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    expanded = weights.expand_as(value)
    reduction_dims = (0, 1, 3, 4)
    return (value * expanded).sum(dim=reduction_dims) / expanded.sum(dim=reduction_dims)


def _horizontal_operators(
    value: torch.Tensor,
    lat: torch.Tensor,
    lon: torch.Tensor,
    earth_radius_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return centered spherical-grid derivatives and a scalar Laplacian."""

    center = value[..., 1:-1, 1:-1]
    lat_center = lat[1:-1]
    cosine = torch.cos(lat_center).view(1, 1, 1, -1, 1)
    tangent = torch.tan(lat_center).view(1, 1, 1, -1, 1)

    d_dlon = (value[..., 1:-1, 2:] - value[..., 1:-1, :-2]) / (
        lon[2:] - lon[:-2]
    ).view(1, 1, 1, 1, -1)
    d_dlat = (value[..., 2:, 1:-1] - value[..., :-2, 1:-1]) / (
        lat[2:] - lat[:-2]
    ).view(1, 1, 1, -1, 1)

    lon_left = (lon[1:-1] - lon[:-2]).view(1, 1, 1, 1, -1)
    lon_right = (lon[2:] - lon[1:-1]).view(1, 1, 1, 1, -1)
    d2_dlon2 = 2.0 * (
        (value[..., 1:-1, 2:] - center) / lon_right
        - (center - value[..., 1:-1, :-2]) / lon_left
    ) / (lon_left + lon_right)

    lat_lower = (lat[1:-1] - lat[:-2]).view(1, 1, 1, -1, 1)
    lat_upper = (lat[2:] - lat[1:-1]).view(1, 1, 1, -1, 1)
    d2_dlat2 = 2.0 * (
        (value[..., 2:, 1:-1] - center) / lat_upper
        - (center - value[..., :-2, 1:-1]) / lat_lower
    ) / (lat_lower + lat_upper)

    dx = d_dlon / (earth_radius_m * cosine)
    dy = d_dlat / earth_radius_m
    laplacian = (d2_dlat2 - tangent * d_dlat + d2_dlon2 / cosine.square()) / (
        earth_radius_m**2
    )
    return center, dx, dy, laplacian


def _current_and_prediction(
    current_vars: Mapping[str, torch.Tensor],
    predicted_vars: Mapping[str, torch.Tensor],
    config: HorizontalMomentumConfig,
    *,
    output_device: torch.device | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    required = ["u", "v"] + (["z"] if config.include_geopotential_gradient else [])
    if any(name not in current_vars or name not in predicted_vars for name in required):
        missing = ", ".join(
            name for name in required if name not in current_vars or name not in predicted_vars
        )
        raise KeyError(f"missing horizontal-momentum variables: {missing}")
    device = predicted_vars["u"].device if output_device is None else output_device
    predicted = {
        name: predicted_vars[name].to(device=device, dtype=torch.float32)
        for name in required
    }
    current = {
        name: current_vars[name][:, -1:].detach().to(device=device, dtype=torch.float32)
        for name in required
    }
    reference = predicted["u"]
    if reference.ndim != 5 or reference.shape[1] != 1:
        raise ValueError("predicted flow variables must have [batch, 1, level, lat, lon] shape")
    if any(value.shape != reference.shape for value in predicted.values()) or any(
        value.shape != reference.shape for value in current.values()
    ):
        raise ValueError("current and predicted horizontal-momentum variables must share shape")
    if any(not bool(torch.isfinite(value).all()) for value in (*current.values(), *predicted.values())):
        raise ValueError("horizontal-momentum variables must be finite")
    return current, predicted


def _residuals(
    current: Mapping[str, torch.Tensor],
    prediction: Mapping[str, torch.Tensor],
    *,
    indices: torch.Tensor,
    lat: torch.Tensor,
    lon: torch.Tensor,
    config: HorizontalMomentumConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_current = {name: value.index_select(2, indices) for name, value in current.items()}
    selected_prediction = {name: value.index_select(2, indices) for name, value in prediction.items()}
    midpoint = {
        name: 0.5 * (selected_current[name] + selected_prediction[name])
        for name in selected_prediction
    }
    u_center, du_dx, du_dy, laplacian_u = _horizontal_operators(
        midpoint["u"], lat, lon, config.earth_radius_m
    )
    v_center, dv_dx, dv_dy, laplacian_v = _horizontal_operators(
        midpoint["v"], lat, lon, config.earth_radius_m
    )
    tendency_u = (
        (selected_prediction["u"] - selected_current["u"]) / config.timestep_seconds
    )[..., 1:-1, 1:-1]
    tendency_v = (
        (selected_prediction["v"] - selected_current["v"]) / config.timestep_seconds
    )[..., 1:-1, 1:-1]
    coriolis = (
        2.0
        * config.earth_rotation_rad_s
        * torch.sin(lat[1:-1]).view(1, 1, 1, -1, 1)
    )
    median_dy = config.earth_radius_m * torch.median(torch.diff(lat).abs())
    median_dx = torch.median(
        config.earth_radius_m
        * torch.cos(lat[1:-1])
        * torch.median(torch.diff(lon).abs())
    )
    grid_spacing_squared = 0.5 * (median_dx.square() + median_dy.square())
    viscosity = config.diffusion_number * grid_spacing_squared / config.timestep_seconds

    pressure_x = torch.zeros_like(u_center)
    pressure_y = torch.zeros_like(v_center)
    if config.include_geopotential_gradient:
        _, pressure_x, pressure_y, _ = _horizontal_operators(
            midpoint["z"], lat, lon, config.earth_radius_m
        )
    curvature_u = torch.zeros_like(u_center)
    curvature_v = torch.zeros_like(v_center)
    if config.include_spherical_metric_curvature:
        tangent_over_radius = torch.tan(lat[1:-1]).view(1, 1, 1, -1, 1) / config.earth_radius_m
        curvature_u = -u_center * v_center * tangent_over_radius
        curvature_v = u_center.square() * tangent_over_radius
    residual_u = (
        tendency_u + u_center * du_dx + v_center * du_dy + curvature_u - coriolis * v_center
        + pressure_x - viscosity * laplacian_u
    )
    residual_v = (
        tendency_v + u_center * dv_dx + v_center * dv_dy + curvature_v + coriolis * u_center
        + pressure_y - viscosity * laplacian_v
    )
    return residual_u, residual_v, viscosity


def compute_horizontal_momentum(
    current_vars: Mapping[str, torch.Tensor],
    predicted_vars: Mapping[str, torch.Tensor],
    *,
    lat_degrees: Sequence[float] | torch.Tensor,
    lon_degrees: Sequence[float] | torch.Tensor,
    levels: Sequence[float] | torch.Tensor,
    config: HorizontalMomentumConfig = HorizontalMomentumConfig(),
    target_vars: Mapping[str, torch.Tensor] | None = None,
) -> HorizontalMomentumResult:
    """Compute a midpoint horizontal-momentum response on a spherical grid."""

    _validate_config(config)
    current, predicted = _current_and_prediction(current_vars, predicted_vars, config)
    if config.residual_mode == "target_residual_match" and target_vars is None:
        raise ValueError("target_vars is required when residual_mode is 'target_residual_match'")
    device = predicted["u"].device
    indices = _level_indices(
        levels,
        config.active_levels,
        level_count=predicted["u"].shape[2],
        device=device,
    )
    lat, lon, weights = _coordinates(
        lat_degrees,
        lon_degrees,
        shape=(predicted["u"].shape[-2], predicted["u"].shape[-1]),
        device=device,
    )

    prediction_residual_u, prediction_residual_v, viscosity = _residuals(
        current, predicted, indices=indices, lat=lat, lon=lon, config=config
    )
    target_residual_u = target_residual_v = None
    if target_vars is not None:
        _, target = _current_and_prediction(
            current_vars, target_vars, config, output_device=device
        )
        target = {name: value.detach() for name, value in target.items()}
        target_residual_u, target_residual_v, _ = _residuals(
            current, target, indices=indices, lat=lat, lon=lon, config=config
        )
        target_residual_u = target_residual_u.detach()
        target_residual_v = target_residual_v.detach()
    if config.residual_mode == "target_residual_match":
        assert target_residual_u is not None and target_residual_v is not None
        residual_u = prediction_residual_u - target_residual_u
        residual_v = prediction_residual_v - target_residual_v
    else:
        residual_u = prediction_residual_u
        residual_v = prediction_residual_v
    residual_u_energy_by_level = _weighted_mean_by_level(residual_u.square(), weights)
    residual_v_energy_by_level = _weighted_mean_by_level(residual_v.square(), weights)
    prediction_residual_u_energy_by_level = _weighted_mean_by_level(prediction_residual_u.square(), weights)
    prediction_residual_v_energy_by_level = _weighted_mean_by_level(prediction_residual_v.square(), weights)
    zonal_scales, meridional_scales, level_weights = _normalization_vectors(
        config, level_count=indices.numel(), device=device
    )
    zonal_by_level = residual_u_energy_by_level / zonal_scales.square()
    meridional_by_level = residual_v_energy_by_level / meridional_scales.square()
    zonal = torch.sum(level_weights * zonal_by_level)
    meridional = torch.sum(level_weights * meridional_by_level)
    residual_u_energy = torch.sum(level_weights * residual_u_energy_by_level)
    residual_v_energy = torch.sum(level_weights * residual_v_energy_by_level)
    prediction_residual_u_energy = torch.sum(level_weights * prediction_residual_u_energy_by_level)
    prediction_residual_v_energy = torch.sum(level_weights * prediction_residual_v_energy_by_level)
    target_residual_u_energy_by_level = (
        None if target_residual_u is None else _weighted_mean_by_level(target_residual_u.square(), weights)
    )
    target_residual_v_energy_by_level = (
        None if target_residual_v is None else _weighted_mean_by_level(target_residual_v.square(), weights)
    )
    target_residual_u_energy = (
        None if target_residual_u_energy_by_level is None else torch.sum(level_weights * target_residual_u_energy_by_level)
    )
    target_residual_v_energy = (
        None if target_residual_v_energy_by_level is None else torch.sum(level_weights * target_residual_v_energy_by_level)
    )
    components = (zonal, meridional, viscosity)
    if any(not bool(torch.isfinite(component.detach())) for component in components):
        raise FloatingPointError("horizontal-momentum objective produced a non-finite value")
    return HorizontalMomentumResult(
        loss=config.zonal_equation_weight * zonal + config.meridional_equation_weight * meridional,
        zonal=zonal,
        meridional=meridional,
        zonal_residual_energy_m2_s4=residual_u_energy,
        meridional_residual_energy_m2_s4=residual_v_energy,
        zonal_by_level=zonal_by_level,
        meridional_by_level=meridional_by_level,
        zonal_residual_energy_by_level_m2_s4=residual_u_energy_by_level,
        meridional_residual_energy_by_level_m2_s4=residual_v_energy_by_level,
        eddy_viscosity_m2_s=viscosity.detach(),
        prediction_zonal_residual_energy_m2_s4=prediction_residual_u_energy,
        prediction_meridional_residual_energy_m2_s4=prediction_residual_v_energy,
        prediction_zonal_residual_energy_by_level_m2_s4=prediction_residual_u_energy_by_level,
        prediction_meridional_residual_energy_by_level_m2_s4=prediction_residual_v_energy_by_level,
        target_zonal_residual_energy_m2_s4=target_residual_u_energy,
        target_meridional_residual_energy_m2_s4=target_residual_v_energy,
        target_zonal_residual_energy_by_level_m2_s4=target_residual_u_energy_by_level,
        target_meridional_residual_energy_by_level_m2_s4=target_residual_v_energy_by_level,
    )
