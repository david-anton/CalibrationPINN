from typing import Callable, TypeAlias

import torch

from parametricpinn.training.loss_2d.momentumbase import (
    MomentumFunc,
    StressFunc,
    StressFuncSingle,
    TModule,
    TractionEnergyFunc,
    TractionFunc,
    momentum_equation_func,
    stress_func,
    traction_energy_func,
    traction_func,
)
from parametricpinn.training.loss_2d.momentumbase_linearelasticity import (
    StrainEnergyFunc,
    _strain_func,
    calculate_E_from_K_and_G_factory,
    calculate_G_from_E_and_nu,
    calculate_K_from_E_and_nu_factory,
    calculate_nu_from_K_and_G_factory,
    strain_energy_func,
)
from parametricpinn.types import Tensor

VoigtMaterialTensorFunc: TypeAlias = Callable[[Tensor], Tensor]


def momentum_equation_func_factory(model: str) -> MomentumFunc:
    stress_func_single_input = _get_stress_func(model)
    return momentum_equation_func(stress_func_single_input)


def stress_func_factory(model: str) -> StressFunc:
    stress_func_single_input = _get_stress_func(model)
    return stress_func(stress_func_single_input)


def traction_func_factory(model: str) -> TractionFunc:
    stress_func_single_input = _get_stress_func(model)
    return traction_func(stress_func_single_input)


def strain_energy_func_factory(model: str) -> StrainEnergyFunc:
    stress_func_single_input = _get_stress_func(model)
    return strain_energy_func(stress_func_single_input)


def traction_energy_func_factory(model: str) -> TractionEnergyFunc:
    stress_func_single_input = _get_stress_func(model)
    return traction_energy_func(stress_func_single_input)


def _get_stress_func(model: str) -> StressFuncSingle:
    if model == "plane strain":
        stress_func = _stress_func_plane_strain
    elif model == "plane stress":
        stress_func = _stress_func_plane_stress
    return stress_func


def _stress_func_plane_strain(
    ansatz: TModule, x_coor: Tensor, x_param: Tensor
) -> Tensor:
    K = torch.unsqueeze(x_param[0], dim=0)
    G = torch.unsqueeze(x_param[1], dim=0)
    volumetric_strain = _volumetric_strain_func(ansatz, x_coor, x_param)
    deviatoric_strain = _deviatoric_strain_func(ansatz, x_coor, x_param)
    return K * volumetric_strain + 2 * G * deviatoric_strain


def _volumetric_strain_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    strain = _strain_func(ansatz, x_coor, x_param)
    trace_strain = torch.trace(strain)
    identity = torch.eye(2)
    return trace_strain * identity


def _deviatoric_strain_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    strain = _strain_func(ansatz, x_coor, x_param)
    volumetric_strain = _volumetric_strain_func(ansatz, x_coor, x_param)
    return strain - (volumetric_strain / 2)


def _stress_func_plane_stress(
    ansatz: TModule, x_coor: Tensor, x_param: Tensor
) -> Tensor:
    K = torch.unsqueeze(x_param[0], dim=0)
    G = torch.unsqueeze(x_param[1], dim=0)
    strain = _strain_func(ansatz, x_coor, x_param)
    eps_xx = torch.unsqueeze(strain[0, 0], dim=0)
    eps_yy = torch.unsqueeze(strain[1, 1], dim=0)
    eps_xy = torch.unsqueeze(strain[0, 1], dim=0)
    sig_xx = (2 * G / (3 * K + 4 * G)) * (
        (6 * K + 2 * G) * eps_xx + (3 * K - 2 * G) * eps_yy
    )
    sig_yy = (2 * G / (3 * K + 4 * G)) * (
        (3 * K - 2 * G) * eps_xx + (6 * K + 2 * G) * eps_yy
    )
    sig_xy = G * 2 * eps_xy
    return torch.stack(
        (
            torch.concat((sig_xx, sig_xy), dim=0),
            torch.concat((sig_xy, sig_yy), dim=0),
        ),
        dim=0,
    )
