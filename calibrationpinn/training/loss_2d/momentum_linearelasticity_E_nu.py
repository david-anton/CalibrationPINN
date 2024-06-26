from typing import Callable, TypeAlias

import torch

from calibrationpinn.training.loss_2d.momentumbase import (
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
from calibrationpinn.training.loss_2d.momentumbase_linearelasticity import (
    StrainEnergyFunc,
    _strain_func,
    calculate_E_from_K_and_G_factory,
    calculate_G_from_E_and_nu,
    calculate_K_from_E_and_nu_factory,
    calculate_nu_from_K_and_G_factory,
    strain_energy_func,
)
from calibrationpinn.types import Tensor

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
        stress_func = _stress_func(_voigt_material_tensor_func_plane_strain)
    elif model == "plane stress":
        stress_func = _stress_func(_voigt_material_tensor_func_plane_stress)
    return stress_func


def _voigt_material_tensor_func_plane_strain(x_param: Tensor) -> Tensor:
    E = torch.unsqueeze(x_param[0], dim=0)
    nu = torch.unsqueeze(x_param[1], dim=0)
    return (E / ((1.0 + nu) * (1.0 - 2 * nu))) * torch.stack(
        (
            torch.concat((1.0 - nu, nu, torch.tensor([0.0]).to(x_param.device)), dim=0),
            torch.concat((nu, 1.0 - nu, torch.tensor([0.0]).to(x_param.device)), dim=0),
            torch.concat(
                (
                    torch.tensor([0.0]).to(x_param.device),
                    torch.tensor([0.0]).to(x_param.device),
                    (1.0 - 2 * nu) / 2.0,
                ),
                dim=0,
            ),
        ),
        dim=0,
    )


def _voigt_material_tensor_func_plane_stress(x_param: Tensor) -> Tensor:
    E = torch.unsqueeze(x_param[0], dim=0)
    nu = torch.unsqueeze(x_param[1], dim=0)
    return (E / (1.0 - nu**2)) * torch.stack(
        (
            torch.concat(
                (
                    torch.tensor([1.0]).to(x_param.device),
                    nu,
                    torch.tensor([0.0]).to(x_param.device),
                ),
                dim=0,
            ),
            torch.concat(
                (
                    nu,
                    torch.tensor([1.0]).to(x_param.device),
                    torch.tensor([0.0]).to(x_param.device),
                ),
                dim=0,
            ),
            torch.concat(
                (
                    torch.tensor([0.0]).to(x_param.device),
                    torch.tensor([0.0]).to(x_param.device),
                    (1.0 - nu) / 2.0,
                ),
                dim=0,
            ),
        ),
        dim=0,
    )


def _stress_func(
    voigt_material_tensor_func: VoigtMaterialTensorFunc,
) -> StressFuncSingle:
    voigt_stress_func = _voigt_stress_func(voigt_material_tensor_func)

    def _func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
        voigt_stress = voigt_stress_func(ansatz, x_coor, x_param)
        voigt_stress_xx = torch.unsqueeze(voigt_stress[0], dim=0)
        voigt_stress_xy = torch.unsqueeze(voigt_stress[2], dim=0)
        voigt_stress_yy = torch.unsqueeze(voigt_stress[1], dim=0)
        return torch.stack(
            (
                torch.concat((voigt_stress_xx, voigt_stress_xy), dim=0),
                torch.concat((voigt_stress_xy, voigt_stress_yy), dim=0),
            ),
            dim=0,
        )

    return _func


def _voigt_stress_func(
    voigt_material_tensor_func: VoigtMaterialTensorFunc,
) -> StressFuncSingle:
    def _func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
        voigt_material_tensor = voigt_material_tensor_func(x_param)
        voigt_strain = _voigt_strain_func(ansatz, x_coor, x_param)
        return torch.matmul(voigt_material_tensor, voigt_strain)

    return _func


def _voigt_strain_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    strain = _strain_func(ansatz, x_coor, x_param)
    strain_xx = torch.unsqueeze(strain[0, 0], dim=0)
    strain_yy = torch.unsqueeze(strain[1, 1], dim=0)
    strain_xy = torch.unsqueeze(strain[0, 1], dim=0)
    return torch.concat((strain_xx, strain_yy, 2 * strain_xy), dim=0)
