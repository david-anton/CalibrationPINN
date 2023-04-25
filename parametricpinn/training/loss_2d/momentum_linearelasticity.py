from typing import Callable, TypeAlias

import torch

from parametricpinn.training.loss_2d.momentumbase import (
    MomentumFunc,
    StressFunc,
    TModule,
    TractionFunc,
    _strain_func,
    momentum_equation_func,
    traction_func,
)
from parametricpinn.types import Tensor

VoigtMaterialTensorFunc: TypeAlias = Callable[[Tensor], Tensor]


def momentum_equation_func_factory(model: str) -> MomentumFunc:
    if model == "plane strain":
        stress_func = _stress_func(_voigt_material_tensor_func_plane_strain)
    elif model == "plane stress":
        stress_func = _stress_func(_voigt_material_tensor_func_plane_stress)
    return momentum_equation_func(stress_func)


def traction_func_factory(model: str) -> TractionFunc:
    if model == "plane strain":
        stress_func = _stress_func(_voigt_material_tensor_func_plane_strain)
    elif model == "plane stress":
        stress_func = _stress_func(_voigt_material_tensor_func_plane_stress)
    return traction_func(stress_func)


def _voigt_material_tensor_func_plane_strain(x_param: Tensor) -> Tensor:
    E = x_param[0]
    nu = x_param[1]
    return (E / ((1.0 + nu) * (1.0 - 2 * nu))) * torch.tensor(
        [
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, (1.0 - 2 * nu) / 2.0],
        ]
    )
    # E = torch.unsqueeze(x_param[0], dim=0)
    # nu = torch.unsqueeze(x_param[1], dim=0)
    # return (E / ((1.0 + nu) * (1.0 - 2 * nu))) * torch.tensor(
    #     [
    #         [1.0 - nu, nu, 0.0],
    #         [nu, 1.0 - nu, 0.0],
    #         [0.0, 0.0, (1.0 - 2 * nu) / 2.0],
    #     ]
    # )


def _voigt_material_tensor_func_plane_stress(x_param: Tensor) -> Tensor:
    E = x_param[0]
    nu = x_param[1]
    return (E / (1.0 - nu**2)) * torch.tensor(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ]
    )
    # E = torch.unsqueeze(x_param[0], dim=0)
    # nu = torch.unsqueeze(x_param[1], dim=0)
    # return (E / (1.0 - nu**2)) * torch.tensor(
    #     [
    #         [1.0, nu, 0.0],
    #         [nu, 1.0, 0.0],
    #         [0.0, 0.0, (1.0 - nu) / 2.0],
    #     ]
    # )


def _stress_func(voigt_material_tensor_func: VoigtMaterialTensorFunc) -> StressFunc:
    voigt_stress_func = _voigt_stress_func(voigt_material_tensor_func)

    def _func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
        voigt_stress = voigt_stress_func(ansatz, x_coor, x_param)
        return torch.tensor(
            [[voigt_stress[0], voigt_stress[2]], [voigt_stress[2], voigt_stress[1]]]
        )
        # return torch.vstack(
        #     (
        #         torch.concat(
        #             (
        #                 torch.unsqueeze(voigt_stress[0], dim=0),
        #                 torch.unsqueeze(voigt_stress[2], dim=0),
        #             )
        #         ),
        #         torch.concat(
        #             (
        #                 torch.unsqueeze(voigt_stress[2], dim=0),
        #                 torch.unsqueeze(voigt_stress[1], dim=0),
        #             )
        #         ),
        #     )
        # )

    return _func


def _voigt_stress_func(
    voigt_material_tensor_func: VoigtMaterialTensorFunc,
) -> StressFunc:
    def _func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
        voigt_material_tensor = voigt_material_tensor_func(x_param)
        voigt_strain = _voigt_strain_func(ansatz, x_coor, x_param)
        return torch.matmul(voigt_material_tensor, voigt_strain)

    return _func


def _voigt_strain_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    strain = _strain_func(ansatz, x_coor, x_param)
    strain_xx = strain[0, 0]
    strain_yy = strain[1, 1]
    strain_xy = strain[0, 1]
    return torch.tensor([strain_xx, strain_yy, 2 * strain_xy])
