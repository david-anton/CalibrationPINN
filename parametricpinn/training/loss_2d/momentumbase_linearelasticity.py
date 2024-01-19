from typing import Callable, TypeAlias

import torch
from torch.func import vmap

from parametricpinn.training.loss_2d.momentumbase import (
    StressFuncSingle,
    TModule,
    _transform_ansatz,
    jacobian_displacement_func,
)
from parametricpinn.types import Module, Tensor

StrainFunc: TypeAlias = Callable[[TModule, Tensor, Tensor], Tensor]
StrainEnergyFunc: TypeAlias = Callable[[Module, Tensor, Tensor, Tensor], Tensor]


def strain_energy_func(stress_func: StressFuncSingle) -> StrainEnergyFunc:
    def _func(
        ansatz: Module, x_coors: Tensor, x_params: Tensor, area: Tensor
    ) -> Tensor:
        _ansatz = _transform_ansatz(ansatz)
        vmap_func = lambda _x_coor, _x_param: _strain_energy_func(
            _ansatz, _x_coor, _x_param, stress_func
        )
        strain_energies = vmap(vmap_func)(x_coors, x_params)
        num_collocation_points = x_coors.size(dim=0)
        return 1 / 2 * (area / num_collocation_points) * torch.sum(strain_energies)

    return _func


def _strain_energy_func(
    ansatz: TModule,
    x_coor: Tensor,
    x_param: Tensor,
    stress_func: StressFuncSingle,
) -> Tensor:
    stress = stress_func(ansatz, x_coor, x_param)
    strain = _strain_func(ansatz, x_coor, x_param)
    return torch.unsqueeze(torch.einsum("ij,ij", stress, strain), dim=0)


def _strain_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    jac_u = jacobian_displacement_func(ansatz, x_coor, x_param)
    return 1 / 2 * (jac_u + torch.transpose(jac_u, 0, 1))
