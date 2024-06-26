from typing import Callable, TypeAlias

import torch
from torch.func import grad, jacrev, vmap

from calibrationpinn.types import Module, Tensor

TModule: TypeAlias = Callable[[Tensor, Tensor], Tensor]
StressFuncSingle: TypeAlias = Callable[[TModule, Tensor, Tensor], Tensor]
MomentumFunc: TypeAlias = Callable[[Module, Tensor, Tensor, Tensor], Tensor]
StressFunc: TypeAlias = Callable[[Module, Tensor, Tensor], Tensor]
TractionFunc: TypeAlias = Callable[[Module, Tensor, Tensor, Tensor], Tensor]
TractionEnergyFunc: TypeAlias = Callable[
    [Module, Tensor, Tensor, Tensor, Tensor], Tensor
]


def momentum_equation_func(stress_func: StressFuncSingle) -> MomentumFunc:
    def _func(
        ansatz: Module, x_coors: Tensor, x_params: Tensor, volume_forces: Tensor
    ) -> Tensor:
        _ansatz = _transform_ansatz(ansatz)
        vmap_func = lambda _x_coor, _x_param, _volume_force: _momentum_equation_func(
            _ansatz, _x_coor, _x_param, _volume_force, stress_func
        )
        return vmap(vmap_func)(x_coors, x_params, volume_forces)

    return _func


def stress_func(stress_func: StressFuncSingle) -> StressFunc:
    def _func(ansatz: Module, x_coors: Tensor, x_params: Tensor) -> Tensor:
        _ansatz = _transform_ansatz(ansatz)
        vmap_func = lambda _x_coor, _x_param: stress_func(_ansatz, _x_coor, _x_param)
        return vmap(vmap_func)(x_coors, x_params)

    return _func


def traction_func(stress_func: StressFuncSingle) -> TractionFunc:
    def _func(
        ansatz: Module, x_coors: Tensor, x_params: Tensor, normals: Tensor
    ) -> Tensor:
        _ansatz = _transform_ansatz(ansatz)
        vmap_func = lambda _x_coor, _x_param, _normal: _traction_func(
            _ansatz, _x_coor, _x_param, _normal, stress_func
        )
        return vmap(vmap_func)(x_coors, x_params, normals)

    return _func


def traction_energy_func(stress_func: StressFuncSingle) -> TractionEnergyFunc:
    def _func(
        ansatz: Module,
        x_coors: Tensor,
        x_params: Tensor,
        normals: Tensor,
        area_fractions: Tensor,
    ) -> Tensor:
        _ansatz = _transform_ansatz(ansatz)
        vmap_func = lambda _x_coor, _x_param, _normal: _traction_energy_func(
            _ansatz, _x_coor, _x_param, _normal, stress_func
        )
        traction_energies = vmap(vmap_func)(x_coors, x_params, normals)
        traction_energy_fractions = traction_energies * area_fractions
        return 1 / 2 * torch.sum(traction_energy_fractions)

    return _func


def _momentum_equation_func(
    ansatz: TModule,
    x_coor: Tensor,
    x_param: Tensor,
    volume_force: Tensor,
    stress_func: StressFuncSingle,
) -> Tensor:
    return _divergence_stress_func(ansatz, x_coor, x_param, stress_func) + volume_force


def _traction_func(
    ansatz: TModule,
    x_coor: Tensor,
    x_param: Tensor,
    normal: Tensor,
    stress_func: StressFuncSingle,
) -> Tensor:
    stress = stress_func(ansatz, x_coor, x_param)
    return torch.matmul(stress, normal)


def _traction_energy_func(
    ansatz: TModule,
    x_coor: Tensor,
    x_param: Tensor,
    normal: Tensor,
    stress_func: StressFuncSingle,
) -> Tensor:
    traction = _traction_func(ansatz, x_coor, x_param, normal, stress_func)
    displacements = _displacement_func(ansatz, x_coor, x_param)
    return torch.unsqueeze(torch.einsum("i,i", traction, displacements), dim=0)


def _divergence_stress_func(
    ansatz: TModule, x_coor: Tensor, x_param: Tensor, stress_func: StressFuncSingle
) -> Tensor:
    def _stress_func(x: Tensor, y: Tensor, idx_i: int, idx_j: int):
        return stress_func(
            ansatz,
            torch.concat((torch.unsqueeze(x, dim=0), torch.unsqueeze(y, dim=0))),
            x_param,
        )[idx_i, idx_j]

    stress_xx_x = torch.unsqueeze(
        grad(_stress_func, argnums=0)(x_coor[0], x_coor[1], 0, 0), dim=0
    )
    stress_xy_x = torch.unsqueeze(
        grad(_stress_func, argnums=0)(x_coor[0], x_coor[1], 0, 1), dim=0
    )
    stress_xy_y = torch.unsqueeze(
        grad(_stress_func, argnums=1)(x_coor[0], x_coor[1], 0, 1), dim=0
    )
    stress_yy_y = torch.unsqueeze(
        grad(_stress_func, argnums=1)(x_coor[0], x_coor[1], 1, 1), dim=0
    )
    return torch.concat((stress_xx_x + stress_xy_y, stress_xy_x + stress_yy_y), dim=0)


def jacobian_displacement_func(
    ansatz: TModule, x_coor: Tensor, x_param: Tensor
) -> Tensor:
    displacement_func = lambda _x_coor: _displacement_func(ansatz, _x_coor, x_param)
    return jacrev(displacement_func, argnums=0)(x_coor)


def _displacement_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    return ansatz(x_coor, x_param)


def _transform_ansatz(ansatz: Module) -> TModule:
    return lambda x_coor, x_param: ansatz(torch.concat((x_coor, x_param)))
