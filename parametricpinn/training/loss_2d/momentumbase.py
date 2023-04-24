from typing import Callable, TypeAlias

import torch
from torch.func import grad, jacrev, vmap

from parametricpinn.types import Module, Tensor

TModule: TypeAlias = Callable[[Tensor, Tensor], Tensor]
StressFunc: TypeAlias = Callable[[TModule, Tensor, Tensor], Tensor]
MomentumFunc: TypeAlias = Callable[[Module, Tensor, Tensor, Tensor], Tensor]
TractionFunc: TypeAlias = Callable[[Module, Tensor, Tensor, Tensor], Tensor]


def momentum_equation_func(stress_func: StressFunc) -> MomentumFunc:
    def _func(
        ansatz: Module, x_coors: Tensor, x_params: Tensor, volume_forces: Tensor
    ) -> Tensor:
        _ansatz = _transform_ansatz(ansatz)
        vmap_func = lambda _x_coor, _x_param, _volume_force: _momentum_equation_func(
            _ansatz, _x_coor, _x_param, _volume_force, stress_func
        )
        return vmap(vmap_func)(x_coors, x_params, volume_forces)

    return _func


def traction_func(stress_func: StressFunc) -> TractionFunc:
    def _func(
        ansatz: Module, x_coors: Tensor, x_params: Tensor, normals: Tensor
    ) -> Tensor:
        _ansatz = _transform_ansatz(ansatz)
        vmap_func = lambda _x_coor, _x_param, _normal: _traction_func(
            _ansatz, _x_coor, _x_param, _normal, stress_func
        )
        return vmap(vmap_func)(x_coors, x_params, normals)

    return _func


def _momentum_equation_func(
    ansatz: TModule,
    x_coor: Tensor,
    x_param: Tensor,
    volume_force: Tensor,
    stress_func: StressFunc,
) -> Tensor:
    return _divergence_stress_func(ansatz, x_coor, x_param, stress_func) + volume_force


def _traction_func(
    ansatz: TModule,
    x_coor: Tensor,
    x_param: Tensor,
    normal: Tensor,
    stress_func: StressFunc,
) -> Tensor:
    stress = stress_func(ansatz, x_coor, x_param)
    return torch.matmul(stress, normal)


def _divergence_stress_func(
    ansatz: TModule, x_coor: Tensor, x_param: Tensor, stress_func: StressFunc
) -> Tensor:
    def _stress_func(x: Tensor, y: Tensor, idx_x: int, idx_y: int):
        return stress_func(ansatz, torch.tensor((x, y)), x_param)[idx_x, idx_y]

    stress_xx_x = grad(_stress_func, argnums=0)(x_coor[0], x_coor[0], 0, 0)
    stress_xy_x = grad(_stress_func, argnums=0)(x_coor[0], x_coor[0], 0, 1)
    stress_xy_y = grad(_stress_func, argnums=1)(x_coor[0], x_coor[0], 0, 1)
    stress_yy_y = grad(_stress_func, argnums=1)(x_coor[0], x_coor[0], 1, 1)
    return torch.tensor([stress_xx_x + stress_xy_y, stress_xy_x + stress_yy_y])


def _strain_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    jac_u = _jacobian_displacement_func(ansatz, x_coor, x_param)
    return 1 / 2 * (jac_u + torch.transpose(jac_u, 0, 1))


def _jacobian_displacement_func(
    ansatz: TModule, x_coor: Tensor, x_param: Tensor
) -> Tensor:
    return jacrev(_displacement_func, argnums=1)(ansatz, x_coor, x_param)


def _displacement_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    return ansatz(x_coor, x_param)


def _transform_ansatz(ansatz: Module) -> TModule:
    return lambda x_coor, x_param: ansatz(torch.concat((x_coor, x_param)))[0]
