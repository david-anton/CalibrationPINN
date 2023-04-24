from typing import Callable, TypeAlias

import torch
from torch.func import grad, vmap

from parametricpinn.types import Module, Tensor

TModule: TypeAlias = Callable[[Tensor, Tensor], Tensor]


def momentum_equation_func(
    ansatz: Module, x_coor: Tensor, x_param: Tensor, volume_force: Tensor
) -> Tensor:
    _ansatz = _transform_ansatz(ansatz)
    vmap_func = lambda _x_coor, _x_param, _volume_force: _momentum_equation_func(
        _ansatz, _x_coor, _x_param, _volume_force
    )
    return vmap(vmap_func)(x_coor, x_param, volume_force)


def stress_func(ansatz: Module, x_coor: Tensor, x_param: Tensor) -> Tensor:
    _ansatz = _transform_ansatz(ansatz)
    vmap_func = lambda _x_coor, _x_param: _stress_func(_ansatz, _x_coor, _x_param)
    return vmap(vmap_func)(x_coor, x_param)


def _momentum_equation_func(
    ansatz: TModule, x_coor: Tensor, x_param: Tensor, volume_force: Tensor
) -> Tensor:
    x_E = x_param
    return x_E * _u_xx_func(ansatz, x_coor, x_param) + volume_force


def _stress_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    x_E = x_param
    return x_E * _u_x_func(ansatz, x_coor, x_param)


def _u_x_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    return grad(ansatz, argnums=0)(x_coor, x_param)


def _u_xx_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    u_x_func = lambda _x_coor, _x_param: _u_x_func(ansatz, _x_coor, _x_param)[0]
    return grad(u_x_func, argnums=0)(x_coor, x_param)


def _transform_ansatz(ansatz: Module) -> TModule:
    return lambda x_coor, x_param: ansatz(torch.concat((x_coor, x_param)))[0]
