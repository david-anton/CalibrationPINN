from typing import Callable, TypeAlias

import torch
from torch.func import grad, jacrev, vmap

from parametricpinn.types import Module, Tensor

TModule: TypeAlias = Callable[[Tensor, Tensor], Tensor]


def _first_piola_stress_tensor(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    F = _deformation_gradient_func(ansatz, x_coor, x_param)
    free_energy_func = lambda _deformation_gradient: _free_energy_func(_deformation_gradient, x_param)
    return 


def _free_energy_func(deformation_gradient: Tensor, x_param: Tensor) -> Tensor:
    F = deformation_gradient
    J = _calculate_determinant(F)
    I_c = _right_cuachy_green_tensor_func(F)
    lambda_ = _first_lame_constant_lambda(x_param)
    mu_ = _second_lame_constant_mu(x_param)
    C = mu_ / 2
    D = lambda_ / 2
    return C * (I_c - 2 - 2 * torch.log(J)) + D * (J - 1) ** 2


def _calculate_determinant(tensor: Tensor) -> Tensor:
    return torch.unsqueeze(torch.det(tensor), dim=0)


def _first_lame_constant_lambda(x_param: Tensor) -> Tensor:
    E = _extract_youngs_modulus(x_param)
    nu = _extract_poissons_ratio(x_param)
    return (
        E
        * nu
        / (torch.tensor([1.0]).to(x_param.device) + nu)
        * (
            torch.tensor([1.0]).to(x_param.device)
            - torch.tensor([2.0]).to(x_param.device) * nu
        )
    )


def _second_lame_constant_mu(x_param: Tensor) -> Tensor:
    E = _extract_youngs_modulus(x_param)
    nu = _extract_poissons_ratio(x_param)
    return (
        E
        / torch.tensor([2.0]).to(x_param.device)
        * (torch.tensor([1.0]).to(x_param.device) + nu)
    )


def _extract_youngs_modulus(x_param: Tensor) -> Tensor:
    return torch.unsqueeze(x_param[0], dim=0)


def _extract_poissons_ratio(x_param: Tensor) -> Tensor:
    return torch.unsqueeze(x_param[1], dim=0)


def _right_cuachy_green_tensor_func(deformation_gradient: Tensor) -> Tensor:
    F = deformation_gradient
    return torch.matmul(F.T, F)


def _deformation_gradient_func(
    ansatz: TModule, x_coor: Tensor, x_param: Tensor
) -> Tensor:
    jac_u = _jacobian_displacement_func(ansatz, x_coor, x_param)
    I = torch.eye(n=2)
    return jac_u + I


def _jacobian_displacement_func(
    ansatz: TModule, x_coor: Tensor, x_param: Tensor
) -> Tensor:
    displacement_func = lambda _x_coor: _displacement_func(ansatz, _x_coor, x_param)
    return jacrev(displacement_func, argnums=0)(x_coor)


def _displacement_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    return ansatz(x_coor, x_param)


def _transform_ansatz(ansatz: Module) -> TModule:
    return lambda x_coor, x_param: ansatz(torch.concat((x_coor, x_param)))
