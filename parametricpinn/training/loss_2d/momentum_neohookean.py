from typing import Callable, TypeAlias

import torch
from torch.func import grad, jacrev, vmap

from parametricpinn.types import Module, Tensor

TModule: TypeAlias = Callable[[Tensor, Tensor], Tensor]
MomentumFunc: TypeAlias = Callable[[Module, Tensor, Tensor, Tensor], Tensor]
StressFunc: TypeAlias = Callable[[Module, Tensor, Tensor], Tensor]
TractionFunc: TypeAlias = Callable[[Module, Tensor, Tensor, Tensor], Tensor]


def momentum_equation_func_factory() -> MomentumFunc:
    def _func(
        ansatz: Module, x_coors: Tensor, x_params: Tensor, volume_forces: Tensor
    ) -> Tensor:
        _ansatz = _transform_ansatz(ansatz)
        vmap_func = lambda _x_coor, _x_param, _volume_force: _momentum_equation_func(
            _ansatz, _x_coor, _x_param, _volume_force
        )
        return vmap(vmap_func)(x_coors, x_params, volume_forces)

    return _func


def stress_func_factory() -> StressFunc:
    def _func(ansatz: Module, x_coors: Tensor, x_params: Tensor) -> Tensor:
        _ansatz = _transform_ansatz(ansatz)
        vmap_func = lambda _x_coor, _x_param: _first_piola_stress_tensor(
            _ansatz, _x_coor, _x_param
        )
        return vmap(vmap_func)(x_coors, x_params)

    return _func


def traction_func_factory() -> TractionFunc:
    def _func(
        ansatz: Module, x_coors: Tensor, x_params: Tensor, normals: Tensor
    ) -> Tensor:
        _ansatz = _transform_ansatz(ansatz)
        vmap_func = lambda _x_coor, _x_param, _normal: _traction_func(
            _ansatz, _x_coor, _x_param, _normal
        )
        return vmap(vmap_func)(x_coors, x_params, normals)

    return _func


def _momentum_equation_func(
    ansatz: TModule,
    x_coor: Tensor,
    x_param: Tensor,
    volume_force: Tensor,
) -> Tensor:
    return _divergence_stress_func(ansatz, x_coor, x_param) + volume_force


def _divergence_stress_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    def _stress_func(x: Tensor, y: Tensor, idx_i: int, idx_j: int):
        return _first_piola_stress_tensor(
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


def _traction_func(
    ansatz: TModule,
    x_coor: Tensor,
    x_param: Tensor,
    normal: Tensor,
) -> Tensor:
    stress = _first_piola_stress_tensor(ansatz, x_coor, x_param)
    return torch.matmul(stress, normal)


def _first_piola_stress_tensor(
    ansatz: TModule, x_coor: Tensor, x_param: Tensor
) -> Tensor:
    F = _deformation_gradient_func(ansatz, x_coor, x_param)
    free_energy_func = lambda _deformation_gradient: _free_energy_func(
        _deformation_gradient, x_param
    )
    return grad(free_energy_func)(F)


def _free_energy_func(deformation_gradient: Tensor, x_param: Tensor) -> Tensor:
    # Formulation for plane stress
    F = deformation_gradient
    J = _calculate_determinant(F)
    I_c = _right_cuachy_green_tensor_func(F)
    lambda_ = _first_lame_constant_lambda(x_param)
    mu_ = _second_lame_constant_mu(x_param)
    C = mu_ / 2
    D = lambda_ / 2
    free_energy = C * (I_c - 2 - 2 * torch.log(J)) + D * (J - 1) ** 2
    return torch.squeeze(free_energy, 0)


def _calculate_determinant(tensor: Tensor) -> Tensor:
    return torch.unsqueeze(torch.det(tensor), dim=0)


def _right_cuachy_green_tensor_func(deformation_gradient: Tensor) -> Tensor:
    F = deformation_gradient
    return torch.matmul(F.T, F)


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


def _deformation_gradient_func(
    ansatz: TModule, x_coor: Tensor, x_param: Tensor
) -> Tensor:
    jac_u = _jacobian_displacement_func(ansatz, x_coor, x_param)
    I = torch.eye(n=2, device=jac_u.device)
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
