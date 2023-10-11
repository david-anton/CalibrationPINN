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
    divergence_stress = torch.concat((stress_xx_x + stress_xy_y, stress_xy_x + stress_yy_y), dim=0)
    # print(f"Div sigma: {divergence_stress}")
    return divergence_stress


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
    # print(f"F: {F}")
    free_energy_func = lambda deformation_gradient: _free_energy_func(
        deformation_gradient, x_param
    )
    print(f"Psi: {free_energy_func}")
    return grad(free_energy_func)(F)


def _free_energy_func(deformation_gradient: Tensor, x_param: Tensor) -> Tensor:
    # Plane stress assumed
    F = deformation_gradient
    J = _calculate_determinant(F)
    # print(f"J: {J}")
    C = _calculate_right_cuachy_green_tensor(F)
    # print(f"C: {C}")
    I_c = _calculate_first_invariant(C)
    # print(f"I_c: {I_c}")
    param_lambda = _calculate_first_lame_constant_lambda(x_param)
    param_mu = _calculate_second_lame_constant_mu(x_param)
    param_C = param_mu / 2
    param_D = param_lambda / 2
    free_energy = param_C * (I_c - 2 - 2 * torch.log(J)) + param_D * (J - 1) ** 2
    return torch.squeeze(free_energy, 0)


def _calculate_determinant(tensor: Tensor) -> Tensor:
    determinant = torch.det(tensor)
    return torch.unsqueeze(determinant, dim=0)


def _calculate_right_cuachy_green_tensor(deformation_gradient: Tensor) -> Tensor:
    F = deformation_gradient
    return torch.matmul(F.T, F)


def _calculate_first_invariant(tensor: Tensor) -> Tensor:
    invariant = torch.trace(tensor)
    return torch.unsqueeze(invariant, dim=0)


def _calculate_first_lame_constant_lambda(x_param: Tensor) -> Tensor:
    E = _extract_youngs_modulus(x_param)
    nu = _extract_poissons_ratio(x_param)
    return (E * nu) / (
        (torch.tensor([1.0]).to(x_param.device) + nu)
        * (
            torch.tensor([1.0]).to(x_param.device)
            - torch.tensor([2.0]).to(x_param.device) * nu
        )
    )


def _calculate_second_lame_constant_mu(x_param: Tensor) -> Tensor:
    E = _extract_youngs_modulus(x_param)
    nu = _extract_poissons_ratio(x_param)
    return E / (
        torch.tensor([2.0]).to(x_param.device)
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
    print(f"jac_u: {jac_u}")
    I = torch.eye(n=2, device=jac_u.device)
    return jac_u + I


def _jacobian_displacement_func(
    ansatz: TModule, x_coor: Tensor, x_param: Tensor
) -> Tensor:
    displacement_func = lambda _x_coor: _displacement_func(ansatz, _x_coor, x_param)
    return jacrev(displacement_func, argnums=0)(x_coor)


def _displacement_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    print(f"Is displacement finite: {torch.isfinite(ansatz(x_coor, x_param))}")
    return ansatz(x_coor, x_param)


def _transform_ansatz(ansatz: Module) -> TModule:
    return lambda x_coor, x_param: ansatz(torch.concat((x_coor, x_param)))
