import torch

from parametricpinn.training.loss_2d.momentumbase import (
    MomentumFunc,
    StressFunc,
    TModule,
    TractionFunc,
    jacobian_displacement_func,
    momentum_equation_func,
    stress_func,
    traction_func,
)
from parametricpinn.types import Tensor


def momentum_equation_func_factory() -> MomentumFunc:
    stress_func_single_input = _first_piola_stress_tensor_func
    return momentum_equation_func(stress_func_single_input)


def stress_func_factory() -> StressFunc:
    stress_func_single_input = _first_piola_stress_tensor_func
    return stress_func(stress_func_single_input)


def traction_func_factory() -> TractionFunc:
    stress_func_single_input = _first_piola_stress_tensor_func
    return traction_func(stress_func_single_input)


def _first_piola_stress_tensor_func(
    ansatz: TModule, x_coor: Tensor, x_param: Tensor
) -> Tensor:
    # Plane stress assumed
    F = _deformation_gradient_func(ansatz, x_coor, x_param)
    T_inv_F = torch.transpose(torch.inverse(F), 0, 1)
    det_F = _calculate_determinant(F)
    param_lambda = _calculate_first_lame_constant_lambda(x_param)
    param_mu = _calculate_second_lame_constant_mu(x_param)
    param_C = param_mu / 2
    param_D = param_lambda / 2
    return 2 * param_C * (F - T_inv_F) + 2 * param_D * (det_F - 1) * det_F * T_inv_F


def _calculate_determinant(tensor: Tensor) -> Tensor:
    determinant = torch.det(tensor)
    return torch.unsqueeze(determinant, dim=0)


def _calculate_first_lame_constant_lambda(x_param: Tensor) -> Tensor:
    device = x_param.device
    E = _extract_youngs_modulus(x_param)
    nu = _extract_poissons_ratio(x_param)
    return (E * nu) / (
        (torch.tensor([1.0]).to(device) + nu)
        * (torch.tensor([1.0]).to(device) - torch.tensor([2.0]).to(device) * nu)
    )


def _calculate_second_lame_constant_mu(x_param: Tensor) -> Tensor:
    device = x_param.device
    E = _extract_youngs_modulus(x_param)
    nu = _extract_poissons_ratio(x_param)
    return E / (torch.tensor([2.0]).to(device) * (torch.tensor([1.0]).to(device) + nu))


def _extract_youngs_modulus(x_param: Tensor) -> Tensor:
    return torch.unsqueeze(x_param[0], dim=0)


def _extract_poissons_ratio(x_param: Tensor) -> Tensor:
    return torch.unsqueeze(x_param[1], dim=0)


def _deformation_gradient_func(
    ansatz: TModule, x_coor: Tensor, x_param: Tensor
) -> Tensor:
    jac_u = jacobian_displacement_func(ansatz, x_coor, x_param)
    I = torch.eye(n=2, device=jac_u.device)
    return jac_u + I


# def _first_piola_stress_tensor_func(
#     ansatz: TModule, x_coor: Tensor, x_param: Tensor
# ) -> Tensor:
#     F = _deformation_gradient_func(ansatz, x_coor, x_param)
#     free_energy_func = lambda deformation_gradient: _free_energy_func(
#         deformation_gradient, x_param
#     )
#     return grad(free_energy_func)(F)


# def _free_energy_func(deformation_gradient: Tensor, x_param: Tensor) -> Tensor:
#     # Plane stress assumed
#     F = deformation_gradient
#     J = _calculate_determinant(F)
#     C = _calculate_right_cuachy_green_tensor(F)
#     I_c = _calculate_first_invariant(C)
#     param_lambda = _calculate_first_lame_constant_lambda(x_param)
#     param_mu = _calculate_second_lame_constant_mu(x_param)
#     param_C = param_mu / 2
#     param_D = param_lambda / 2
#     free_energy = param_C * (I_c - 2 - 2 * torch.log(J)) + param_D * (J - 1) ** 2
#     return torch.squeeze(free_energy, 0)

# def _calculate_right_cuachy_green_tensor(deformation_gradient: Tensor) -> Tensor:
#     F = deformation_gradient
#     return torch.matmul(F.T, F)


# def _calculate_first_invariant(tensor: Tensor) -> Tensor:
#     invariant = torch.trace(tensor)
#     return torch.unsqueeze(invariant, dim=0)
