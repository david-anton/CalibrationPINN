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
    ### Plane stress assumed ???
    # Identity
    I = torch.eye(n=2, device=x_coor.device)

    # Deformation gradient
    F = _deformation_gradient_func(ansatz, x_coor, x_param)
    J = _calculate_determinant(F)

    # Unimodular deformation tensors
    uni_F = (J ** (-1 / 3)) * F  # unimodular deformation gradient
    transpose_uni_F = torch.transpose(uni_F, 0, 1)
    uni_C = transpose_uni_F * uni_F  # unimodular right Cauchy-Green tensor

    # Invariants of unimodular deformation tensors
    uni_I_c = torch.trace(uni_C)

    # Material parameters
    param_K = _extract_bulk_modulus_K(x_param)
    param_c_10 = _extract_rivlin_saunders_c_10(x_param)

    # 2. Piola-Kirchoff stress tensor
    inv_uni_C = torch.inverse(uni_C)
    T = J * param_K * (J - 1) * inv_uni_C + 2 * (J ** (-2 / 3)) * (
        param_c_10 * I - (1 / 3) * param_c_10 * uni_I_c * inv_uni_C
    )

    # 1. Piola-Kirchoff stress tensor
    P = F * T
    return P


def _deformation_gradient_func(
    ansatz: TModule, x_coor: Tensor, x_param: Tensor
) -> Tensor:
    jac_u = jacobian_displacement_func(ansatz, x_coor, x_param)
    I = torch.eye(n=2, device=jac_u.device)
    return jac_u + I


def _calculate_determinant(tensor: Tensor) -> Tensor:
    determinant = torch.det(tensor)
    return torch.unsqueeze(determinant, dim=0)


def _extract_bulk_modulus_K(x_param: Tensor) -> Tensor:
    return torch.unsqueeze(x_param[0], dim=0)


def _extract_rivlin_saunders_c_10(x_param: Tensor) -> Tensor:
    return torch.unsqueeze(x_param[1], dim=0)
