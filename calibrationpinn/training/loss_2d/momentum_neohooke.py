import torch

from calibrationpinn.training.loss_2d.momentumbase import (
    MomentumFunc,
    StressFunc,
    TModule,
    TractionFunc,
    jacobian_displacement_func,
    momentum_equation_func,
    stress_func,
    traction_func,
)
from calibrationpinn.types import Tensor


def momentum_equation_func_factory() -> MomentumFunc:
    stress_func_single_input = _first_piola_stress_tensor_func
    return momentum_equation_func(stress_func_single_input)


def first_piola_kirchhoff_stress_func_factory() -> StressFunc:
    stress_func_single_input = _first_piola_stress_tensor_func
    return stress_func(stress_func_single_input)


def cauchy_stress_func_factory() -> StressFunc:
    stress_func_single_input = _cauchy_stress_func
    return stress_func(stress_func_single_input)


def traction_func_factory() -> TractionFunc:
    stress_func_single_input = _first_piola_stress_tensor_func
    return traction_func(stress_func_single_input)


def _first_piola_stress_tensor_func(
    ansatz: TModule, x_coor: Tensor, x_param: Tensor
) -> Tensor:
    ### Plane strain assumed
    # References:
    # - "Nonlinear Finite Element Method", P. Wriggers, 2008
    # - https://reference.wolfram.com/language/PDEModels/tutorial/StructuralMechanics/Hyperelasticity.html, 7.2.2024

    device = x_coor.device
    # Deformation gradient
    F_2D = _deformation_gradient_func(ansatz, x_coor, x_param)
    F = torch.stack(
        (
            torch.concat((F_2D[0, :], torch.tensor([0.0], device=device)), dim=0),
            torch.concat((F_2D[1, :], torch.tensor([0.0], device=device)), dim=0),
            torch.tensor([0.0, 0.0, 1.0], device=device),
        ),
        dim=0,
    )

    # Right Cauchy-Green tensor
    C = _calculate_right_cauchy_green_tensor(F)

    # Invariants
    J = _calculate_determinant(F)
    I_C = _calculate_first_invariant(C)

    # Material parameters
    param_K = _extract_bulk_modulus_K(x_param)
    param_G = _extract_shear_modulus_G(x_param)

    # 2. Piola-Kirchoff stress tensor
    I = torch.eye(3, device=device)
    C_inverse = torch.inverse(C)
    T_vol = param_K / 2 * (J**2 - 1) * C_inverse
    T_iso = param_G * (J ** (-2 / 3)) * (I - (1 / 3) * I_C * C_inverse)
    T = T_vol + T_iso

    # 1. Piola-Kirchoff stress tensor
    P = torch.matmul(F, T)
    P_2D = P[0:2, 0:2]
    return P_2D


def _cauchy_stress_func(ansatz: TModule, x_coor: Tensor, x_param: Tensor) -> Tensor:
    P = _first_piola_stress_tensor_func(ansatz, x_coor, x_param)
    F = _deformation_gradient_func(ansatz, x_coor, x_param)
    F_transpose = torch.transpose(F, 0, 1)
    J = _calculate_determinant(F)
    return J ** (-1) * torch.matmul(P, F_transpose)


def _deformation_gradient_func(
    ansatz: TModule, x_coor: Tensor, x_param: Tensor
) -> Tensor:
    jac_u = jacobian_displacement_func(ansatz, x_coor, x_param)
    I = torch.eye(n=2, device=jac_u.device)
    return jac_u + I


def _calculate_determinant(tensor: Tensor) -> Tensor:
    determinant = torch.det(tensor)
    return torch.unsqueeze(determinant, dim=0)


def _calculate_right_cauchy_green_tensor(deformation_gradient: Tensor) -> Tensor:
    F = deformation_gradient
    F_transpose = torch.transpose(F, 0, 1)
    return torch.matmul(F_transpose, F)


def _calculate_first_invariant(tensor: Tensor):
    return torch.unsqueeze(torch.trace(tensor), dim=0)


def _extract_bulk_modulus_K(x_param: Tensor) -> Tensor:
    return torch.unsqueeze(x_param[0], dim=0)


def _extract_shear_modulus_G(x_param: Tensor) -> Tensor:
    return torch.unsqueeze(x_param[1], dim=0)
