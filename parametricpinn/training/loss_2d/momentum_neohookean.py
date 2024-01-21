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
    ### Plane strain assumed
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
    J = _calculate_determinant(F)

    # Right Cauchy-Green tensor
    C = _calculate_right_cauchy_green_tensor(F)

    # Material parameters
    param_K = _extract_bulk_modulus_K(x_param)
    param_c_10 = _extract_rivlin_saunders_c_10(x_param)

    # Isochoric deformation tensors and invariants
    C_iso = (J ** (-2 / 3)) * C  # Isochoric right Cauchy-Green tensor
    I_C_iso = torch.trace(C_iso)  # Isochoric first invariant

    # 2. Piola-Kirchoff stress tensor
    I = torch.eye(3)
    inv_C_iso = torch.inverse(C_iso)
    T = J * param_K * (J - 1) * inv_C_iso + 2 * (J ** (-2 / 3)) * (
        param_c_10 * I - (1 / 3) * param_c_10 * I_C_iso * inv_C_iso
    )

    # 1. Piola-Kirchoff stress tensor
    P = F * T
    P_2D = P[0:2, 0:2]
    return P_2D


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
    transposed_F = torch.transpose(F, 0, 1)
    return torch.matmul(transposed_F, F)


def _calculate_first_invariant(tensor: Tensor):
    return torch.unsqueeze(torch.trace(tensor), dim=0)


def _extract_bulk_modulus_K(x_param: Tensor) -> Tensor:
    return torch.unsqueeze(x_param[0], dim=0)


def _extract_rivlin_saunders_c_10(x_param: Tensor) -> Tensor:
    return torch.unsqueeze(x_param[1], dim=0)
