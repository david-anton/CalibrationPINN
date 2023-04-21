from typing import Callable, TypeAlias

import torch
from torch.autograd.functional import jacobian

from parametricpinn.types import Module, Tensor

StressFunc: TypeAlias = Callable[[Module, Tensor, Tensor], Tensor]
MomentumFunc: TypeAlias = Callable[[Module, Tensor, Tensor, Tensor], Tensor]
TractionFunc: TypeAlias = Callable[[Module, Tensor, Tensor, Tensor], Tensor]


def _momentum_equation_func(stress_func: StressFunc) -> MomentumFunc:
    def momentum_equation_func(
        ansatz: Module, x_coor: Tensor, x_param: Tensor, volume_force: Tensor
    ) -> Tensor:
        return (
            _divergence_stress_func(ansatz, x_coor, x_param, stress_func) + volume_force
        )

    return momentum_equation_func


def _traction_func(stress_func: StressFunc) -> TractionFunc:
    def traction_func(
        ansatz: Module, x_coor: Tensor, x_param: Tensor, normal: Tensor
    ) -> Tensor:
        stress = stress_func(ansatz, x_coor, x_param)
        return torch.matmul(stress, normal)

    return traction_func


def _divergence_stress_func(
    ansatz: Module, x_coor: Tensor, x_param: Tensor, stress_func: StressFunc
) -> Tensor:
    stress_xx_func = lambda x, y: stress_func(ansatz, torch.concat((x, y)), x_param)[
        :, 0, 0
    ].view((-1, 1))
    stress_yy_func = lambda x, y: stress_func(ansatz, torch.concat((x, y)), x_param)[
        :, 1, 1
    ].view((-1, 1))
    stress_xy_func = lambda x, y: stress_func(ansatz, torch.concat((x, y)), x_param)[
        :, 0, 1
    ].view((-1, 1))
    sigxx_x = torch.autograd.grad(
        stress_xx_func,
        x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
        is_grads_batched=True,
    )[0]


# def _divergence_stress_func(
#     ansatz: Module, x_coor: Tensor, x_param: Tensor, stress_func: StressFunc
# ) -> Tensor:
#     def jacobian_voigt_stress(x_coor: Tensor) -> Tensor:
#         return jacobian(
#             voigt_stress_func,
#             (x_coor),
#             create_graph=True,
#             strict=True,
#             vectorize=True,
#             strategy="reverse-mode",
#         )
#         # return jacobian(
#         #     voigt_stress_func,
#         #     (x_coor),
#         #     create_graph=True,
#         #     strict=True,
#         #     vectorize=True,
#         #     strategy="reverse-mode",
#         # )

#     def voigt_stress_func(x_coor: Tensor) -> Tensor:
#         stress = stress_func(ansatz, x_coor, x_param)
#         return torch.transpose(
#             torch.vstack((stress[:, 0, 0], stress[:, 1, 1], stress[:, 0, 1])), 0, 1
#         )
#         # return torch.tensor([stress[0, 0], stress[1, 1], stress[0, 1]])

#     jac_voigt_stress = jacobian_voigt_stress(x_coor)
#     sigxx_x = jac_voigt_stress[:, 0, 0]
#     sigxy_y = jac_voigt_stress[:, 2, 1]
#     sigyx_x = jac_voigt_stress[:, 2, 0]
#     sigyy_y = jac_voigt_stress[:, 1, 1]
#     return torch.transpose(torch.vstack((sigxx_x + sigxy_y, sigyx_x + sigyy_y)), 0, 1)
#     # sigxx_x = jac_voigt_stress[0, 0]
#     # sigxy_y = jac_voigt_stress[2, 1]
#     # sigyx_x = jac_voigt_stress[2, 0]
#     # sigyy_y = jac_voigt_stress[1, 1]
#     # return torch.tensor([sigxx_x + sigxy_y, sigyx_x + sigyy_y])


def _strain_func(ansatz: Module, x_coor: Tensor, x_param: Tensor) -> Tensor:
    jac_u = _jacobian_displacement_func(ansatz, x_coor, x_param)
    return 1 / 2 * (jac_u + torch.transpose(jac_u, 1, 2))
    # return 1 / 2 * (jac_u + torch.transpose(jac_u, 0, 1))


def _jacobian_displacement_func(
    ansatz: Module, x_coor: Tensor, x_param: Tensor
) -> Tensor:
    displacement_func = lambda x_coor: _displacement_func(ansatz, x_coor, x_param)
    return jacobian(
        displacement_func,
        (x_coor),
        create_graph=True,
        strict=True,
        vectorize=True,
        strategy="reverse-mode",
    )
    # return jacobian(
    #     displacement_func,
    #     (x_coor),
    #     create_graph=True,
    #     strict=True,
    #     strategy="reverse-mode",
    # )


def _displacement_func(ansatz: Module, x_coor: Tensor, x_param: Tensor) -> Tensor:
    x = torch.concat((x_coor, x_param), dim=0)
    return ansatz(x)
