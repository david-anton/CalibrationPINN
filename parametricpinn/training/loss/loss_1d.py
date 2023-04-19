import torch

from parametricpinn.types import Module, Tensor


def momentum_equation_func_1D(
    ansatz: Module, x_coor: Tensor, x_E: Tensor, volume_force: Tensor
) -> Tensor:
    stress = stress_func_1D(ansatz, x_coor, x_E)
    stress_x = torch.autograd.grad(
        stress,
        x_coor,
        grad_outputs=torch.ones_like(stress),
        retain_graph=True,
        create_graph=True,
    )[0]
    return stress_x + volume_force


def stress_func_1D(ansatz: Module, x_coor: Tensor, x_E: Tensor) -> Tensor:
    x = torch.concat((x_coor, x_E), dim=1)
    u = ansatz(x)
    u_x = torch.autograd.grad(
        u,
        x_coor,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]
    return x_E * u_x
