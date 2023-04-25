import torch


def func(x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.tensor([x_0, x_1]) ** 2)


x = torch.tensor([2.0, 3.0])

grad_x = torch.func.grad(func, argnums=0)(x[0], x[1])


# ### solution
# def func(x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
#     return torch.sum(torch.concat((x_0, x_1)) ** 2)


# x = torch.tensor([2.0, 3.0])

# grad_x = torch.func.grad(func, argnums=0)(
#     torch.unsqueeze(x[0], dim=0), torch.unsqueeze(x[1], dim=0)
# )

# print(grad_x)
