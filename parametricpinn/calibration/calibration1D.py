# Standard library imports

# Third-party imports
import torch
import torch.nn as nn

# Local library imports
from parametricpinn.settings import get_device
from parametricpinn.types import Tensor, Module


num_epochs = 5
loss_metric = torch.nn.MSELoss()
device = get_device()


# class Predictor(nn.Module):
#     def __init__(self, ansatz: Module, initial_E: float) -> None:
#         super().__init__()
#         self.ansatz = ansatz
#         self.E = nn.Parameter(torch.tensor([initial_E]), requires_grad=True)
#         self._freeze_ansatz(self.ansatz)

#     def _freeze_ansatz(self, ansatz: Module) -> None:
#         ansatz.train(False)
#         for parameters in ansatz.parameters():
#             parameters.requires_grad = False

#     def forward(self, coordinates: Tensor) -> Tensor:
#         length_input = coordinates.shape[0]
#         x_E = self.E.expand(length_input, 1)
#         x = torch.concat((coordinates, x_E), dim=1)
#         return self.ansatz(x)


def calibrate_model(
    ansatz: Module, coordinates: Tensor, data: Tensor
) -> tuple[float, list[float]]:
    initial_E = 210000.0
    # ansatz.to(device)
    # ansatz.requires_grad_(False)
    estimated_E = nn.Parameter(torch.tensor([initial_E]), requires_grad=True).to(device)
    length_input = coordinates.shape[0]

    def predict(estimated_E: Tensor) -> Tensor:
        x_E = estimated_E.expand(length_input, 1)
        x_E = x_E.clone()  # contiguous?
        x = torch.concat((coordinates, x_E), dim=1)
        return ansatz(x)

    def loss_func(estimated_E: Tensor, coordinates: Tensor, data: Tensor) -> Tensor:
        coordinates = coordinates.to(device)
        data = data.to(device)
        y = predict(estimated_E)
        return loss_metric(y, data)

    optimizer = torch.optim.LBFGS(
        params=[estimated_E],
        lr=1.0,
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def loss_func_closure() -> float:
        optimizer.zero_grad()
        loss = loss_func(estimated_E, coordinates, data)
        loss.backward()
        return loss.item()

    loss_hist = []
    for _ in range(num_epochs):
        loss = loss_func(estimated_E, coordinates, data)
        optimizer.step(loss_func_closure)
        loss_hist.append(loss.cpu().item())

    return float(estimated_E.cpu().item()), loss_hist
