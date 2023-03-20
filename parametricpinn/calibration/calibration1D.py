# Standard library imports

# Third-party imports
import torch
import torch.nn as nn

# Local library imports
from parametricpinn.settings import get_device
from parametricpinn.types import Tensor, Module


num_epochs = 100
loss_metric = torch.nn.MSELoss()
device = get_device()


class Predictor(nn.Module):
    def __init__(self, ansatz: Module, initial_E: float) -> None:
        super().__init__()
        self.ansatz = ansatz
        self.E = nn.Parameter(torch.tensor([initial_E]), requires_grad=True)
        self._freeze_ansatz(self.ansatz)

    def _freeze_ansatz(self, ansatz: Module) -> None:
        ansatz.train(False)
        for parameters in ansatz.parameters():
            parameters.requires_grad = False

    def forward(self, coordinates: Tensor) -> Tensor:
        length_input = coordinates.shape[0]
        x_E = self.E.expand(length_input, 1)
        x = torch.concat((coordinates, x_E), dim=1)
        return self.ansatz(x)


def calibrate_model(
    ansatz: Module, coordinates: Tensor, data: Tensor
) -> tuple[float, list[float]]:
    predictor = Predictor(ansatz, initial_E=210000.0).to(device)

    def loss_func(predictor: Module, coordinates: Tensor, data: Tensor) -> Tensor:
        coordinates = coordinates.to(device)
        data = data.to(device)
        y = predictor(coordinates)
        return loss_metric(y, data)

    optimizer = torch.optim.LBFGS(
        params=predictor.parameters(),
        lr=1.0,
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def loss_func_closure() -> Tensor:
        optimizer.zero_grad()
        loss = loss_func(predictor, coordinates, data)
        loss.backward()
        return loss

    loss_hist = []
    for _ in range(num_epochs):
        loss = loss_func(predictor, coordinates, data)
        optimizer.step(loss_func_closure)
        loss_hist.append(loss.cpu().item())

    estimates_E = float(predictor.E.cpu().item())
    return estimates_E, loss_hist
