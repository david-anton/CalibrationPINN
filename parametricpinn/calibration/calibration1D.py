# Standard library imports

# Third-party imports
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Local library imports
from parametricpinn.types import Tensor, Module


num_epochs = 10
loss_metric = torch.nn.MSELoss()
loss_weight = 1e6


# class AnalyticalAnsatz(nn.Module):
#     def __init__(self, ansatz: Module, initial_E: float):
#         super().__init__()
#         self.E = torch.nn.Parameter(torch.tensor([initial_E]), requires_grad=True)

#     def forward(self, coordinates):
#         length = 100.0
#         traction = 10.0
#         volume_force = 5.0
#         return (traction / self.E) * coordinates + (volume_force / self.E) * (
#             length * coordinates - 1 / 2 * coordinates**2
#         )


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
    initial_E = 210000.0
    predictor = Predictor(ansatz, initial_E)

    def plot_loss_func(ansatz: Module, coordinates: Tensor, data: Tensor) -> None:
        E_list = torch.linspace(180000.0, 240000.0, 1000).tolist()

        loss_list = []
        for E in E_list:
            predictor = Predictor(ansatz, E)
            y = predictor(coordinates)
            loss = loss_metric(y, data)
            loss_list.append(loss.item())

        figure, axes = plt.subplots()
        axes.set_title("Loss function")
        axes.plot(E_list, loss_list, label="loss PDE")
        axes.set_ylabel("MSE")
        axes.set_xlabel("epoch")
        figure.savefig("loss_func.png", bbox_inches="tight")
        plt.clf()

    def loss_func(coordinates: Tensor, data: Tensor) -> Tensor:
        y = predictor(coordinates)
        return loss_weight * loss_metric(y, data)

    optimizer = torch.optim.LBFGS(
        params=predictor.parameters(),
        lr=1.0,
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def loss_func_closure() -> float:
        optimizer.zero_grad()
        loss = loss_func(coordinates, data)
        loss.backward()
        return loss.item()

    loss_hist = []
    for _ in range(num_epochs):
        loss = loss_func(coordinates, data)
        optimizer.step(loss_func_closure)
        loss_hist.append(loss.cpu().item())
        print(f"Estimates E: {predictor.E}, Loss: {loss}, Grad: {predictor.E.grad}")

    plot_loss_func(ansatz, coordinates, data)
    return float(predictor.E.cpu().item()), loss_hist
