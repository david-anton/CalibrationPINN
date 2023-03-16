# Standard library imports
from collections import namedtuple

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Local library imports
from parametricpinn.ansatz import create_normalized_HBC_ansatz_1D
from parametricpinn.network import FFNN
from parametricpinn.settings import get_device, set_default_dtype, set_seed


### Configuration
# Set up
length = 100.0
min_youngs_modulus = 180000.0
max_youngs_modulus = 240000.0
traction = 10.0
volume_force = 5.0
# Network
layer_sizes = [2, 16, 16, 1]
# Training
num_samples_train = 100
num_points_pde = 100
batch_size_train = num_samples_train
num_epochs = 100
# Validation
num_samples_valid = 100
valid_interval = 1
num_points_valid = 1000
batch_size_valid = num_points_valid

set_default_dtype(torch.float64)
set_seed(0)
device = get_device()


### Solution
def calculate_displacements_solution(
    coordinates, length, youngs_modulus, traction, volume_force
):
    return (traction / youngs_modulus) * coordinates + (
        volume_force / youngs_modulus
    ) * (length * coordinates - 1 / 2 * coordinates**2)


### Data
TrainingData = namedtuple("TrainingData", ["x_coor", "x_E", "y_true"])


class TrainingDataset(Dataset):
    def __init__(
        self,
        length,
        traction,
        min_youngs_modulus,
        max_youngs_modulus,
        num_points_pde,
        num_samples,
    ):
        self._length = length
        self._traction = traction
        self._min_youngs_modulus = min_youngs_modulus
        self._max_youngs_modulus = max_youngs_modulus
        self._num_points_pde = num_points_pde
        self._num_points_stress_bc = 1
        self._num_samples = num_samples
        self._samples_pde = []
        self._samples_stress_bc = []

        self._generate_samples()

    def _generate_samples(self):
        youngs_modulus_list = self._generate_uniform_youngs_modulus_list()
        for i in range(self._num_samples):
            youngs_modulus = youngs_modulus_list[i]
            self._add_pde_sample(youngs_modulus)
            self._add_stress_bc_sample(youngs_modulus)

    def _generate_uniform_youngs_modulus_list(self):
        return torch.linspace(
            self._min_youngs_modulus, self._max_youngs_modulus, self._num_samples
        ).tolist()

    def _add_pde_sample(self, youngs_modulus):
        x_coor_pde = torch.linspace(
            0.0, length, self._num_points_pde, requires_grad=True
        ).view([self._num_points_pde, 1])
        x_E_pde = torch.full((self._num_points_pde, 1), youngs_modulus)
        y_true_pde = torch.zeros_like(x_coor_pde)
        sample_pde = TrainingData(x_coor=x_coor_pde, x_E=x_E_pde, y_true=y_true_pde)
        self._samples_pde.append(sample_pde)

    def _add_stress_bc_sample(self, youngs_modulus):
        x_coor_stress_bc = torch.full(
            (self._num_points_stress_bc, 1), length, requires_grad=True
        )
        x_E_stress_bc = torch.full((self._num_points_stress_bc, 1), youngs_modulus)
        y_true_stress_bc = torch.full((self._num_points_stress_bc, 1), traction)
        sample_stress_bc = TrainingData(
            x_coor=x_coor_stress_bc, x_E=x_E_stress_bc, y_true=y_true_stress_bc
        )
        self._samples_stress_bc.append(sample_stress_bc)

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        sample_pde = self._samples_pde[idx]
        sample_stress_bc = self._samples_stress_bc[idx]
        return sample_pde, sample_stress_bc


def collate_training_data(batch):
    x_coor_pde_batch = []
    x_E_pde_batch = []
    y_true_pde_batch = []
    x_coor_stress_bc_batch = []
    x_E_stress_bc_batch = []
    y_true_stress_bc_batch = []

    for sample_pde, sample_stress_bc in batch:
        x_coor_pde_batch.append(sample_pde.x_coor)
        x_E_pde_batch.append(sample_pde.x_E)
        y_true_pde_batch.append(sample_pde.y_true)
        x_coor_stress_bc_batch.append(sample_stress_bc.x_coor)
        x_E_stress_bc_batch.append(sample_stress_bc.x_E)
        y_true_stress_bc_batch.append(sample_stress_bc.y_true)

    batch_pde = TrainingData(
        x_coor=torch.concat(x_coor_pde_batch, dim=0),
        x_E=torch.concat(x_E_pde_batch, dim=0),
        y_true=torch.concat(y_true_pde_batch, dim=0),
    )
    batch_stress_bc = TrainingData(
        x_coor=torch.concat(x_coor_stress_bc_batch, dim=0),
        x_E=torch.concat(x_E_stress_bc_batch, dim=0),
        y_true=torch.concat(y_true_stress_bc_batch, dim=0),
    )
    return batch_pde, batch_stress_bc


class ValidationDataset(Dataset):
    def __init__(
        self,
        length,
        min_youngs_modulus,
        max_youngs_modulus,
        traction,
        volume_force,
        num_points,
        num_samples,
    ):
        self._length = length
        self._min_youngs_modulus = min_youngs_modulus
        self._max_youngs_modulus = max_youngs_modulus
        self._traction = traction
        self._volume_force = volume_force
        self._num_points = num_points
        self._num_samples = num_samples
        self._samples_x = []
        self._samples_y_true = []

        self._generate_samples()

    def _generate_samples(self):
        youngs_modulus_list = self._generate_random_youngs_modulus_list()
        coordinates_list = self._generate_random_coordinates_list()
        for i in range(self._num_samples):
            youngs_modulus = youngs_modulus_list[i]
            coordinates = coordinates_list[i]
            self._add_input_sample(coordinates, youngs_modulus)
            self._add_output_sample(coordinates, youngs_modulus)

    def _generate_random_youngs_modulus_list(self):
        return (
            self._min_youngs_modulus
            + torch.rand((self._num_samples))
            * (self._max_youngs_modulus - self._min_youngs_modulus)
        ).tolist()

    def _generate_random_coordinates_list(self):
        coordinates_array = torch.rand((self._num_points, self._num_samples)) * length
        return torch.chunk(coordinates_array, self._num_samples, dim=1)

    def _add_input_sample(self, coordinates, youngs_modulus):
        x_coor = coordinates
        x_E = torch.full((self._num_points, 1), youngs_modulus)
        x = torch.concat((x_coor, x_E), dim=1)
        self._samples_x.append(x)

    def _add_output_sample(self, coordinates, youngs_modulus):
        y_true = calculate_displacements_solution(
            coordinates=coordinates,
            length=self._length,
            youngs_modulus=youngs_modulus,
            traction=self._traction,
            volume_force=self._volume_force,
        )
        self._samples_y_true.append(y_true)

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        sample_x = self._samples_x[idx]
        sample_y_true = self._samples_y_true[idx]
        return sample_x, sample_y_true


def collate_validation_data(batch):
    x_batch = []
    y_true_batch = []

    for sample_x, sample_y_true in batch:
        x_batch.append(sample_x)
        y_true_batch.append(sample_y_true)

    batch_x = torch.concat(x_batch, dim=0)
    batch_y_true = torch.concat(y_true_batch, dim=0)
    return batch_x, batch_y_true


### Loss function
loss_metric = torch.nn.MSELoss()


def loss_func(model, pde_data, stress_bc_data):
    def loss_func_pde(model, pde_data):
        x_coor = pde_data.x_coor.to(device)
        x_E = pde_data.x_E.to(device)
        y_true = pde_data.y_true.to(device)
        y = pinn_func(model, x_coor, x_E)
        return loss_metric(y_true, y)

    def loss_func_stress_bc(model, stress_bc_data):
        x_coor = stress_bc_data.x_coor.to(device)
        x_E = stress_bc_data.x_E.to(device)
        y_true = stress_bc_data.y_true.to(device)
        y = stress_func(model, x_coor, x_E)
        return loss_metric(y_true, y)

    loss_pde = loss_func_pde(model, pde_data)
    loss_stress_bc = loss_func_stress_bc(model, stress_bc_data)
    return loss_pde, loss_stress_bc


def pinn_func(model, x_coor, x_E):
    stress = stress_func(model, x_coor, x_E)
    stress_x = torch.autograd.grad(
        stress,
        x_coor,
        grad_outputs=torch.ones_like(stress),
        retain_graph=True,
        create_graph=True,
    )[0]
    return stress_x + volume_force


def stress_func(model, x_coor, x_E):
    x = torch.concat((x_coor, x_E), dim=1)
    u = model(x)
    u_x = torch.autograd.grad(
        u,
        x_coor,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]
    return x_E * u_x


### Validation
valid_metric_mae = torch.nn.L1Loss()


def validate_model(model, valid_dataloader):
    with torch.no_grad():
        model.eval()
        valid_batches = iter(valid_dataloader)
        mae_hist_batches = []
        mare_hist_batches = []
        batch_counter = 0

        for x, y_true in valid_batches:
            x = x.to(device)
            y_true = y_true.to(device)
            y = model(x)
            mae_batch = valid_metric_mae(y_true, y)
            mare_batch = torch.mean((torch.abs(y_true - y)) / y_true)
            mae_hist_batches.append(mae_batch.item())
            mare_hist_batches.append(mare_batch.item())
            batch_counter += 1

        mean_mae = sum(mae_hist_batches) / batch_counter
        mean_mare = sum(mare_hist_batches) / batch_counter
    return mean_mae, mean_mare


### Plotting
class PlotterConfig:
    def __init__(self) -> None:
        # font sizes
        self.label_size = 20
        # font size in legend
        self.font_size = 16
        self.font = {"size": self.label_size}

        # major ticks
        self.major_tick_label_size = 20
        self.major_ticks_size = self.font_size
        self.major_ticks_width = 2

        # minor ticks
        self.minor_tick_label_size = 14
        self.minor_ticks_size = 12
        self.minor_ticks_width = 1

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300


def plot_loss_hist(loss_hist_pde, loss_hist_stress_bc, file_name, config):
    figure, axes = plt.subplots()
    axes.set_title("Loss history", **config.font)
    axes.plot(loss_hist_pde, label="loss PDE")
    axes.plot(loss_hist_stress_bc, label="loss stress BC")
    axes.set_yscale("log")
    axes.set_ylabel("MSE", **config.font)
    axes.set_xlabel("epoch", **config.font)
    axes.tick_params(axis="both", which="minor", labelsize=config.minor_tick_label_size)
    axes.tick_params(axis="both", which="major", labelsize=config.major_tick_label_size)
    axes.legend(fontsize=plotter_config.font_size, loc="best")
    figure.savefig(file_name, bbox_inches="tight", dpi=plotter_config.dpi)
    plt.clf()


def plot_valid_hist(valid_epochs, valid_hist, valid_metric, file_name, config):
    figure, axes = plt.subplots()
    axes.set_title(valid_metric, **config.font)
    axes.plot(valid_epochs, valid_hist, label=valid_metric)
    axes.set_yscale("log")
    axes.set_xlabel("epoch", **config.font)
    axes.tick_params(axis="both", which="minor", labelsize=config.minor_tick_label_size)
    axes.tick_params(axis="both", which="major", labelsize=config.major_tick_label_size)
    figure.savefig(file_name, bbox_inches="tight", dpi=plotter_config.dpi)
    plt.clf()


def plot_displacements(youngs_modulus, file_name, config):
    num_points = 1000
    x_coor = np.linspace(0.0, length, num_points).reshape((num_points, 1))
    solution = calculate_displacements_solution(
        coordinates=x_coor,
        length=length,
        youngs_modulus=youngs_modulus,
        traction=traction,
        volume_force=volume_force,
    )
    x_coor = torch.Tensor(x_coor)
    x_E = torch.full((num_points, 1), youngs_modulus)
    x = torch.concat((x_coor, x_E), dim=1)
    prediction = model(x).detach().numpy()
    figure, axes = plt.subplots()
    axes.plot(x_coor, solution, label="solution")
    axes.plot(x_coor, prediction, label="prediction")
    axes.set_xlabel("coordinate [mm]", **config.font)
    axes.set_ylabel("displacements [mm]", **config.font)
    axes.tick_params(axis="both", which="minor", labelsize=config.minor_tick_label_size)
    axes.tick_params(axis="both", which="major", labelsize=config.major_tick_label_size)
    axes.legend(fontsize=plotter_config.font_size, loc="best")
    figure.savefig(file_name, bbox_inches="tight", dpi=plotter_config.dpi)
    plt.clf()


####################################################################################################
if __name__ == "__main__":
    min_coordinate = torch.Tensor([0.0])
    max_coordinate = torch.Tensor([length])
    min_displacement = torch.Tensor([0.0])
    max_displacement = calculate_displacements_solution(
        coordinates=max_coordinate,
        length=length,
        youngs_modulus=min_youngs_modulus,
        traction=traction,
        volume_force=volume_force,
    )
    min_inputs = torch.Tensor([min_coordinate, min_youngs_modulus])
    max_inputs = torch.Tensor([max_coordinate, max_youngs_modulus])
    min_output = min_displacement
    max_output = max_displacement

    network = FFNN(layer_sizes=layer_sizes)
    model = create_normalized_HBC_ansatz_1D(
        network=network,
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        min_outputs=min_output,
        max_outputs=max_output,
    ).to(device)

    train_dataset = TrainingDataset(
        length=length,
        traction=traction,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        num_points_pde=num_points_pde,
        num_samples=num_samples_train,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_train,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_training_data,
    )

    valid_dataset = ValidationDataset(
        length=length,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        traction=traction,
        volume_force=volume_force,
        num_points=num_points_valid,
        num_samples=num_samples_valid,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size_valid,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_validation_data,
    )

    optimizer = torch.optim.LBFGS(
        params=model.parameters(),
        lr=1.0,
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    loss_hist_pde = []
    loss_hist_stress_bc = []
    valid_hist_mae = []
    valid_hist_mare = []
    valid_epochs = []

    # Closure for LBFGS
    def loss_func_closure():
        optimizer.zero_grad()
        loss_pde, loss_stress_bc = loss_func(model, batch_pde, batch_stress_bc)
        loss = loss_pde + loss_stress_bc
        loss.backward()
        return loss

    for epoch in range(num_epochs):
        train_batches = iter(train_dataloader)
        loss_hist_pde_batches = []
        loss_hist_stress_bc_batches = []
        batch_counter = 0

        for batch_pde, batch_stress_bc in train_batches:
            # Forward pass
            loss_pde, loss_stress_bc = loss_func(model, batch_pde, batch_stress_bc)
            loss = loss_pde + loss_stress_bc

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update parameters
            optimizer.step(loss_func_closure)

            loss_pde = loss_pde.cpu().item()
            loss_stress_bc = loss_stress_bc.cpu().item()

            loss_hist_pde_batches.append(loss_pde)
            loss_hist_stress_bc_batches.append(loss_stress_bc)
            batch_counter += 1

        mean_batch_loss_pde = sum(loss_hist_pde_batches) / batch_counter
        mean_batch_loss_stress_bc = sum(loss_hist_stress_bc_batches) / batch_counter
        loss_hist_pde.append(mean_batch_loss_pde)
        loss_hist_stress_bc.append(mean_batch_loss_stress_bc)

        if epoch % 1 == 0:
            mae, mare = validate_model(model, valid_dataloader)
            valid_hist_mae.append(mae)
            valid_hist_mare.append(mare)
            valid_epochs.append(epoch)

    plotter_config = PlotterConfig()

    plot_loss_hist(
        loss_hist_pde=loss_hist_pde,
        loss_hist_stress_bc=loss_hist_stress_bc,
        file_name="loss_p_pinn.png",
        config=plotter_config,
    )

    plot_valid_hist(
        valid_epochs=valid_epochs,
        valid_hist=valid_hist_mae,
        valid_metric="mean absolute error",
        file_name="mae_p_pinn.png",
        config=plotter_config,
    )

    plot_valid_hist(
        valid_epochs=valid_epochs,
        valid_hist=valid_hist_mare,
        valid_metric="mean absolute relative error",
        file_name="mare_p_pinn.png",
        config=plotter_config,
    )

    plot_displacements(
        youngs_modulus=180000,
        file_name="displacements_p_pinn_E_180000.png",
        config=plotter_config,
    )
    plot_displacements(
        youngs_modulus=187634,
        file_name="displacements_p_pinn_E_187634.png",
        config=plotter_config,
    )
    plot_displacements(
        youngs_modulus=190000,
        file_name="displacements_p_pinn_E_190000.png",
        config=plotter_config,
    )
    plot_displacements(
        youngs_modulus=200000,
        file_name="displacements_p_pinn_E_200000.png",
        config=plotter_config,
    )
    plot_displacements(
        youngs_modulus=210000,
        file_name="displacements_p_pinn_E_210000.png",
        config=plotter_config,
    )
    plot_displacements(
        youngs_modulus=220000,
        file_name="displacements_p_pinn_E_220000.png",
        config=plotter_config,
    )
    plot_displacements(
        youngs_modulus=230000,
        file_name="displacements_p_pinn_E_230000.png",
        config=plotter_config,
    )
    plot_displacements(
        youngs_modulus=238356,
        file_name="displacements_p_pinn_E_238356.png",
        config=plotter_config,
    )
    plot_displacements(
        youngs_modulus=240000,
        file_name="displacements_p_pinn_E_240000.png",
        config=plotter_config,
    )
