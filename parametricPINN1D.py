# Standard library imports

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Local library imports
from parametricpinn.ansatz import create_normalized_HBC_ansatz_1D
from parametricpinn.data import (
    TrainingDataset1D,
    collate_training_data_1D,
    calculate_displacements_solution_1D,
    ValidationDataset1D,
    collate_validation_data_1D,
)
from parametricpinn.network import FFNN
from parametricpinn.settings import get_device, set_default_dtype, set_seed
from parametricpinn.training.loss import momentum_equation_func_1D, stress_func_1D


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


### Loss function
loss_metric = torch.nn.MSELoss()


def loss_func(model, pde_data, stress_bc_data):
    def loss_func_pde(model, pde_data):
        x_coor = pde_data.x_coor.to(device)
        x_E = pde_data.x_E.to(device)
        y_true = pde_data.y_true.to(device)
        y = momentum_equation_func_1D(model, x_coor, x_E, volume_force)
        return loss_metric(y_true, y)

    def loss_func_stress_bc(model, stress_bc_data):
        x_coor = stress_bc_data.x_coor.to(device)
        x_E = stress_bc_data.x_E.to(device)
        y_true = stress_bc_data.y_true.to(device)
        y = stress_func_1D(model, x_coor, x_E)
        return loss_metric(y_true, y)

    loss_pde = loss_func_pde(model, pde_data)
    loss_stress_bc = loss_func_stress_bc(model, stress_bc_data)
    return loss_pde, loss_stress_bc


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
    solution = calculate_displacements_solution_1D(
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
    max_displacement = calculate_displacements_solution_1D(
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

    train_dataset = TrainingDataset1D(
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
        collate_fn=collate_training_data_1D,
    )

    valid_dataset = ValidationDataset1D(
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
        collate_fn=collate_validation_data_1D,
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
        youngs_modulus=187634,
        file_name="displacements_p_pinn_E_187634.png",
        config=plotter_config,
    )
    plot_displacements(
        youngs_modulus=238356,
        file_name="displacements_p_pinn_E_238356.png",
        config=plotter_config,
    )
