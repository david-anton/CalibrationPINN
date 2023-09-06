import matplotlib.pyplot as plt
import numpy as np
import torch

from parametricpinn.ansatz import BayesianAnsatz, StandardAnsatz
from parametricpinn.data import calculate_displacements_solution_1D
from parametricpinn.io import ProjectDirectory
from parametricpinn.types import Device, NPArray, Tensor


class DisplacementsPlotterConfig1D:
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


def plot_displacements_1D(
    ansatz: StandardAnsatz,
    length: float,
    youngs_modulus_list: list[float],
    traction: float,
    volume_force: float,
    output_subdir: str,
    project_directory: ProjectDirectory,
    config: DisplacementsPlotterConfig1D,
    device: Device,
) -> None:
    for youngs_modulus in youngs_modulus_list:
        _plot_one_displacements_1D(
            ansatz=ansatz,
            length=length,
            youngs_modulus=youngs_modulus,
            traction=traction,
            volume_force=volume_force,
            output_subdir=output_subdir,
            project_directory=project_directory,
            config=config,
            device=device,
        )


def _plot_one_displacements_1D(
    ansatz: StandardAnsatz,
    length: float,
    youngs_modulus: float,
    traction: float,
    volume_force: float,
    output_subdir: str,
    project_directory: ProjectDirectory,
    config: DisplacementsPlotterConfig1D,
    device: Device,
) -> None:
    num_points = 1000
    x_coor = torch.linspace(0.0, length, num_points).view((num_points, 1))
    solution = (
        calculate_displacements_solution_1D(
            coordinates=x_coor,
            length=length,
            youngs_modulus=youngs_modulus,
            traction=traction,
            volume_force=volume_force,
        )
        .detach()
        .numpy()
        .ravel()
    )
    x_E = torch.full((num_points, 1), youngs_modulus)
    x = torch.concat((x_coor, x_E), dim=1).to(device)
    prediction = ansatz(x).cpu().detach().numpy().ravel()
    x_coor = x_coor.detach().numpy().ravel()
    figure, axes = plt.subplots()
    axes.plot(x_coor, solution, label="solution")
    axes.plot(x_coor, prediction, label="prediction")
    axes.set_xlabel("coordinate [mm]", **config.font)
    axes.set_ylabel("displacements [mm]", **config.font)
    axes.tick_params(axis="both", which="minor", labelsize=config.minor_tick_label_size)
    axes.tick_params(axis="both", which="major", labelsize=config.major_tick_label_size)
    axes.legend(fontsize=config.font_size, loc="best")
    file_name = f"displacements_pinn_E_{youngs_modulus}.png"
    output_path = project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdir
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)
    plt.clf()


class BayesianDisplacementsPlotterConfig1D(DisplacementsPlotterConfig1D):
    def __init__(self) -> None:
        super().__init__()
        # standard deviation
        self.fill_color = "gray"
        self.alpha = 0.2


def plot_bayesian_displacements_1D(
    ansatz: BayesianAnsatz,
    parameter_samples: NPArray,
    length: float,
    youngs_modulus_list: list[float],
    traction: float,
    volume_force: float,
    output_subdir: str,
    project_directory: ProjectDirectory,
    config: BayesianDisplacementsPlotterConfig1D,
    device: Device,
) -> None:
    for youngs_modulus in youngs_modulus_list:
        _plot_one_bayesian_displacements_1D(
            ansatz=ansatz,
            parameter_samples=parameter_samples,
            length=length,
            youngs_modulus=youngs_modulus,
            traction=traction,
            volume_force=volume_force,
            output_subdir=output_subdir,
            project_directory=project_directory,
            config=config,
            device=device,
        )


def _plot_one_bayesian_displacements_1D(
    ansatz: BayesianAnsatz,
    parameter_samples: NPArray,
    length: float,
    youngs_modulus: float,
    traction: float,
    volume_force: float,
    output_subdir: str,
    project_directory: ProjectDirectory,
    config: BayesianDisplacementsPlotterConfig1D,
    device: Device,
) -> None:
    num_points = 1000
    x_coor = torch.linspace(0.0, length, num_points).view((num_points, 1))
    solution = (
        calculate_displacements_solution_1D(
            coordinates=x_coor,
            length=length,
            youngs_modulus=youngs_modulus,
            traction=traction,
            volume_force=volume_force,
        )
        .detach()
        .numpy()
        .ravel()
    )
    x_E = torch.full((num_points, 1), youngs_modulus)
    x = torch.concat((x_coor, x_E), dim=1).to(device)
    means, stds = _calculate_mean_and_std_for_bayesian_prediction(
        ansatz, parameter_samples, x
    )
    x_coor = x_coor.detach().numpy().ravel()
    figure, axes = plt.subplots()
    axes.plot(x_coor, solution, label="solution")
    axes.plot(x_coor, means, label="mean")
    axes.fill_between(
        x_coor, means - stds, means + stds, color=config.fill_color, alpha=config.alpha
    )
    axes.set_xlabel("coordinate [mm]", **config.font)
    axes.set_ylabel("displacements [mm]", **config.font)
    axes.tick_params(axis="both", which="minor", labelsize=config.minor_tick_label_size)
    axes.tick_params(axis="both", which="major", labelsize=config.major_tick_label_size)
    axes.legend(fontsize=config.font_size, loc="best")
    file_name = f"displacements_pinn_E_{youngs_modulus}.png"
    output_path = project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdir
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)
    plt.clf()


def _calculate_mean_and_std_for_bayesian_prediction(
    ansatz: BayesianAnsatz, parameter_samples: NPArray, x: Tensor
) -> tuple[NPArray, NPArray]:
    parameter_sample_np_list = list(parameter_samples)
    prediction_list = []
    for parameter_sample_np in parameter_sample_np_list:
        parameter_sample = torch.from_numpy(parameter_sample_np)
        ansatz.network.set_flattened_parameters(parameter_sample)
        prediction = ansatz(x).cpu().detach().numpy().ravel()
        prediction_list.append(prediction)
    predictions = np.stack(prediction_list, axis=0)
    means = np.mean(predictions, axis=0)
    standard_deviations = np.std(predictions, axis=0)
    return means, standard_deviations
