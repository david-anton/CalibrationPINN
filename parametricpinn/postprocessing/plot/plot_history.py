import matplotlib.pyplot as plt
import torch

from parametricpinn.data import calculate_displacements_solution_1D
from parametricpinn.io import ProjectDirectory
from parametricpinn.types import Module


class HistoryPlotterConfig:
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


def plot_loss_history(
    loss_hist_pde: list[float],
    loss_hist_stress_bc: list[float],
    file_name: str,
    output_subdir: str,
    project_directory: ProjectDirectory,
    config: HistoryPlotterConfig,
) -> None:
    figure, axes = plt.subplots()
    axes.set_title("Loss history", **config.font)
    axes.plot(loss_hist_pde, label="loss PDE")
    axes.plot(loss_hist_stress_bc, label="loss stress BC")
    axes.set_yscale("log")
    axes.set_ylabel("MSE", **config.font)
    axes.set_xlabel("epoch", **config.font)
    axes.tick_params(axis="both", which="minor", labelsize=config.minor_tick_label_size)
    axes.tick_params(axis="both", which="major", labelsize=config.major_tick_label_size)
    axes.legend(fontsize=config.font_size, loc="best")
    output_path = project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdir
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)
    plt.clf()


def plot_valid_history(
    valid_epochs: list[int],
    valid_hist: list[float],
    valid_metric: str,
    file_name: str,
    output_subdir: str,
    project_directory: ProjectDirectory,
    config: HistoryPlotterConfig,
) -> None:
    figure, axes = plt.subplots()
    axes.set_title(valid_metric, **config.font)
    axes.plot(valid_epochs, valid_hist, label=valid_metric)
    axes.set_yscale("log")
    axes.set_xlabel("epoch", **config.font)
    axes.tick_params(axis="both", which="minor", labelsize=config.minor_tick_label_size)
    axes.tick_params(axis="both", which="major", labelsize=config.major_tick_label_size)
    output_path = project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdir
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)
    plt.clf()
