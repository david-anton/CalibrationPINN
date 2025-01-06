import math
from typing import TypeAlias, Union

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf as PltBackendPGF
from matplotlib.ticker import MaxNLocator
import numpy as np
import scipy.stats

from calibrationpinn.io import ProjectDirectory
from calibrationpinn.statistics.utility import (
    MomentsMultivariateNormal,
    MomentsUnivariateNormal,
)
from calibrationpinn.types import NPArray, PLTFigure

matplotlib.backend_bases.register_backend("pgf", PltBackendPGF)

TrueParameter: TypeAlias = Union[float, None]
TrueParametersTuple: TypeAlias = tuple[TrueParameter, ...]

cm_in_inches = 1 / 2.54  # centimeters in inches


class UnivariateNormalPlotterConfig:
    def __init__(self) -> None:
        # font sizes
        self.font_size = 6  # 14
        # font size in legend
        self.font = {"size": self.font_size}

        # title pad
        self.title_pad = 5

        # truth
        self.truth_color = "tab:orange"
        self.truth_linestyle = "solid"

        # confidence interval
        self.interval_num_stds = 1.959964  # quantile for 95% interval

        # histogram
        self.hist_bins = 128
        self.hist_range_in_std = 4
        self.hist_color = "tab:cyan"

        # pdf
        self.pdf_color = "tab:blue"
        self.pdf_linestyle = "solid"
        self.pdf_mean_color = "tab:red"
        self.pdf_mean_linestyle = "solid"
        self.pdf_interval_color = "tab:red"
        self.pdf_interval_linestyle = "dashed"

        # trace
        self.trace_color = "tab:blue"
        self.trace_linestyle = "solid"

        # major ticks
        self.major_tick_label_size = 6  # 12
        self.major_ticks_size = self.font_size
        self.major_ticks_width = 2

        # minor ticks
        self.minor_tick_label_size = 6  # 12
        self.minor_ticks_size = self.font_size
        self.minor_ticks_width = 1

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300
        self.figure_size = (6.5 * cm_in_inches, 5.0 * cm_in_inches)
        self.file_format = "pdf"


def plot_posterior_normal_distributions(
    parameter_names: tuple[str, ...],
    true_parameters: TrueParametersTuple,
    moments: MomentsMultivariateNormal,
    samples: NPArray,
    mcmc_algorithm: str,
    output_subdir: str,
    project_directory: ProjectDirectory,
) -> None:
    num_parameters = len(parameter_names)
    if num_parameters == 1:
        parameter_name = parameter_names[0]
        true_parameter = true_parameters[0]
        means = moments.mean
        covariance = moments.covariance
        mean_univariate = means[0]
        std_univariate = np.sqrt(covariance[0])
        moments_univariate = MomentsUnivariateNormal(
            mean=mean_univariate, standard_deviation=std_univariate
        )
        config = UnivariateNormalPlotterConfig()
        plot_univariate_normal_distribution(
            parameter_name,
            true_parameter,
            moments_univariate,
            samples,
            mcmc_algorithm,
            output_subdir,
            project_directory,
            config,
        )
    else:
        plot_multivariate_normal_distribution(
            parameter_names,
            true_parameters,
            moments,
            samples,
            mcmc_algorithm,
            output_subdir,
            project_directory,
        )


def plot_multivariate_normal_distribution(
    parameter_names: tuple[str, ...],
    true_parameters: TrueParametersTuple,
    moments: MomentsMultivariateNormal,
    samples: NPArray,
    mcmc_algorithm: str,
    output_subdir: str,
    project_directory: ProjectDirectory,
) -> None:
    num_parameters = len(parameter_names)
    means = moments.mean
    covariance = moments.covariance

    for parameter_idx in range(num_parameters):
        parameter_name = parameter_names[parameter_idx]
        true_parameter = true_parameters[parameter_idx]
        mean_univariate = means[parameter_idx]
        std_univariate = np.sqrt(covariance[parameter_idx, parameter_idx])
        moments_univariate = MomentsUnivariateNormal(
            mean=mean_univariate, standard_deviation=std_univariate
        )
        samples_univariate = samples[:, parameter_idx]
        config = UnivariateNormalPlotterConfig()
        plot_univariate_normal_distribution(
            parameter_name,
            true_parameter,
            moments_univariate,
            samples_univariate,
            mcmc_algorithm,
            output_subdir,
            project_directory,
            config,
        )


def plot_univariate_normal_distribution(
    parameter_name: str,
    true_parameter: TrueParameter,
    moments: MomentsUnivariateNormal,
    samples: NPArray,
    mcmc_algorithm: str,
    output_subdir: str,
    project_directory: ProjectDirectory,
    config: UnivariateNormalPlotterConfig,
) -> None:
    _plot_univariate_normal_distribution_histogram(
        parameter_name=parameter_name,
        true_parameter=true_parameter,
        moments=moments,
        samples=samples,
        mcmc_algorithm=mcmc_algorithm,
        output_subdir=output_subdir,
        project_directory=project_directory,
        config=config,
    )
    _plot_sampling_trace(
        parameter_name=parameter_name,
        true_parameter=true_parameter,
        moments=moments,
        samples=samples,
        mcmc_algorithm=mcmc_algorithm,
        output_subdir=output_subdir,
        project_directory=project_directory,
        config=config,
    )


def _plot_univariate_normal_distribution_histogram(
    parameter_name: str,
    true_parameter: TrueParameter,
    moments: MomentsUnivariateNormal,
    samples: NPArray,
    mcmc_algorithm: str,
    output_subdir: str,
    project_directory: ProjectDirectory,
    config: UnivariateNormalPlotterConfig,
) -> None:
    # title = "Posterior probability density"
    mean = moments.mean
    standard_deviation = moments.standard_deviation
    figure, axes = plt.subplots(figsize=config.figure_size)
    # Truth
    if true_parameter is not None:
        axes.axvline(
            x=true_parameter,
            color=config.truth_color,
            linestyle=config.truth_linestyle,
            label="NLS-FEM",
        )
    # Histogram
    range_hist = config.hist_range_in_std * standard_deviation
    axes.hist(
        samples,
        bins=config.hist_bins,
        range=(mean - range_hist, mean + range_hist),
        density=True,
        color=config.hist_color,
        label="samples",
    )
    # PDF
    x = np.linspace(
        start=mean - range_hist, stop=mean + range_hist, num=10000, endpoint=True
    )
    y = scipy.stats.norm.pdf(x, loc=mean, scale=standard_deviation)
    axes.plot(
        x,
        y,
        color=config.pdf_color,
        linestyle=config.pdf_linestyle,
        label="approx. PDF",
    )
    x_ticks = [
        mean - (config.interval_num_stds * standard_deviation),
        mean,
        mean + (config.interval_num_stds * standard_deviation),
    ]
    x_tick_labels = [
        (str(int(round(tick, 0))) if tick >= 100.0 else str(round(tick, 2)))
        for tick in x_ticks
    ]
    axes.axvline(
        x=mean,
        color=config.pdf_mean_color,
        linestyle=config.pdf_mean_linestyle,
        label="mean",
    )
    axes.axvline(
        x=mean - config.interval_num_stds * standard_deviation,
        color=config.pdf_interval_color,
        linestyle=config.pdf_interval_linestyle,
        label=r"$95\%$" + "-interval",
    )
    axes.axvline(
        x=mean + config.interval_num_stds * standard_deviation,
        color=config.pdf_interval_color,
        linestyle=config.pdf_interval_linestyle,
    )
    axes.set_xticks(x_ticks)
    axes.set_xticklabels(x_tick_labels)
    # axes.set_title(title, pad=config.title_pad, **config.font)
    axes.legend(fontsize=config.font_size, loc="best")
    axes.set_xlabel(infer_parameter_label(parameter_name), **config.font)
    axes.set_ylabel("probability density", **config.font)
    axes.tick_params(axis="both", which="minor", labelsize=config.minor_tick_label_size)
    axes.tick_params(axis="both", which="major", labelsize=config.major_tick_label_size)
    axes.ticklabel_format(
        axis="y",
        style="scientific",
        scilimits=(0, 0),
        useOffset=False,
        useMathText=True,
    )
    axes.yaxis.get_major_locator().set_params(integer=True)
    axes.yaxis.offsetText.set_fontsize(config.font_size)
    # Save plot
    file_name = f"estimated_pdf_{parameter_name.lower()}_{mcmc_algorithm.lower()}.{config.file_format}"
    output_path = project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdir
    )
    figure.savefig(
        output_path, format=config.file_format, dpi=config.dpi, bbox_inches="tight"
    )  # bbox_inches="tight"

    def save_figure_as_pgf(figure: PLTFigure) -> None:
        file_name_pgf = (
            f"estimated_pdf_{parameter_name.lower()}_{mcmc_algorithm.lower()}.pgf"
        )
        output_path_pgf = project_directory.create_output_file_path(
            file_name=file_name_pgf, subdir_name=output_subdir
        )
        figure.savefig(output_path_pgf, format="pgf")

    save_figure_as_pgf(figure)
    plt.close()


def _plot_sampling_trace(
    parameter_name: str,
    true_parameter: TrueParameter,
    moments: MomentsUnivariateNormal,
    samples: NPArray,
    mcmc_algorithm: str,
    output_subdir: str,
    project_directory: ProjectDirectory,
    config: UnivariateNormalPlotterConfig,
) -> None:
    title = "MCMC trace"
    mean = moments.mean
    standard_deviation = moments.standard_deviation
    figure, axes = plt.subplots()
    # Truth
    if true_parameter is not None:
        axes.axhline(
            y=true_parameter,
            color=config.truth_color,
            linestyle=config.truth_linestyle,
            label="truth",
        )
    # Trace plot
    num_samples = len(samples)
    x = np.linspace(1, num_samples, num=num_samples, dtype=int)
    y = samples
    axes.plot(
        x, y, color=config.pdf_color, linestyle=config.trace_linestyle, label="trace"
    )
    axes.axhline(
        y=mean,
        color=config.pdf_mean_color,
        linestyle=config.pdf_mean_linestyle,
        label=r"$\mu$",
    )
    axes.axhline(
        y=mean - config.interval_num_stds * standard_deviation,
        color=config.pdf_interval_color,
        linestyle=config.pdf_interval_linestyle,
        label=r"$95\%$" + " interval",
    )
    axes.axhline(
        y=mean + config.interval_num_stds * standard_deviation,
        color=config.pdf_interval_color,
        linestyle=config.pdf_interval_linestyle,
    )
    y_ticks = [
        mean - (config.interval_num_stds * standard_deviation),
        mean,
        mean + (config.interval_num_stds * standard_deviation),
    ]
    y_tick_labels = [
        str(round(tick, 2)) if tick >= 1.0 else str(round(tick, 6)) for tick in y_ticks
    ]
    axes.set_yticks(y_ticks)
    axes.set_yticklabels(y_tick_labels)
    min_sample = np.amin(samples)
    max_sample = np.amax(samples)
    min_y = (
        min_sample
        if true_parameter is None
        or (true_parameter is not None and min_sample < true_parameter)
        else true_parameter
    )
    max_y = (
        max_sample
        if true_parameter is None
        or (true_parameter is not None and max_sample > true_parameter)
        else true_parameter
    )
    range_y = max_y - min_y
    axes.set_ylim(min_y - 0.1 * range_y, max_y + 0.1 * range_y)
    x_ticks = [0, int(math.floor(num_samples / 2)), num_samples]
    x_tick_labels = [str(tick) for tick in x_ticks]
    axes.set_xticks(x_ticks)
    axes.set_xticklabels(x_tick_labels)
    axes.set_title(title, pad=config.title_pad, **config.font)
    axes.legend(fontsize=config.font_size, loc="best")
    axes.set_xlabel("samples", **config.font)
    axes.set_ylabel(infer_parameter_label(parameter_name), **config.font)
    axes.tick_params(axis="both", which="minor", labelsize=config.minor_tick_label_size)
    axes.tick_params(axis="both", which="major", labelsize=config.major_tick_label_size)
    file_name = f"sampling_trace_{parameter_name.lower()}_{mcmc_algorithm.lower()}.{config.file_format}"
    output_path = project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdir
    )
    figure.savefig(
        output_path, bbox_inches="tight", format=config.file_format, dpi=config.dpi
    )
    plt.close()


def infer_parameter_label(parameter_name: str) -> str:
    if parameter_name == "bulk modulus":
        return parameter_name + " " + r"$[Nmm^{-2}]$"
    elif parameter_name == "shear modulus":
        return parameter_name + " " + r"$[Nmm^{-2}]$"
    else:
        return parameter_name
