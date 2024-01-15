from typing import TypeAlias, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from parametricpinn.io import ProjectDirectory
from parametricpinn.statistics.utility import (
    MomentsMultivariateNormal,
    MomentsUnivariateNormal,
)
from parametricpinn.types import NPArray

TrueParameter: TypeAlias = Union[float, None]
TrueParametersTuple: TypeAlias = tuple[TrueParameter, ...]


class UnivariateNormalPlotterConfig:
    def __init__(self) -> None:
        # font sizes
        self.label_size = 16
        # font size in legend
        self.font_size = 14
        self.font = {"size": self.label_size}

        # truth
        self.truth_color = "tab:orange"
        self.truth_linestyle = "solid"

        # histogram
        self.hist_bins = 128
        self.hist_range_in_std = 5
        self.hist_color = "tab:cyan"

        # pdf
        self.pdf_color = "tab:blue"
        self.pdf_linestyle = "solid"
        self.pdf_mean_color = "tab:red"
        self.pdf_mean_linestyle = "solid"
        self.pdf_interval_color = "tab:red"
        self.pdf_interval_linestyle = "dashed"

        # confidence interval
        self.interval_num_stds = 1.959964  # quantile for 95% interval

        # major ticks
        self.major_tick_label_size = 12
        self.major_ticks_size = self.font_size
        self.major_ticks_width = 2

        # minor ticks
        self.minor_tick_label_size = 12
        self.minor_ticks_size = 12
        self.minor_ticks_width = 1

        # scientific notation
        self.scientific_notation_size = self.font_size

        # save options
        self.dpi = 300


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
    mean = moments.mean
    standard_deviation = moments.standard_deviation
    figure, axes = plt.subplots()
    # Truth
    if true_parameter is not None:
        axes.axvline(
            x=true_parameter,
            color=config.truth_color,
            linestyle=config.truth_linestyle,
            label="truth",
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
    axes.plot(x, y, color=config.pdf_color, linestyle=config.pdf_linestyle, label="pdf")
    x_ticks = [
        mean - (config.interval_num_stds * standard_deviation),
        mean,
        mean + (config.interval_num_stds * standard_deviation),
    ]
    x_tick_labels = [
        str(round(tick, 2)) if tick >= 1.0 else str(round(tick, 4)) for tick in x_ticks
    ]
    axes.axvline(
        x=mean,
        color=config.pdf_mean_color,
        linestyle=config.pdf_mean_linestyle,
        label=r"$\mu$",
    )
    axes.axvline(
        x=mean - config.interval_num_stds * standard_deviation,
        color=config.pdf_interval_color,
        linestyle=config.pdf_interval_linestyle,
        label=r"$95\% interval$",
    )
    axes.axvline(
        x=mean + config.interval_num_stds * standard_deviation,
        color=config.pdf_interval_color,
        linestyle=config.pdf_interval_linestyle,
    )
    axes.set_xticks(x_ticks)
    axes.set_xticklabels(x_tick_labels)
    axes.legend(fontsize=config.font_size, loc="best")
    axes.set_xlabel(parameter_name, **config.font)
    axes.set_ylabel("probability density", **config.font)
    axes.tick_params(axis="both", which="minor", labelsize=config.minor_tick_label_size)
    axes.tick_params(axis="both", which="major", labelsize=config.major_tick_label_size)
    file_name = f"estimated_pdf_{parameter_name.lower()}_{mcmc_algorithm.lower()}.png"
    output_path = project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdir
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)
    plt.clf()
