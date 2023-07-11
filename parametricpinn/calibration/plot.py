import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from parametricpinn.calibration.statistics import (
    MomentsMultivariateNormal,
    MomentsUnivariateNormal,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.types import NPArray


class UnivariateNormalPlotterConfig:
    def __init__(self) -> None:
        # font sizes
        self.label_size = 20
        # font size in legend
        self.font_size = 16
        self.font = {"size": self.label_size}

        # histogram
        self.bins = 30
        self.range_in_std = 5

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


def plot_posterior_normal_distributions(
    parameter_names: tuple[str, ...],
    moments: MomentsMultivariateNormal,
    samples: NPArray,
    output_subdir: str,
    project_directory: ProjectDirectory,
) -> None:
    num_parameters = len(parameter_names)
    if num_parameters == 1:
        parameter_name = parameter_names[0]
        means = moments.mean
        covariance = moments.covariance
        mean_univariate = means[0]
        std_univariate = np.sqrt(covariance[0])
        moments_univariate = MomentsUnivariateNormal(
            mean=mean_univariate, standard_deviation=std_univariate
        )
        config = UnivariateNormalPlotterConfig()
        plot_univariate_univariate_normal_distribution(
            parameter_name,
            moments_univariate,
            samples,
            output_subdir,
            project_directory,
            config,
        )
    else:
        plot_multivariate_normal_distribution(
            parameter_names, moments, samples, output_subdir, project_directory
        )


def plot_multivariate_normal_distribution(
    parameter_names: tuple[str, ...],
    moments: MomentsMultivariateNormal,
    samples: NPArray,
    output_subdir: str,
    project_directory: ProjectDirectory,
) -> None:
    num_parameters = len(parameter_names)
    means = moments.mean
    covariance = moments.covariance


    for parameter_idx in range(num_parameters):
        parameter_name = parameter_names[parameter_idx]
        mean_univariate = means[parameter_idx]
        std_univariate = np.sqrt(covariance[parameter_idx, parameter_idx])
        moments_univariate = MomentsUnivariateNormal(
            mean=mean_univariate, standard_deviation=std_univariate
        )
        samples_univariate = samples[:, parameter_idx]
        config = UnivariateNormalPlotterConfig()
        plot_univariate_univariate_normal_distribution(
            parameter_name,
            moments_univariate,
            samples_univariate,
            output_subdir,
            project_directory,
            config,
        )


def plot_univariate_univariate_normal_distribution(
    parameter_name: str,
    moments: MomentsUnivariateNormal,
    samples: NPArray,
    output_subdir: str,
    project_directory: ProjectDirectory,
    config: UnivariateNormalPlotterConfig,
) -> None:
    mean = moments.mean
    standard_deviation = moments.standard_deviation
    figure, axes = plt.subplots()
    # Histogram
    range_hist = config.range_in_std * standard_deviation
    axes.hist(samples, bins=config.bins, range=(mean - range_hist, mean + range_hist))
    # PDF
    x = np.linspace(
        start=mean - range_hist, stop=mean + range_hist, num=10000, endpoint=True
    )
    y = scipy.stats.norm.pdf(x, loc=mean, scale=standard_deviation)
    axes.plot(x, y, "r-", label="PDF")
    axes.legend(fontsize=config.font_size, loc="best")
    axes.set_xlabel(parameter_name, **config.font)
    axes.set_ylabel("probability density", **config.font)
    axes.tick_params(axis="both", which="minor", labelsize=config.minor_tick_label_size)
    axes.tick_params(axis="both", which="major", labelsize=config.major_tick_label_size)
    file_name = f"estimated_pdf_{parameter_name.lower()}.png"
    output_path = project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdir
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=config.dpi)
    plt.clf()
