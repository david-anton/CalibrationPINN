import os
from typing import TypeAlias

import numpy as np
import pandas as pd

from parametricpinn.calibration import calibrate
from parametricpinn.calibration.bayesianinference.mcmc import MCMCConfig
from parametricpinn.calibration.bayesianinference.mcmc.base import (
    MomentsMultivariateNormal,
)
from parametricpinn.calibration.bayesianinference.plot import (
    plot_posterior_normal_distributions,
)
from parametricpinn.calibration.leastsquares import LeastSquaresConfig
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import PandasDataWriter
from parametricpinn.types import Device, NPArray

ParametersList: TypeAlias = list[NPArray]


def test_coverage(
    calibration_configs: tuple[MCMCConfig, ...],
    parameter_names: tuple[str, ...],
    true_parameters: NPArray,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> None:
    num_considered_parameters = np.shape(true_parameters)[1]
    considered_parameter_names = parameter_names[:num_considered_parameters]
    credible_standard_factor = 1.959963984540

    def perform_bayesian_inference() -> tuple[NPArray, NPArray, NPArray]:
        def calibrate_once(
            calibration_config: MCMCConfig,
            true_parameters_tuple: tuple[float, ...],
            output_subdir_case: str,
        ) -> tuple[NPArray, NPArray]:
            moments, samples = calibrate(calibration_config, device)
            postprocess_one_mcmc_result(
                calibration_config=calibration_config,
                moments=moments,
                samples=samples,
                true_parameters_tuple=true_parameters_tuple,
                output_subdir=output_subdir_case,
            )
            means = moments.mean
            if np.shape(moments.covariance) == (1,):
                standard_deviations = np.sqrt(moments.covariance)
            else:
                standard_deviations = np.sqrt(np.diagonal(moments.covariance))
            return means, standard_deviations

        def postprocess_one_mcmc_result(
            true_parameters_tuple: tuple[float, ...],
            moments: MomentsMultivariateNormal,
            samples: NPArray,
            calibration_config: MCMCConfig,
            output_subdir: str,
        ):
            extended_true_parameters_tuple = true_parameters_tuple + tuple(
                None for _ in range(len(parameter_names) - np.shape(true_parameters)[1])
            )
            plot_posterior_normal_distributions(
                parameter_names=parameter_names,
                true_parameters=extended_true_parameters_tuple,
                moments=moments,
                samples=samples,
                mcmc_algorithm=calibration_config.algorithm_name,
                output_subdir=output_subdir,
                project_directory=project_directory,
            )

        means_list: list[NPArray] = []
        standard_deviations_list: list[NPArray] = []
        for count, config in enumerate(calibration_configs):
            true_parameters_tuple = tuple(true_parameters[count, :])
            output_subdir_case = os.path.join(output_subdir, f"case_{count:02d}")
            means, standard_deviations = calibrate_once(
                calibration_config=config,
                true_parameters_tuple=true_parameters_tuple,
                output_subdir_case=output_subdir_case,
            )
            means_list.append(means[:num_considered_parameters])
            standard_deviations_list.append(
                standard_deviations[:num_considered_parameters]
            )

        stacked_means = np.stack(tuple(means_list), axis=0)
        stacked_biases = stacked_means - true_parameters
        stacked_standard_deviations = np.stack(tuple(standard_deviations_list), axis=0)
        return stacked_means, stacked_biases, stacked_standard_deviations

    def test_credibe_intervals(means: NPArray, standard_deviations: NPArray) -> NPArray:
        credible_interval = credible_standard_factor * standard_deviations
        lower_bound = means - credible_interval
        upper_bound = means + credible_interval
        condition = np.logical_and(
            true_parameters >= lower_bound, true_parameters <= upper_bound
        )
        coverage_individual = np.where(condition, True, False)
        coverage_combined = coverage_individual.all(axis=1, keepdims=True)
        return np.concatenate((coverage_individual, coverage_combined), axis=1)

    def save_results(
        means: NPArray,
        biases: NPArray,
        standard_deviations: NPArray,
        credible_test_results: NPArray,
    ) -> None:
        def compile_header() -> tuple[str, ...]:
            true_parameter_names = [f"true {p}" for p in considered_parameter_names]
            mean_names = [f"mean {p}" for p in considered_parameter_names]
            bias_names = [f"bias {p}" for p in considered_parameter_names]
            standard_deviation_names = [
                f"standard deviation {p}" for p in considered_parameter_names
            ]
            credible_test_names = [
                f"is {p} in credible interval" for p in considered_parameter_names
            ] + ["are all parameters in credible interval"]
            return tuple(
                true_parameter_names
                + mean_names
                + bias_names
                + standard_deviation_names
                + credible_test_names
            )

        def compile_results() -> NPArray:
            return np.concatenate(
                (
                    true_parameters,
                    means,
                    biases,
                    standard_deviations,
                    credible_test_results,
                ),
                axis=1,
            )

        header = compile_header()
        results = compile_results()
        data_frame = pd.DataFrame(results, columns=header)
        data_writer = PandasDataWriter(project_directory)
        file_name = "results.csv"
        data_writer.write(
            data=data_frame,
            file_name=file_name,
            subdir_name=output_subdir,
            header=header,
        )

    def save_statistics_summary(biases: NPArray, standard_deviations: NPArray) -> None:
        def compile_header() -> tuple[str, ...]:
            mean_bias_names = [f"mean bias {p}" for p in considered_parameter_names]
            mean_stdandard_deviation_names = [
                f"mean standard deviation {p}" for p in considered_parameter_names
            ]
            return tuple(mean_bias_names + mean_stdandard_deviation_names)

        def compile_results() -> NPArray:
            mean_biases = np.mean(biases, axis=0, keepdims=True)
            mean_standard_deviations = np.mean(
                standard_deviations, axis=0, keepdims=True
            )
            return np.concatenate(
                (mean_biases, mean_standard_deviations),
                axis=1,
            )

        header = compile_header()
        results = compile_results()
        data_frame = pd.DataFrame(results, columns=header)
        data_writer = PandasDataWriter(project_directory)
        file_name = "summary_statistics.csv"
        data_writer.write(
            data=data_frame,
            file_name=file_name,
            subdir_name=output_subdir,
            header=header,
        )

    def save_coverage_test_summary(credible_test_results: NPArray) -> None:
        def compile_header() -> tuple[str, ...]:
            return considered_parameter_names + ("combined",)

        def compile_index() -> tuple[str]:
            return ("coverage [%]",)

        def compile_coverage_results() -> NPArray:
            return np.mean(credible_test_results, axis=0, keepdims=True) * 100

        header = compile_header()
        index = compile_index()
        coverage_results = compile_coverage_results()
        data = pd.DataFrame(coverage_results, index=index, columns=header)
        data_writer = PandasDataWriter(project_directory)
        file_name = "summary_coverage_test.csv"
        data_writer.write(
            data=data,
            file_name=file_name,
            subdir_name=output_subdir,
            header=header,
            index=True,
        )

    means, biases, standard_deviations = perform_bayesian_inference()
    credible_test_results = test_credibe_intervals(means, standard_deviations)
    save_results(means, biases, standard_deviations, credible_test_results)
    save_statistics_summary(biases, standard_deviations)
    save_coverage_test_summary(credible_test_results)


def test_least_squares_calibration(
    calibration_configs: tuple[LeastSquaresConfig, ...],
    parameter_names: tuple[str, ...],
    true_parameters: NPArray,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> None:
    def calibrate_model() -> NPArray:
        def calibrate_once(calibration_config: LeastSquaresConfig) -> NPArray:
            identified_parameters, _ = calibrate(calibration_config, device)
            return identified_parameters

        return np.stack(
            tuple(calibrate_once(config) for config in calibration_configs), axis=0
        )

    def calculate_absolute_relative_errors(identified_parameters: NPArray) -> NPArray:
        return (
            np.absolute(identified_parameters - true_parameters) / true_parameters
        ) * 100.0

    def save_results(identified_parameters: NPArray, relative_errors: NPArray) -> None:
        def compile_header() -> tuple[str, ...]:
            true_parameter_names = [f"true {p}" for p in parameter_names]
            identified_parameter_names = [f"identified {p}" for p in parameter_names]
            relative_error_names = [f"abs. rel. error {p} [%]" for p in parameter_names]
            return tuple(
                true_parameter_names + identified_parameter_names + relative_error_names
            )

        def compile_results() -> NPArray:
            return np.concatenate(
                (true_parameters, identified_parameters, relative_errors), axis=1
            )

        header = compile_header()
        results = compile_results()
        data_frame = pd.DataFrame(results, columns=header)
        data_writer = PandasDataWriter(project_directory)
        file_name = "results.csv"
        data_writer.write(
            data=data_frame,
            file_name=file_name,
            subdir_name=output_subdir,
            header=header,
        )

    def save_results_summary(relative_errors: NPArray) -> None:
        def compile_header() -> tuple[str, ...]:
            return ("mean", "standard error")

        def compile_index() -> tuple[str, ...]:
            return tuple(f"abs. rel. error {p} [%]" for p in parameter_names)

        def compile_results() -> NPArray:
            means = np.mean(relative_errors, axis=0)
            std_errors = np.std(relative_errors, axis=0, ddof=0) / np.sqrt(
                len(relative_errors)
            )
            shape = (-1, 1)
            return np.concatenate(
                (np.reshape(means, shape), np.reshape(std_errors, shape)), axis=1
            )

        header = compile_header()
        index = compile_index()
        results = compile_results()
        data = pd.DataFrame(results, index=index, columns=header)
        data_writer = PandasDataWriter(project_directory)
        file_name = "summary.csv"
        data_writer.write(
            data=data,
            file_name=file_name,
            subdir_name=output_subdir,
            header=header,
            index=True,
        )

    identified_parameters = calibrate_model()
    abs_rel_errors = calculate_absolute_relative_errors(identified_parameters)
    save_results(identified_parameters, abs_rel_errors)
    save_results_summary(abs_rel_errors)
