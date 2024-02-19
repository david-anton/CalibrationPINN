import os
from typing import TypeAlias

import numpy as np
import pandas as pd

from parametricpinn.calibration import calibrate
from parametricpinn.calibration.bayesianinference.mcmc import MCMCConfig, MCMCOutput
from parametricpinn.calibration.leastsquares import (
    LeastSquaresConfig,
    LeastSquaresOutput,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import PandasDataWriter
from parametricpinn.types import Device, NPArray

ParametersList: TypeAlias = list[NPArray]


def test_coverage() -> None:
    # plot distribution
    pass


def test_least_squares_calibration(
    calibration_configs: tuple[LeastSquaresConfig, ...],
    parameter_names: tuple[str],
    true_parameters: NPArray,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> None:
    test_subdir_name = "least_squares_test"

    def identify_parameters() -> NPArray:
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
        subdir_name = os.path.join(output_subdir, test_subdir_name)
        data_writer.write(
            data=data_frame, file_name=file_name, subdir_name=subdir_name, header=header
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
        subdir_name = os.path.join(output_subdir, test_subdir_name)
        data_writer.write(
            data=data,
            file_name=file_name,
            subdir_name=subdir_name,
            header=header,
            index=True,
        )

    identified_parameters_list = identify_parameters()
    abs_relative_errors = calculate_absolute_relative_errors(identified_parameters_list)
    save_results(identified_parameters_list, abs_relative_errors)
    save_results_summary(abs_relative_errors)
