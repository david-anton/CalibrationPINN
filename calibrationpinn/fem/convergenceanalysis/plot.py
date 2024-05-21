import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from calibrationpinn.io import ProjectDirectory


def plot_error_convergence_analysis(
    error_record: list[float],
    element_size_record: list[float],
    error_norm: str,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
) -> None:
    figure, axes = plt.subplots()
    log_element_size = np.log(np.array(element_size_record))
    log_error = np.log(np.array(error_record))

    slope, intercept, _, _, _ = stats.linregress(log_element_size, log_error)
    regression_log_error = slope * log_element_size + intercept

    axes.plot(log_element_size, log_error, "ob", label="simulation")
    axes.plot(log_element_size, regression_log_error, "--r", label="regression")
    axes.set_xlabel("log element size")
    axes.set_ylabel(f"log error {error_norm}")
    axes.set_title("Convergence analysis")
    axes.legend(loc="best")
    axes.text(
        log_element_size[1],
        log_error[-1],
        f"convergence rate: {slope:.6}",
        style="italic",
        bbox={"facecolor": "red", "alpha": 0.5, "pad": 10},
    )
    file_name = f"convergence_log_{error_norm}_error.png"
    output_path = project_directory.create_output_file_path(
        file_name, output_subdirectory
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=256)
    plt.clf()
