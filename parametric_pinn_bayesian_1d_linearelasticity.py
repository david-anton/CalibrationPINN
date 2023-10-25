from datetime import date

import torch

from parametricpinn.ansatz import (
    BayesianAnsatz,
    create_bayesian_normalized_hbc_ansatz_stretched_rod,
)
from parametricpinn.data.trainingdata_linearelasticity_1d import (
    StretchedRodTrainingDataset1D,
    StretchedRodTrainingDataset1DConfig,
    create_training_dataset,
)
from parametricpinn.data.validationdata_linearelasticity_1d import (
    calculate_displacements_solution,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.network import BFFNN
from parametricpinn.postprocessing.plot import (
    BayesianDisplacementsPlotterConfig1D,
    plot_bayesian_displacements_1d,
)
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.training.training_bayesian_linearelasticity_stretchedrod import (
    MeasurementsStds,
    ParameterPriorStds,
    TrainingConfiguration,
    train_parametric_pinn,
)
from parametricpinn.types import Tensor

### Configuration
retrain_parametric_pinn = True
# Set up
length = 100.0
traction = 1.0
volume_force = 1.0
min_youngs_modulus = 165000.0
max_youngs_modulus = 255000.0
displacement_left = 0.0
# Network
layer_sizes = [2, 8, 8, 1]
std_weight = 1.0
std_bias = 1.0
# Ansatz
distance_function = "normalized linear"
# Training
num_samples_train = 128
num_points_pde = 128
training_batch_size = num_samples_train
number_mcmc_iterations = int(1e4)
# Data
std_pde = 0.01
std_stress_bc = 0.01
# Output
current_date = date.today().strftime("%Y%m%d")
output_date = current_date
output_subdirectory = f"{output_date}_Parametric_PINN_Bayesian_1D"


### Set up simulation
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)


def create_datasets() -> StretchedRodTrainingDataset1D:
    config_training_dataset = StretchedRodTrainingDataset1DConfig(
        length=length,
        traction=traction,
        volume_force=volume_force,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        num_points_pde=num_points_pde,
        num_samples=num_samples_train,
    )
    train_dataset = create_training_dataset(config_training_dataset)
    return train_dataset


def create_ansatz() -> BayesianAnsatz:
    def _determine_normalization_values() -> dict[str, Tensor]:
        min_coordinate = 0.0
        max_coordinate = length
        min_inputs = torch.tensor([min_coordinate, min_youngs_modulus])
        max_inputs = torch.tensor([max_coordinate, max_youngs_modulus])
        min_displacement = displacement_left
        max_displacement = calculate_displacements_solution(
            coordinates=max_coordinate,
            length=length,
            youngs_modulus=min_youngs_modulus,
            traction=traction,
            volume_force=volume_force,
        )
        min_output = torch.tensor([min_displacement])
        max_output = torch.tensor([max_displacement])
        return {
            "min_inputs": min_inputs.to(device),
            "max_inputs": max_inputs.to(device),
            "min_output": min_output.to(device),
            "max_output": max_output.to(device),
        }

    normalization_values = _determine_normalization_values()
    network = BFFNN(layer_sizes=layer_sizes)
    return create_bayesian_normalized_hbc_ansatz_stretched_rod(
        displacement_left=torch.tensor([displacement_left]).to(device),
        min_inputs=normalization_values["min_inputs"],
        max_inputs=normalization_values["max_inputs"],
        min_outputs=normalization_values["min_output"],
        max_outputs=normalization_values["max_output"],
        network=network,
        distance_function_type=distance_function
    ).to(device)


ansatz = create_ansatz()


def training_step() -> None:
    parameter_prior_stds = ParameterPriorStds(weight=std_weight, bias=std_bias)
    measurements_standard_deviations = MeasurementsStds(
        pde=std_pde, stress_bc=std_stress_bc
    )

    train_config = TrainingConfiguration(
        ansatz=ansatz,
        parameter_prior_stds=parameter_prior_stds,
        training_dataset=training_dataset,
        measurements_standard_deviations=measurements_standard_deviations,
        number_mcmc_iterations=number_mcmc_iterations,
        output_subdirectory=output_subdirectory,
        project_directory=project_directory,
        device=device,
    )

    def _plot_exemplary_displacements() -> None:
        displacements_plotter_config = BayesianDisplacementsPlotterConfig1D()

        plot_bayesian_displacements_1d(
            ansatz=ansatz,
            parameter_samples=parameter_samples,
            length=length,
            youngs_modulus_list=[187634, 238695],
            traction=traction,
            volume_force=volume_force,
            output_subdir=output_subdirectory,
            project_directory=project_directory,
            config=displacements_plotter_config,
            device=device,
        )

    _, parameter_samples = train_parametric_pinn(train_config)
    _plot_exemplary_displacements()


if retrain_parametric_pinn:
    training_dataset = create_datasets()
    training_step()
