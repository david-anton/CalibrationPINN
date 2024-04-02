import os
from datetime import date
from time import perf_counter

import torch

from parametricpinn.ansatz import (
    BayesianAnsatz,
    StandardAnsatz,
    create_bayesian_normalized_hbc_ansatz_stretched_rod,
    create_standard_normalized_hbc_ansatz_stretched_rod,
)
from parametricpinn.bayesian.likelihood import Likelihood
from parametricpinn.bayesian.prior import create_univariate_normal_distributed_prior
from parametricpinn.calibration import (
    CalibrationData,
    CalibrationDataGenerator1D,
    EfficientNUTSConfig,
    HamiltonianConfig,
    MetropolisHastingsConfig,
    test_coverage,
)
from parametricpinn.calibration.bayesianinference.likelihoods import (
    create_bayesian_ppinn_likelihood_for_noise,
)
from parametricpinn.calibration.utility import load_model
from parametricpinn.data.parameterssampling import sample_uniform_grid
from parametricpinn.data.trainingdata_1d import (
    StretchedRodTrainingDataset1D,
    StretchedRodTrainingDataset1DConfig,
    create_training_dataset,
)
from parametricpinn.data.validationdata_linearelasticity_1d import (
    StretchedRodSimulationDatasetLinearElasticity1D,
    StretchedRodSimulationDatasetLinearElasticity1DConfig,
    calculate_linear_elastic_displacements_solution,
    create_simulation_dataset,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import CSVDataReader
from parametricpinn.network import BFFNN, FFNN
from parametricpinn.postprocessing.plot import (
    BayesianDisplacementsPlotterConfig1D,
    DisplacementsPlotterConfig1D,
    plot_bayesian_displacements_1d,
    plot_displacements_1d,
)
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.training.training_bayesian_linearelasticity_stretchedrod import (
    BayesianTrainingConfiguration,
    MeasurementsStds,
    ParameterPriorStds,
    train_bayesian_parametric_pinn,
)
from parametricpinn.training.training_standard_linearelasticity_stretchedrod import (
    StandardTrainingConfiguration,
    train_parametric_pinn,
)
from parametricpinn.types import NPArray, Tensor

### Configuration
pretrain_parametric_pinn = False
# Set up
length = 100.0
traction = 1.0
volume_force = 1.0
min_youngs_modulus = 180000.0
max_youngs_modulus = 240000.0
displacement_left = 0.0
# Network
layer_sizes = [2, 8, 8, 1]
prior_stddev_weight = 1.0
prior_stddev_bias = 1.0
# Ansatz
distance_function = "normalized linear"
# Training
num_samples_train = 8  # 64
num_points_pde = 8  # 128
training_batch_size = num_samples_train
number_pretraining_epochs = 400
number_mcmc_iterations = int(1e3)
mcmc_algorithm_training = "metropolis hastings"
# Validation
num_samples_valid = 64
valid_interval = 1
num_points_valid = 512
batch_size_valid = num_samples_valid
# Loss errors
std_pde_not_pretrained = 1e-1
std_stress_bc_not_pretrained = 1e-1
std_pde_pretrained = 1e-2
std_stress_bc_pretrained = 1e-2
# Calibration
use_random_walk_metropolis_hasting = False
use_hamiltonian = False
use_efficient_nuts = False
# Output
current_date = date.today().strftime("%Y%m%d")
output_date = current_date
output_subdirectory = (
    f"{output_date}_Parametric_PINN_Bayesian_1D_MH_proposaldensity_1e-3_nopretraining"
)


### Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

### Set up simulation
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)


def create_datasets() -> (
    tuple[
        StretchedRodTrainingDataset1D, StretchedRodSimulationDatasetLinearElasticity1D
    ]
):
    parameter_samples = sample_uniform_grid(
        min_parameters=[min_youngs_modulus],
        max_parameters=[max_youngs_modulus],
        num_steps=[num_samples_train],
        device=device,
    )
    config_training_dataset = StretchedRodTrainingDataset1DConfig(
        parameters_samples=parameter_samples,
        length=length,
        traction=traction,
        volume_force=volume_force,
        num_points_pde=num_points_pde,
    )
    train_dataset = create_training_dataset(config_training_dataset)
    config_validation_dataset = StretchedRodSimulationDatasetLinearElasticity1DConfig(
        length=length,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        traction=traction,
        volume_force=volume_force,
        num_points=num_points_valid,
        num_samples=num_samples_valid,
    )
    valid_dataset = create_simulation_dataset(config_validation_dataset)
    return train_dataset, valid_dataset


def determine_normalization_values() -> dict[str, Tensor]:
    min_coordinate = 0.0
    max_coordinate = length
    min_inputs = torch.tensor([min_coordinate, min_youngs_modulus])
    max_inputs = torch.tensor([max_coordinate, max_youngs_modulus])
    min_displacement = displacement_left
    max_displacement = calculate_linear_elastic_displacements_solution(
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


def create_bayesian_ansatz() -> BayesianAnsatz:
    normalization_values = determine_normalization_values()
    network = BFFNN(layer_sizes=layer_sizes)
    return create_bayesian_normalized_hbc_ansatz_stretched_rod(
        min_inputs=normalization_values["min_inputs"],
        max_inputs=normalization_values["max_inputs"],
        min_outputs=normalization_values["min_output"],
        max_outputs=normalization_values["max_output"],
        network=network,
        distance_function_type=distance_function,
        device=device,
    ).to(device)


def create_standard_ansatz() -> StandardAnsatz:
    normalization_values = determine_normalization_values()
    network = FFNN(layer_sizes=layer_sizes)
    return create_standard_normalized_hbc_ansatz_stretched_rod(
        min_inputs=normalization_values["min_inputs"],
        max_inputs=normalization_values["max_inputs"],
        min_outputs=normalization_values["min_output"],
        max_outputs=normalization_values["max_output"],
        network=network,
        distance_function_type=distance_function,
        device=device,
    ).to(device)


def pretraining_step() -> None:
    ansatz = create_standard_ansatz()
    train_config = StandardTrainingConfiguration(
        ansatz=ansatz,
        training_dataset=training_dataset,
        number_training_epochs=number_pretraining_epochs,
        training_batch_size=training_batch_size,
        validation_dataset=validation_dataset,
        output_subdirectory=output_subdirectory,
        project_directory=project_directory,
        device=device,
    )

    def _plot_exemplary_displacements() -> None:
        displacements_plotter_config = DisplacementsPlotterConfig1D()

        plot_displacements_1d(
            ansatz=ansatz,
            length=length,
            youngs_modulus_list=[187634, 238695],
            traction=traction,
            volume_force=volume_force,
            output_subdir=output_subdirectory,
            project_directory=project_directory,
            config=displacements_plotter_config,
            device=device,
        )

    train_parametric_pinn(train_config=train_config)
    _plot_exemplary_displacements()


def bayesian_training_step(is_pretrained: bool = False) -> None:
    if is_pretrained:
        standard_ansatz = create_standard_ansatz()
        name_model_parameters_file = "model_parameters"
        model = load_model(
            model=standard_ansatz,
            name_model_parameters_file=name_model_parameters_file,
            input_subdir=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )
        initial_parameters = model.network.get_flattened_parameters()
        measurements_standard_deviations = MeasurementsStds(
            pde=std_pde_pretrained, stress_bc=std_stress_bc_pretrained
        )
    else:
        initial_parameters = None
        measurements_standard_deviations = MeasurementsStds(
            pde=std_pde_not_pretrained, stress_bc=std_stress_bc_not_pretrained
        )

    ansatz = create_bayesian_ansatz()
    parameter_prior_stds = ParameterPriorStds(
        weight=prior_stddev_weight, bias=prior_stddev_bias
    )

    train_config = BayesianTrainingConfiguration(
        ansatz=ansatz,
        initial_parameters=initial_parameters,
        parameter_prior_stds=parameter_prior_stds,
        training_dataset=training_dataset,
        measurements_standard_deviations=measurements_standard_deviations,
        mcmc_algorithm=mcmc_algorithm_training,
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

    _, parameter_samples = train_bayesian_parametric_pinn(train_config)
    _plot_exemplary_displacements()


def calibration_step() -> None:
    print("Start calibration ...")
    exact_youngs_modulus = 195000
    num_test_cases = 1
    num_data_sets = 1
    num_data_points = 32
    std_noise = 5 * 1e-4

    def generate_calibration_data() -> tuple[CalibrationData, NPArray]:
        true_parameters = torch.tensor([exact_youngs_modulus])
        calibration_data_generator = CalibrationDataGenerator1D(
            true_parameters=true_parameters,
            traction=traction,
            volume_force=volume_force,
            length=length,
            num_cases=num_test_cases,
            num_data_sets=num_data_sets,
            num_data_points=num_data_points,
            std_noise=std_noise,
            solution_func=calculate_linear_elastic_displacements_solution,
            device=device,
        )
        calibration_data = calibration_data_generator.generate_data()
        return calibration_data[0], true_parameters.detach().cpu().numpy()

    model = create_bayesian_ansatz()
    csv_data_reader = CSVDataReader(project_directory)
    name_model_parameters_file = "bayesian_model_parameters"
    model_parameter_samples = torch.from_numpy(
        csv_data_reader.read(
            file_name=name_model_parameters_file,
            subdir_name=output_subdirectory,
            read_from_output_dir=True,
            header=None,
        )
    ).to(device)

    data, true_parameter = generate_calibration_data()

    likelihood = create_bayesian_ppinn_likelihood_for_noise(
        model=model,
        model_parameter_samples=model_parameter_samples,
        data=data,
        device=device,
    )

    prior_mean_youngs_modulus = 210000
    prior_std_youngs_modulus = 10000
    prior = create_univariate_normal_distributed_prior(
        mean=prior_mean_youngs_modulus,
        standard_deviation=prior_std_youngs_modulus,
        device=device,
    )

    parameter_names = ("Youngs modulus",)
    initial_parameters = torch.tensor([prior_mean_youngs_modulus], device=device)

    output_subdir_calibration = os.path.join(
        output_subdirectory, "calibration_with_model_error"
    )

    def set_up_metropolis_hastings_config(
        likelihood: Likelihood,
    ) -> MetropolisHastingsConfig:
        std_proposal_density_youngs_modulus = 1000.0
        cov_proposal_density = torch.diag(
            torch.tensor(
                [
                    std_proposal_density_youngs_modulus,
                ],
                dtype=torch.float64,
                device=device,
            )
            ** 2
        )
        return MetropolisHastingsConfig(
            likelihood=likelihood,
            prior=prior,
            initial_parameters=initial_parameters,
            num_iterations=int(1e4),
            num_burn_in_iterations=int(1e3),
            cov_proposal_density=cov_proposal_density,
        )

    def set_up_hamiltonian_config(
        likelihood: Likelihood,
    ) -> HamiltonianConfig:
        return HamiltonianConfig(
            likelihood=likelihood,
            prior=prior,
            initial_parameters=initial_parameters,
            num_iterations=int(1e4),
            num_burn_in_iterations=int(1e3),
            num_leabfrog_steps=256,
            leapfrog_step_sizes=torch.tensor(10.0, device=device),
        )

    def set_up_efficient_nuts_configs_config(
        likelihood: Likelihood,
    ) -> EfficientNUTSConfig:
        return EfficientNUTSConfig(
            likelihood=likelihood,
            prior=prior,
            initial_parameters=initial_parameters,
            num_iterations=int(1e4),
            num_burn_in_iterations=int(1e3),
            max_tree_depth=8,
            leapfrog_step_sizes=torch.tensor(10.0, device=device),
        )

    if use_random_walk_metropolis_hasting:
        configs_mh = set_up_metropolis_hastings_config(likelihood)
        start = perf_counter()
        test_coverage(
            calibration_configs=(configs_mh,),
            parameter_names=parameter_names,
            true_parameters=true_parameter,
            output_subdir=os.path.join(
                output_subdir_calibration, "metropolis_hastings"
            ),
            project_directory=project_directory,
            device=device,
        )
        end = perf_counter()
        time = end - start
        print(f"Run time Metropolis-Hasting coverage test: {time}")
        print("############################################################")
    if use_hamiltonian:
        configs_h = set_up_hamiltonian_config(likelihood)
        start = perf_counter()
        test_coverage(
            calibration_configs=(configs_h,),
            parameter_names=parameter_names,
            true_parameters=true_parameter,
            output_subdir=os.path.join(output_subdir_calibration, "hamiltonian"),
            project_directory=project_directory,
            device=device,
        )
        end = perf_counter()
        time = end - start
        print(f"Run time Hamiltonian coverage test: {time}")
        print("############################################################")
    if use_efficient_nuts:
        configs_en = set_up_efficient_nuts_configs_config(likelihood)
        start = perf_counter()
        test_coverage(
            calibration_configs=(configs_en,),
            parameter_names=parameter_names,
            true_parameters=true_parameter,
            output_subdir=os.path.join(output_subdir_calibration, "efficient_nuts"),
            project_directory=project_directory,
            device=device,
        )
        end = perf_counter()
        time = end - start
        print(f"Run time efficient NUTS coverage test: {time}")
        print("############################################################")
    print("Calibration finished.")


training_dataset, validation_dataset = create_datasets()
if pretrain_parametric_pinn:
    pretraining_step()
    bayesian_training_step(is_pretrained=True)
else:
    bayesian_training_step(is_pretrained=False)
calibration_step()
