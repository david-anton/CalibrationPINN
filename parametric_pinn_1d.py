import statistics
from datetime import date

import numpy as np
import scipy.stats
import torch
from torch.utils.data import DataLoader

from parametricpinn.ansatz import create_normalized_hbc_ansatz_1D
from parametricpinn.calibration.likelihood import compile_likelihood
from parametricpinn.calibration.mcmc_metropolishastings import mcmc_metropolishastings
from parametricpinn.data import (
    TrainingData1DPDE,
    TrainingData1DStressBC,
    calculate_displacements_solution_1D,
    collate_training_data_1D,
    collate_validation_data_1D,
    create_training_dataset_1D,
    create_validation_dataset_1D,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.network import FFNN
from parametricpinn.postprocessing.plot import (
    DisplacementsPlotterConfig1D,
    HistoryPlotterConfig,
    plot_displacements_1D,
    plot_loss_history,
    plot_valid_history,
)
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.training.loss_1d import momentum_equation_func, traction_func
from parametricpinn.training.metrics import mean_absolute_error, relative_l2_norm
from parametricpinn.types import Module, Tensor

### Configuration
# Set up
length = 100.0
traction = 10.0
volume_force = 5.0
min_youngs_modulus = 180000.0
max_youngs_modulus = 240000.0
displacement_left = 0.0
# Network
layer_sizes = [2, 16, 16, 1]
# Training
num_samples_train = 128
num_points_pde = 128
batch_size_train = num_samples_train
num_epochs = 200
loss_metric = torch.nn.MSELoss()
# Validation
num_samples_valid = 128
valid_interval = 1
num_points_valid = 1024
batch_size_valid = num_samples_valid
# Output
current_date = date.today().strftime("%Y%m%d")
output_subdir = f"{current_date}_Parametric_PINN_1D"


settings = Settings()
project_directory = ProjectDirectory(settings)

# Set up simulation
set_default_dtype(torch.float64)
set_seed(0)
device = get_device()


### Loss function
def loss_func(
    ansatz: Module,
    pde_data: TrainingData1DPDE,
    stress_bc_data: TrainingData1DStressBC,
) -> tuple[Tensor, Tensor]:
    def loss_func_pde(ansatz: Module, pde_data: TrainingData1DPDE) -> Tensor:
        x_coor = pde_data.x_coor.to(device)
        x_E = pde_data.x_E.to(device)
        volume_force = pde_data.f.to(device)
        y_true = pde_data.y_true.to(device)
        y = momentum_equation_func(ansatz, x_coor, x_E, volume_force)
        return loss_metric(y_true, y)

    def loss_func_stress_bc(
        ansatz: Module, stress_bc_data: TrainingData1DStressBC
    ) -> Tensor:
        x_coor = stress_bc_data.x_coor.to(device)
        x_E = stress_bc_data.x_E.to(device)
        y_true = stress_bc_data.y_true.to(device)
        y = traction_func(ansatz, x_coor, x_E)
        return loss_metric(y_true, y)

    loss_pde = loss_func_pde(ansatz, pde_data)
    loss_stress_bc = loss_func_stress_bc(ansatz, stress_bc_data)
    return loss_pde, loss_stress_bc


### Validation
def validate_model(ansatz: Module, valid_dataloader: DataLoader) -> tuple[float, float]:
    ansatz.eval()
    with torch.no_grad():
        valid_batches = iter(valid_dataloader)
        mae_hist_batches = []
        rl2_hist_batches = []

        for x, y_true in valid_batches:
            x = x.to(device)
            y_true = y_true.to(device)
            y = ansatz(x)
            mae_batch = mean_absolute_error(y_true, y)
            rl2_batch = relative_l2_norm(y_true, y)
            mae_hist_batches.append(mae_batch.cpu().item())
            rl2_hist_batches.append(rl2_batch.cpu().item())

        mean_mae = statistics.mean(mae_hist_batches)
        mean_rl2 = statistics.mean(rl2_hist_batches)
    return mean_mae, mean_rl2


####################################################################################################
if __name__ == "__main__":
    print("Create training data ...")
    train_dataset = create_training_dataset_1D(
        length=length,
        traction=traction,
        volume_force=volume_force,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        num_points_pde=num_points_pde,
        num_samples=num_samples_train,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_train,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_training_data_1D,
    )

    print("Generate validation data ...")
    valid_dataset = create_validation_dataset_1D(
        length=length,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        traction=traction,
        volume_force=volume_force,
        num_points=num_points_valid,
        num_samples=num_samples_valid,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size_valid,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_validation_data_1D,
    )

    min_coordinate = 0.0
    max_coordinate = length
    min_displacement = displacement_left
    max_displacement = calculate_displacements_solution_1D(
        coordinates=max_coordinate,
        length=length,
        youngs_modulus=min_youngs_modulus,
        traction=traction,
        volume_force=volume_force,
    )
    min_inputs = torch.tensor([min_coordinate, min_youngs_modulus]).to(device)
    max_inputs = torch.tensor([max_coordinate, max_youngs_modulus]).to(device)
    min_output = torch.tensor([min_displacement]).to(device)
    max_output = torch.tensor([max_displacement]).to(device)

    network = FFNN(layer_sizes=layer_sizes)
    ansatz = create_normalized_hbc_ansatz_1D(
        displacement_left=torch.tensor([displacement_left]).to(device),
        network=network,
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        min_outputs=min_output,
        max_outputs=max_output,
    ).to(device)

    optimizer = torch.optim.LBFGS(
        params=ansatz.parameters(),
        lr=1.0,
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    loss_hist_pde = []
    loss_hist_stress_bc = []
    valid_hist_mae = []
    valid_hist_rl2 = []
    valid_epochs = []

    # Closure for LBFGS
    def loss_func_closure() -> float:
        optimizer.zero_grad()
        loss_pde, loss_stress_bc = loss_func(ansatz, batch_pde, batch_stress_bc)
        loss = loss_pde + loss_stress_bc
        loss.backward()
        return loss.item()

    print("Start training ...")
    for epoch in range(num_epochs):
        train_batches = iter(train_dataloader)
        loss_hist_pde_batches = []
        loss_hist_stress_bc_batches = []

        for batch_pde, batch_stress_bc in train_batches:
            ansatz.train()

            # Forward pass
            loss_pde, loss_stress_bc = loss_func(ansatz, batch_pde, batch_stress_bc)

            # Update parameters
            optimizer.step(loss_func_closure)

            loss_hist_pde_batches.append(loss_pde.detach().cpu().item())
            loss_hist_stress_bc_batches.append(loss_stress_bc.detach().cpu().item())

        mean_loss_pde = statistics.mean(loss_hist_pde_batches)
        mean_loss_stress_bc = statistics.mean(loss_hist_stress_bc_batches)
        loss_hist_pde.append(mean_loss_pde)
        loss_hist_stress_bc.append(mean_loss_stress_bc)

        if epoch % 1 == 0:
            print(f"Validation: Epoch {epoch} / {num_epochs}")
            mae, rl2 = validate_model(ansatz, valid_dataloader)
            valid_hist_mae.append(mae)
            valid_hist_rl2.append(rl2)
            valid_epochs.append(epoch)

    ### Postprocessing
    print("Postprocessing ...")
    history_plotter_config = HistoryPlotterConfig()

    plot_loss_history(
        loss_hists=[loss_hist_pde, loss_hist_stress_bc],
        loss_hist_names=["PDE", "Stress BC"],
        file_name="loss_pinn.png",
        output_subdir=output_subdir,
        project_directory=project_directory,
        config=history_plotter_config,
    )

    plot_valid_history(
        valid_epochs=valid_epochs,
        valid_hist=valid_hist_mae,
        valid_metric="mean absolute error",
        file_name="mae_pinn.png",
        output_subdir=output_subdir,
        project_directory=project_directory,
        config=history_plotter_config,
    )

    plot_valid_history(
        valid_epochs=valid_epochs,
        valid_hist=valid_hist_rl2,
        valid_metric="rel. L2 norm",
        file_name="rl2_pinn.png",
        output_subdir=output_subdir,
        project_directory=project_directory,
        config=history_plotter_config,
    )

    displacements_plotter_config = DisplacementsPlotterConfig1D()

    plot_displacements_1D(
        ansatz=ansatz,
        length=length,
        youngs_modulus=187634,
        traction=traction,
        volume_force=volume_force,
        output_subdir=output_subdir,
        project_directory=project_directory,
        config=displacements_plotter_config,
    )

    plot_displacements_1D(
        ansatz=ansatz,
        length=length,
        youngs_modulus=238356,
        traction=traction,
        volume_force=volume_force,
        output_subdir=output_subdir,
        project_directory=project_directory,
        config=displacements_plotter_config,
    )

    ## Calibration
    exact_youngs_modulus = 200000
    std_noise = 5 * 1e-4
    coordinates = np.linspace(start=0.0, stop=length, num=128, endpoint=True).reshape(
        (-1, 1)
    )
    clean_data = calculate_displacements_solution_1D(
        coordinates=coordinates,
        length=length,
        youngs_modulus=exact_youngs_modulus,
        traction=traction,
        volume_force=volume_force,
    )
    noisy_data = clean_data + np.random.normal(
        loc=0.0, scale=std_noise, size=clean_data.shape
    )

    mean_youngs_modulus = 210000
    std_youngs_modulus = 15000
    prior = scipy.stats.multivariate_normal(
        mean=np.array([mean_youngs_modulus]), cov=np.array([std_youngs_modulus])
    )
    covariance_error = np.array([std_noise])
    likelihood = compile_likelihood(
        model=ansatz,
        coordinates=coordinates,
        data=noisy_data,
        covariance_error=covariance_error,
        device=device,
    )
    std_proposal_density = 1000
    posterior_moments, samples = mcmc_metropolishastings(
        parameter_names=("Youngs modulus",),
        likelihood=likelihood,
        prior=prior,
        initial_parameters=np.array([mean_youngs_modulus]),
        std_proposal_density=np.array([std_proposal_density]),
        num_iterations=int(1e6),
        output_subdir=output_subdir,
        project_directory=project_directory,
    )
    print(posterior_moments)
