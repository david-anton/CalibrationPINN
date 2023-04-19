# Standard library imports
import statistics

# Third-party imports
import torch
from torch.utils.data import DataLoader

# Local library imports
from parametricpinn.ansatz import HBCAnsatz1D
from parametricpinn.calibration import calibrate_model
from parametricpinn.data import (TrainingDataset1D,
                                 calculate_displacements_solution_1D,
                                 collate_training_data_1D,
                                 collate_validation_data_1D,
                                 create_training_dataset_1D,
                                 create_validation_dataset_1D)
from parametricpinn.io import ProjectDirectory
from parametricpinn.network import FFNN, create_normalized_network
from parametricpinn.postprocessing.plot import (PlotterConfig1D,
                                                plot_displacements_1D,
                                                plot_loss_hist_1D,
                                                plot_valid_hist_1D)
from parametricpinn.settings import (Settings, get_device, set_default_dtype,
                                     set_seed)
from parametricpinn.training.loss import (momentum_equation_func_1D,
                                          stress_func_1D)
from parametricpinn.training.metrics import (mean_absolute_error,
                                             relative_l2_norm)
from parametricpinn.types import Module, Tensor

### Configuration
# Set up
length = 100.0
min_youngs_modulus = 180000.0
max_youngs_modulus = 240000.0
traction = 10.0
volume_force = 5.0
displacement_left = 0.0
# Network
layer_sizes = [2, 16, 16, 1]
# Training
num_samples_train = 100
num_points_pde = 100
batch_size_train = num_samples_train
num_epochs = 100
loss_metric = torch.nn.MSELoss()
# Validation
num_samples_valid = 100
valid_interval = 1
num_points_valid = 1000
batch_size_valid = num_points_valid
# Output
output_subdir = "parametric_PINN_1D"


settings = Settings()
project_directory = ProjectDirectory(settings)

# Set up simulation
set_default_dtype(torch.float64)
set_seed(0)
device = get_device()


### Loss function
def loss_func(
    ansatz: Module,
    pde_data: TrainingDataset1D,
    stress_bc_data: TrainingDataset1D,
    volume_force: Tensor,
) -> tuple[Tensor, Tensor]:
    def loss_func_pde(ansatz, pde_data):
        x_coor = pde_data.x_coor.to(device)
        x_E = pde_data.x_E.to(device)
        y_true = pde_data.y_true.to(device)
        y = momentum_equation_func_1D(ansatz, x_coor, x_E, volume_force)
        return loss_metric(y_true, y)

    def loss_func_stress_bc(ansatz, stress_bc_data):
        x_coor = stress_bc_data.x_coor.to(device)
        x_E = stress_bc_data.x_E.to(device)
        y_true = stress_bc_data.y_true.to(device)
        y = stress_func_1D(ansatz, x_coor, x_E)
        return loss_metric(y_true, y)

    loss_pde = loss_func_pde(ansatz, pde_data)
    loss_stress_bc = loss_func_stress_bc(ansatz, stress_bc_data)
    return loss_pde, loss_stress_bc


### Validation
def validate_model(ansatz: Module, valid_dataloader: DataLoader) -> tuple[float, float]:
    with torch.no_grad():
        ansatz.eval()
        valid_batches = iter(valid_dataloader)
        mae_hist_batches = []
        rl2_hist_batches = []

        for x, y_true in valid_batches:
            x = x.to(device)
            y_true = y_true.to(device)
            y = ansatz(x)
            mae_batch = mean_absolute_error(y_true, y)
            rl2_batch = relative_l2_norm(y_true, y)
            mae_hist_batches.append(mae_batch.item())
            rl2_hist_batches.append(rl2_batch.item())

        mean_mae = statistics.mean(mae_hist_batches)
        mean_rl2 = statistics.mean(rl2_hist_batches)
    return mean_mae, mean_rl2


####################################################################################################
if __name__ == "__main__":
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
    min_inputs = torch.tensor([min_coordinate, min_youngs_modulus])
    max_inputs = torch.tensor([max_coordinate, max_youngs_modulus])
    min_output = torch.tensor([min_displacement])
    max_output = torch.tensor([max_displacement])
    input_range_coordinate = max_coordinate - min_coordinate

    network = FFNN(layer_sizes=layer_sizes)
    normalized_network = create_normalized_network(
        network=network,
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        min_outputs=min_output,
        max_outputs=max_output,
    ).to(device)
    ansatz = HBCAnsatz1D(
        displacement_left=displacement_left,
        input_range_coordinate=input_range_coordinate,
        network=normalized_network,
    )

    train_dataset = create_training_dataset_1D(
        length=length,
        traction=traction,
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
        loss_pde, loss_stress_bc = loss_func(
            ansatz, batch_pde, batch_stress_bc, torch.tensor(volume_force)
        )
        loss = loss_pde + loss_stress_bc
        loss.backward()
        return loss.item()

    for epoch in range(num_epochs):
        train_batches = iter(train_dataloader)
        loss_hist_pde_batches = []
        loss_hist_stress_bc_batches = []

        for batch_pde, batch_stress_bc in train_batches:
            # Forward pass
            loss_pde, loss_stress_bc = loss_func(
                ansatz, batch_pde, batch_stress_bc, torch.tensor(volume_force)
            )
            loss = loss_pde + loss_stress_bc

            # Update parameters
            optimizer.step(loss_func_closure)

            loss_hist_pde_batches.append(loss_pde.cpu().item())
            loss_hist_stress_bc_batches.append(loss_stress_bc.cpu().item())

        mean_batch_loss_pde = statistics.mean(loss_hist_pde_batches)
        mean_batch_loss_stress_bc = statistics.mean(loss_hist_stress_bc_batches)
        loss_hist_pde.append(mean_batch_loss_pde)
        loss_hist_stress_bc.append(mean_batch_loss_stress_bc)

        if epoch % 1 == 0:
            mae, rl2 = validate_model(ansatz, valid_dataloader)
            valid_hist_mae.append(mae)
            valid_hist_rl2.append(rl2)
            valid_epochs.append(epoch)

    ## Preprocessing
    plotter_config = PlotterConfig1D()

    plot_loss_hist_1D(
        loss_hist_pde=loss_hist_pde,
        loss_hist_stress_bc=loss_hist_stress_bc,
        file_name="loss_pinn.png",
        output_subdir=output_subdir,
        project_directory=project_directory,
        config=plotter_config,
    )

    plot_valid_hist_1D(
        valid_epochs=valid_epochs,
        valid_hist=valid_hist_mae,
        valid_metric="mean absolute error",
        file_name="mae_pinn.png",
        output_subdir=output_subdir,
        project_directory=project_directory,
        config=plotter_config,
    )

    plot_valid_hist_1D(
        valid_epochs=valid_epochs,
        valid_hist=valid_hist_rl2,
        valid_metric="rel. L2 norm",
        file_name="rl2_pinn.png",
        output_subdir=output_subdir,
        project_directory=project_directory,
        config=plotter_config,
    )

    plot_displacements_1D(
        ansatz=ansatz,
        length=length,
        youngs_modulus=187634,
        traction=traction,
        volume_force=volume_force,
        file_name="displacements_pinn_E_187634.png",
        output_subdir=output_subdir,
        project_directory=project_directory,
        config=plotter_config,
    )

    plot_displacements_1D(
        ansatz=ansatz,
        length=length,
        youngs_modulus=238356,
        traction=traction,
        volume_force=volume_force,
        file_name="displacements_pinn_E_238356.png",
        output_subdir=output_subdir,
        project_directory=project_directory,
        config=plotter_config,
    )

    # ## Calibration
    # num_data_points = 100
    # E_true = 200000.0
    # coordinates = torch.linspace(
    #     0.0, length, num_data_points, requires_grad=False
    # ).view([num_data_points, 1])
    # data = calculate_displacements_solution_1D(
    #     coordinates, length, E_true, traction, volume_force
    # )
    # coordinates = coordinates.to(device)
    # data = data.to(device)

    # E_estimated, loss_hist_cal = calibrate_model(ansatz, coordinates, data)

    # rel_error_E = (E_estimated - E_true) / E_true

    # print("Calibration results:")
    # print(f"True E: {E_true}")
    # print(f"Estimated E: {E_estimated}")
    # print(f"Relative error E: {rel_error_E}")
