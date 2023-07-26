import math
import os
import statistics
from datetime import date

import numpy as np
import torch
from torch.utils.data import DataLoader

from parametricpinn.ansatz import create_normalized_hbc_ansatz_2D
from parametricpinn.data import (
    TrainingData2DCollocation,
    TrainingData2DSymmetryBC,
    TrainingData2DTractionBC,
    collate_training_data_2D,
    collate_validation_data_2D,
    create_training_dataset_2D,
    create_validation_dataset_2D,
)
from parametricpinn.fem.platewithhole import generate_validation_data, run_simulation
from parametricpinn.io import ProjectDirectory
from parametricpinn.network import FFNN
from parametricpinn.postprocessing.plot import (
    DisplacementsPlotterConfigPWH,
    HistoryPlotterConfig,
    plot_displacements_pwh,
    plot_loss_history,
    plot_valid_history,
)
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.training.loss_2d import (
    momentum_equation_func_factory,
    strain_energy_func_factory,
    stress_func_factory,
    traction_energy_func_factory,
    traction_func_factory,
)
from parametricpinn.training.metrics import mean_absolute_error, relative_l2_norm
from parametricpinn.types import Module, Tensor

### Configuration
# Set up
model = "plane stress"
edge_length = 100.0
radius = 10.0
traction_left_x = -100.0
traction_left_y = 0.0
volume_force_x = 0.0
volume_force_y = 0.0
min_youngs_modulus = 210000.0
max_youngs_modulus = 210000.0
min_poissons_ratio = 0.3
max_poissons_ratio = 0.3
# Network
layer_sizes = [4, 16, 16, 16, 16, 2]
# Training
num_samples_per_parameter = 1
num_collocation_points = 8192
num_points_per_bc = 1024
batch_size_train = 1
num_epochs = 6000
loss_metric = torch.nn.MSELoss(reduction="mean")
weight_pde_loss = 1.0
weight_symmetry_bc_loss = 1.0
weight_traction_bc_loss = 1.0
weight_energy_loss = 1.0
# Validation
regenerate_valid_data = True
input_subdir_valid = "20230616_validation_data_E_210000_nu_03_radius_10"
num_samples_valid = 1
valid_interval = 1
num_points_valid = 1024
batch_size_valid = num_samples_valid
fem_mesh_resolution = 0.1
# Output
current_date = date.today().strftime("%Y%m%d")
output_subdir = f"{current_date}_pinn_E_210000_nu_03_test_refactoring"
output_subdir_preprocessing = f"{current_date}_preprocessing"
save_metadata = True


settings = Settings()
project_directory = ProjectDirectory(settings)

# Set up simulation
set_default_dtype(torch.float64)
set_seed(0)
device = get_device()


### Loss function
momentum_equation_func = momentum_equation_func_factory(model)
stress_func = stress_func_factory(model)
traction_func = traction_func_factory(model)
strain_energy_func = strain_energy_func_factory(model)
traction_energy_func = traction_energy_func_factory(model)
traction_left = torch.tensor([traction_left_x, traction_left_y])
volume_force = torch.tensor([volume_force_x, volume_force_y])

lambda_pde_loss = torch.tensor(weight_pde_loss, requires_grad=True).to(device)
lambda_symmetry_bc_loss = torch.tensor(weight_symmetry_bc_loss, requires_grad=True).to(
    device
)
lambda_traction_bc_loss = torch.tensor(weight_traction_bc_loss, requires_grad=True).to(
    device
)
lambda_energy_loss = torch.tensor(weight_energy_loss, requires_grad=True).to(device)
area_pwh = torch.tensor(edge_length**2 - 1 / 4 * math.pi * radius**2).to(device)


def loss_func(
    ansatz: Module,
    collocation_data: TrainingData2DCollocation,
    symmetry_bc_data: TrainingData2DSymmetryBC,
    traction_bc_data: TrainingData2DTractionBC,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    def loss_func_pde(
        ansatz: Module, collocation_data: TrainingData2DCollocation
    ) -> Tensor:
        x_coor = collocation_data.x_coor.to(device)
        x_E = collocation_data.x_E
        x_nu = collocation_data.x_nu
        x_param = torch.concat((x_E, x_nu), dim=1).to(device)
        volume_force = collocation_data.f.to(device)
        y_true = torch.zeros_like(x_coor).to(device)
        y = momentum_equation_func(ansatz, x_coor, x_param, volume_force)
        return loss_metric(y_true, y)

    def loss_func_symmetry_bc(
        ansatz: Module, symmetry_bc_data: TrainingData2DSymmetryBC
    ) -> Tensor:
        x_coor = symmetry_bc_data.x_coor.to(device)
        x_E = symmetry_bc_data.x_E
        x_nu = symmetry_bc_data.x_nu
        x_param = torch.concat((x_E, x_nu), dim=1).to(device)
        shear_stress_filter = (
            torch.tensor([[0.0, 1.0], [1.0, 0.0]])
            .repeat(2 * num_points_per_bc, 1, 1)
            .to(device)
        )
        stress_tensors = stress_func(ansatz, x_coor, x_param)
        y = shear_stress_filter * stress_tensors
        y_true = (
            torch.tensor([[0.0, 0.0], [0.0, 0.0]])
            .repeat(2 * num_points_per_bc, 1, 1)
            .to(device)
        )
        return loss_metric(y_true, y)

    def loss_func_traction_bc(
        ansatz: Module, traction_bc_data: TrainingData2DTractionBC
    ) -> Tensor:
        x_coor = traction_bc_data.x_coor.to(device)
        x_E = traction_bc_data.x_E
        x_nu = traction_bc_data.x_nu
        x_param = torch.concat((x_E, x_nu), dim=1).to(device)
        normal = traction_bc_data.normal.to(device)
        y_true = traction_bc_data.y_true.to(device)
        y = traction_func(ansatz, x_coor, x_param, normal)
        return loss_metric(y_true, y)

    def loss_func_energy(
        ansatz: Module,
        collocation_data: TrainingData2DCollocation,
        traction_bc_data: TrainingData2DTractionBC,
    ) -> Tensor:
        x_coor_int = collocation_data.x_coor.to(device)
        x_E_int = collocation_data.x_E
        x_nu_int = collocation_data.x_nu
        x_param_int = torch.concat((x_E_int, x_nu_int), dim=1).to(device)
        strain_energy = strain_energy_func(ansatz, x_coor_int, x_param_int, area_pwh)

        x_coor_ext = traction_bc_data.x_coor.to(device)
        x_E_ext = traction_bc_data.x_E
        x_nu_ext = traction_bc_data.x_nu
        x_param_ext = torch.concat((x_E_ext, x_nu_ext), dim=1).to(device)
        normal_ext = traction_bc_data.normal.to(device)
        area_frac_ext = traction_bc_data.area_frac.to(device)
        traction_energy = traction_energy_func(
            ansatz, x_coor_ext, x_param_ext, normal_ext, area_frac_ext
        )
        y = strain_energy - traction_energy
        y_true = torch.tensor(0.0).to(device)
        return loss_metric(y_true, y)

    loss_pde = lambda_pde_loss * loss_func_pde(ansatz, collocation_data)
    loss_symmetry_bc = lambda_symmetry_bc_loss * loss_func_symmetry_bc(
        ansatz, symmetry_bc_data
    )
    loss_traction_bc = lambda_traction_bc_loss * loss_func_traction_bc(
        ansatz, traction_bc_data
    )
    loss_energy = lambda_energy_loss * loss_func_energy(
        ansatz, collocation_data, traction_bc_data
    )
    return loss_pde, loss_symmetry_bc, loss_traction_bc, loss_energy


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


### Preprocessing
def _generate_validation_data() -> None:
    def _generate_random_parameter_list(
        size: int, min_value: float, max_value: float
    ) -> list[float]:
        random_params = min_value + torch.rand(size) * (max_value - min_value)
        return random_params.tolist()

    youngs_moduli = _generate_random_parameter_list(
        num_samples_valid, min_youngs_modulus, max_youngs_modulus
    )
    poissons_ratios = _generate_random_parameter_list(
        num_samples_valid, min_poissons_ratio, max_poissons_ratio
    )
    generate_validation_data(
        model=model,
        youngs_moduli=youngs_moduli,
        poissons_ratios=poissons_ratios,
        edge_length=edge_length,
        radius=radius,
        volume_force_x=volume_force_x,
        volume_force_y=volume_force_y,
        traction_left_x=traction_left_x,
        traction_left_y=traction_left_y,
        save_metadata=save_metadata,
        output_subdir=input_subdir_valid,
        project_directory=project_directory,
        mesh_resolution=fem_mesh_resolution,
    )


def determine_normalization_values() -> dict[str, Tensor]:
    min_coordinate_x = -edge_length
    max_coordinate_x = 0.0
    min_coordinate_y = 0.0
    max_coordinate_y = edge_length
    min_coordinates = torch.tensor([min_coordinate_x, min_coordinate_y])
    max_coordinates = torch.tensor([max_coordinate_x, max_coordinate_y])

    min_parameters = torch.tensor([min_youngs_modulus, min_poissons_ratio])
    max_parameters = torch.tensor([max_youngs_modulus, max_poissons_ratio])

    min_inputs = torch.concat((min_coordinates, min_parameters))
    max_inputs = torch.concat((max_coordinates, max_parameters))

    _output_subdir = os.path.join(
        output_subdir_preprocessing, "results_for_determining_normalization_values"
    )
    simulation_results = run_simulation(
        model=model,
        youngs_modulus=min_youngs_modulus,
        poissons_ratio=max_poissons_ratio,
        edge_length=edge_length,
        radius=radius,
        volume_force_x=volume_force_x,
        volume_force_y=volume_force_y,
        traction_left_x=traction_left_x,
        traction_left_y=traction_left_y,
        save_results=False,
        save_metadata=False,
        output_subdir=_output_subdir,
        project_directory=project_directory,
        mesh_resolution=fem_mesh_resolution,
    )
    min_displacement_x = float(np.amin(simulation_results.displacements_x))
    max_displacement_x = float(np.amax(simulation_results.displacements_x))
    min_displacement_y = float(np.amin(simulation_results.displacements_y))
    max_displacement_y = float(np.amax(simulation_results.displacements_y))
    min_outputs = torch.tensor([min_displacement_x, min_displacement_y])
    max_outputs = torch.tensor([max_displacement_x, max_displacement_y])
    return {
        "min_inputs": min_inputs.to(device),
        "max_inputs": max_inputs.to(device),
        "min_outputs": min_outputs.to(device),
        "max_outputs": max_outputs.to(device),
    }


####################################################################################################
if __name__ == "__main__":
    print("Generate training data ...")
    train_dataset = create_training_dataset_2D(
        edge_length=edge_length,
        radius=radius,
        traction_left=traction_left,
        volume_force=volume_force,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        min_poissons_ratio=min_poissons_ratio,
        max_poissons_ratio=max_poissons_ratio,
        num_collocation_points=num_collocation_points,
        num_points_per_bc=num_points_per_bc,
        num_samples_per_parameter=num_samples_per_parameter,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_training_data_2D,
    )

    print("Load validation data ...")
    if regenerate_valid_data:
        print("Run FE simulations to generate validation data ...")
        _generate_validation_data()

    valid_dataset = create_validation_dataset_2D(
        input_subdir=input_subdir_valid,
        num_points=num_points_valid,
        num_samples=num_samples_valid,
        project_directory=project_directory,
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size_valid,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_validation_data_2D,
    )

    print("Run FE simulation to determine normalization values ...")
    normalization_values = determine_normalization_values()

    network = FFNN(layer_sizes=layer_sizes)
    ansatz = create_normalized_hbc_ansatz_2D(
        displacement_x_right=torch.tensor(0.0).to(device),
        displacement_y_bottom=torch.tensor(0.0).to(device),
        network=network,
        min_inputs=normalization_values["min_inputs"],
        max_inputs=normalization_values["max_inputs"],
        min_outputs=normalization_values["min_outputs"],
        max_outputs=normalization_values["max_outputs"],
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
    loss_hist_symmetry_bc = []
    loss_hist_traction_bc = []
    loss_hist_energy = []
    valid_hist_mae = []
    valid_hist_rl2 = []
    valid_epochs = []

    # Closure for LBFGS
    def loss_func_closure() -> float:
        optimizer.zero_grad()
        loss_collocation, loss_symmetry_bc, loss_traction_bc, loss_energy = loss_func(
            ansatz, batch_collocation, batch_symmetry_bc, batch_traction_bc
        )
        loss = loss_collocation + loss_symmetry_bc + loss_traction_bc + loss_energy
        loss.backward(retain_graph=True)
        return loss.item()

    print("Start training ...")
    for epoch in range(num_epochs):
        train_batches = iter(train_dataloader)
        loss_hist_pde_batches = []
        loss_hist_symmetry_bc_batches = []
        loss_hist_traction_bc_batches = []
        loss_hist_energy_batches = []

        for batch_collocation, batch_symmetry_bc, batch_traction_bc in train_batches:
            ansatz.train()

            # Forward pass
            loss_pde, loss_symmetry_bc, loss_traction_bc, loss_energy = loss_func(
                ansatz, batch_collocation, batch_symmetry_bc, batch_traction_bc
            )

            # Update parameters
            optimizer.step(loss_func_closure)

            loss_hist_pde_batches.append(loss_pde.detach().cpu().item())
            loss_hist_symmetry_bc_batches.append(loss_symmetry_bc.detach().cpu().item())
            loss_hist_traction_bc_batches.append(loss_traction_bc.detach().cpu().item())
            loss_hist_energy_batches.append(loss_energy.detach().cpu().item())

        mean_loss_pde = statistics.mean(loss_hist_pde_batches)
        mean_loss_symmetry_bc = statistics.mean(loss_hist_symmetry_bc_batches)
        mean_loss_traction_bc = statistics.mean(loss_hist_traction_bc_batches)
        mean_loss_energy = statistics.mean(loss_hist_energy_batches)
        loss_hist_pde.append(mean_loss_pde)
        loss_hist_symmetry_bc.append(mean_loss_symmetry_bc)
        loss_hist_traction_bc.append(mean_loss_traction_bc)
        loss_hist_energy.append(mean_loss_energy)

        print("##################################################")
        print(f"Epoch {epoch} / {num_epochs}")
        print(f"PDE: \t\t {mean_loss_pde}")
        print(f"SYMMETRY_BC: \t {mean_loss_symmetry_bc}")
        print(f"TRACTION_BC: \t {mean_loss_traction_bc}")
        print(f"ENERGY: \t {mean_loss_energy}")
        print("##################################################")
        if epoch % valid_interval == 0 or epoch == num_epochs:
            mae, rl2 = validate_model(ansatz, valid_dataloader)
            valid_hist_mae.append(mae)
            valid_hist_rl2.append(rl2)
            valid_epochs.append(epoch)
            print(f"Validation: Epoch {epoch} / {num_epochs}, MAE: {mae}, rL2: {rl2}")

    ### Postprocessing
    print("Postprocessing ...")
    history_plotter_config = HistoryPlotterConfig()

    plot_loss_history(
        loss_hists=[
            loss_hist_pde,
            loss_hist_symmetry_bc,
            loss_hist_traction_bc,
            loss_hist_energy,
        ],
        loss_hist_names=["PDE", "Symmetry BC", "Traction BC", "Energy"],
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

    displacements_plotter_config = DisplacementsPlotterConfigPWH()

    plot_displacements_pwh(
        ansatz=ansatz,
        youngs_modulus_and_poissons_ratio=[(210000, 0.3)],
        model=model,
        edge_length=edge_length,
        radius=radius,
        volume_force_x=volume_force_x,
        volume_force_y=volume_force_y,
        traction_left_x=traction_left_x,
        traction_left_y=traction_left_y,
        output_subdir=output_subdir,
        project_directory=project_directory,
        plot_config=displacements_plotter_config,
        device=device,
        mesh_resolution=0.5,
    )
