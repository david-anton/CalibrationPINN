import statistics
from dataclasses import dataclass
from typing import Optional, TypeAlias

import torch
from torch.utils.data import DataLoader

from parametricpinn.ansatz import StandardAnsatz
from parametricpinn.data.dataset import (
    SimulationData,
    TrainingData1DCollocation,
    TrainingData1DTractionBC,
)
from parametricpinn.data.simulationdata_linearelasticity_1d import (
    StretchedRodSimulationDatasetLinearElasticity1D,
)
from parametricpinn.data.trainingdata_1d import StretchedRodTrainingDataset1D
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.loaderssavers import PytorchModelSaver
from parametricpinn.postprocessing.plot import (
    HistoryPlotterConfig,
    plot_loss_history,
    plot_valid_history,
)
from parametricpinn.training.loss_1d.momentum_linearelasticity import (
    momentum_equation_func,
    traction_func,
)
from parametricpinn.training.metrics import mean_absolute_error, relative_l2_norm
from parametricpinn.types import Device, Tensor


@dataclass
class StandardTrainingConfiguration:
    ansatz: StandardAnsatz
    weight_pde_loss: float
    weight_traction_bc_loss: float
    weight_data_loss: float
    training_dataset_pinn: StretchedRodTrainingDataset1D
    number_training_epochs: int
    training_batch_size: int
    validation_dataset: StretchedRodSimulationDatasetLinearElasticity1D
    output_subdirectory: str
    project_directory: ProjectDirectory
    device: Device
    training_dataset_data: Optional[StretchedRodSimulationDatasetLinearElasticity1D] = (
        None
    )


def train_parametric_pinn(train_config: StandardTrainingConfiguration) -> None:
    ansatz = train_config.ansatz
    weight_pde_loss = train_config.weight_pde_loss
    weight_traction_bc_loss = train_config.weight_traction_bc_loss
    weight_data_loss = train_config.weight_data_loss
    train_dataset_pinn = train_config.training_dataset_pinn
    train_dataset_data = train_config.training_dataset_data
    is_simulation_data = bool(train_dataset_data)
    train_num_epochs = train_config.number_training_epochs
    train_batch_size = train_config.training_batch_size
    valid_dataset = train_config.validation_dataset
    valid_batch_size = len(valid_dataset)
    output_subdir = train_config.output_subdirectory
    project_directory = train_config.project_directory
    device = train_config.device

    loss_metric = torch.nn.MSELoss()

    lambda_pde_loss = torch.tensor(weight_pde_loss, requires_grad=True).to(device)
    lambda_traction_bc_loss = torch.tensor(
        weight_traction_bc_loss, requires_grad=True
    ).to(device)
    lambda_data_loss = torch.tensor(weight_data_loss, requires_grad=True).to(device)

    ### Loss function
    def loss_func(
        ansatz: StandardAnsatz,
        pde_data: TrainingData1DCollocation,
        traction_bc_data: TrainingData1DTractionBC,
        simulation_data: Optional[SimulationData] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:

        def loss_func_pde(
            ansatz: StandardAnsatz, pde_data: TrainingData1DCollocation
        ) -> Tensor:
            x_coor = pde_data.x_coor.to(device)
            x_params = pde_data.x_params.to(device)
            volume_force = pde_data.f.to(device)
            y = momentum_equation_func(ansatz, x_coor, x_params, volume_force)
            y_true = torch.zeros_like(y)
            return loss_metric(y_true, y)

        def loss_func_traction_bc(
            ansatz: StandardAnsatz, traction_bc_data: TrainingData1DTractionBC
        ) -> Tensor:
            x_coor = traction_bc_data.x_coor.to(device)
            x_params = traction_bc_data.x_params.to(device)
            y_true = traction_bc_data.y_true.to(device)
            y = traction_func(ansatz, x_coor, x_params)
            return loss_metric(y_true, y)

        def loss_func_data(
            ansatz: StandardAnsatz, simulation_data: SimulationData
        ) -> Tensor:
            x_coor = simulation_data.x_coor
            x_params = simulation_data.x_params
            x = torch.concat((x_coor, x_params), dim=1).to(device)
            y_true = simulation_data.y_true.to(device)
            y = ansatz(x)
            return loss_metric(y_true, y)

        loss_pde = lambda_pde_loss * loss_func_pde(ansatz, pde_data)
        loss_traction_bc = lambda_traction_bc_loss * loss_func_traction_bc(
            ansatz, traction_bc_data
        )
        if simulation_data is not None:
            loss_data = lambda_data_loss * loss_func_data(ansatz, simulation_data)
        else:
            loss_data = torch.tensor(0.0, device=device)
        return loss_pde, loss_traction_bc, loss_data

    ### Validation
    def validate_model(
        ansatz: StandardAnsatz, valid_dataloader: DataLoader
    ) -> tuple[float, float]:
        ansatz.eval()
        with torch.no_grad():
            valid_batches = iter(valid_dataloader)
            mae_hist_batches = []
            rl2_hist_batches = []

            for valid_batch in valid_batches:
                x_coor = valid_batch.x_coor
                x_params = valid_batch.x_params
                x = torch.concat((x_coor, x_params), dim=1).to(device)
                y_true = valid_batch.y_true.to(device)
                y = ansatz(x)
                mae_batch = mean_absolute_error(y_true, y)
                rl2_batch = relative_l2_norm(y_true, y)
                mae_hist_batches.append(mae_batch.cpu().item())
                rl2_hist_batches.append(rl2_batch.cpu().item())

            mean_mae = statistics.mean(mae_hist_batches)
            mean_rl2 = statistics.mean(rl2_hist_batches)
        return mean_mae, mean_rl2

    ### Training process
    train_dataloader_pinn = DataLoader(
        dataset=train_dataset_pinn,
        batch_size=train_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=train_dataset_pinn.get_collate_func(),
    )

    if train_dataset_data is not None:
        train_dataloader_data = DataLoader(
            dataset=train_dataset_data,
            batch_size=len(train_dataset_data),
            shuffle=False,
            drop_last=False,
            collate_fn=train_dataset_data.get_collate_func(),
        )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=valid_dataset.get_collate_func(),
    )

    optimizer = torch.optim.LBFGS(
        params=ansatz.parameters(),
        lr=1.0,
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    loss_hist_pde = []
    loss_hist_traction_bc = []
    loss_hist_data = []
    valid_hist_mae = []
    valid_hist_rl2 = []
    valid_epochs = []

    # Closure for LBFGS
    def loss_func_closure() -> float:
        optimizer.zero_grad()
        loss_pde, loss_traction_bc, loss_data = loss_func(
            ansatz, batch_pde, batch_traction_bc, batch_data
        )
        loss = loss_pde + loss_traction_bc + loss_data
        loss.backward()
        return loss.item()

    print("Start training ...")
    for epoch in range(train_num_epochs):
        train_batches_pinn = iter(train_dataloader_pinn)
        if is_simulation_data:
            train_batch_data = iter(train_dataloader_data)
            batch_data = next(train_batch_data)
        else:
            batch_data = None
        loss_hist_pde_batches = []
        loss_hist_traction_bc_batches = []
        loss_hist_data_batches = []

        for batch_pde, batch_traction_bc in train_batches_pinn:
            ansatz.train()

            # Forward pass
            loss_pde, loss_traction_bc, loss_data = loss_func(
                ansatz, batch_pde, batch_traction_bc, batch_data
            )

            # Update parameters
            optimizer.step(loss_func_closure)

            loss_hist_pde_batches.append(loss_pde.detach().cpu().item())
            loss_hist_traction_bc_batches.append(loss_traction_bc.detach().cpu().item())
            if is_simulation_data:
                loss_hist_data_batches.append(loss_data.detach().cpu().item())

        mean_loss_pde = statistics.mean(loss_hist_pde_batches)
        mean_loss_traction_bc = statistics.mean(loss_hist_traction_bc_batches)
        loss_hist_pde.append(mean_loss_pde)
        loss_hist_traction_bc.append(mean_loss_traction_bc)
        if is_simulation_data:
            mean_loss_data = statistics.mean(loss_hist_data_batches)
            loss_hist_data.append(mean_loss_data)

        print("##################################################")
        print(f"Epoch {epoch} / {train_num_epochs - 1}")
        print(f"PDE: \t\t {mean_loss_pde}")
        print(f"TRACTION_BC: \t {mean_loss_traction_bc}")
        if is_simulation_data:
            print(f"DATA: \t {mean_loss_data}")
        print("##################################################")
        if epoch % 1 == 0:
            mae, rl2 = validate_model(ansatz, valid_dataloader)
            valid_hist_mae.append(mae)
            valid_hist_rl2.append(rl2)
            valid_epochs.append(epoch)
            print(
                f"Validation: Epoch {epoch} / {train_num_epochs - 1}, MAE: {mae}, rL2: {rl2}"
            )

    ### Postprocessing
    print("Postprocessing ...")
    history_plotter_config = HistoryPlotterConfig()

    if is_simulation_data:
        loss_hists = [
            loss_hist_pde,
            loss_hist_traction_bc,
            loss_hist_data,
        ]
        loss_hist_names = ["PDE", "Traction BC", "Data"]
    else:
        loss_hists = [
            loss_hist_pde,
            loss_hist_traction_bc,
        ]
        loss_hist_names = ["PDE", "Traction BC"]

    plot_loss_history(
        loss_hists=loss_hists,
        loss_hist_names=loss_hist_names,
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

    ### Save model
    print("Save model ...")
    model_saver = PytorchModelSaver(project_directory=project_directory)
    model_saver.save(
        model=ansatz,
        file_name="model_parameters",
        subdir_name=output_subdir,
        device=device,
    )
