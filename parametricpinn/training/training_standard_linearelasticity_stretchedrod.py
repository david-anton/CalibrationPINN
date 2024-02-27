import statistics
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from parametricpinn.ansatz import StandardAnsatz
from parametricpinn.data.dataset import (
    TrainingData1DCollocation,
    TrainingData1DTractionBC,
)
from parametricpinn.data.trainingdata_1d import StretchedRodTrainingDataset1D
from parametricpinn.data.validationdata_linearelasticity_1d import (
    StretchedRodValidationDatasetLinearElasticity1D,
)
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
    training_dataset: StretchedRodTrainingDataset1D
    number_training_epochs: int
    training_batch_size: int
    validation_dataset: StretchedRodValidationDatasetLinearElasticity1D
    output_subdirectory: str
    project_directory: ProjectDirectory
    device: Device


def train_parametric_pinn(train_config: StandardTrainingConfiguration) -> None:
    ansatz = train_config.ansatz
    weight_pde_loss = train_config.weight_pde_loss
    weight_traction_bc_loss = train_config.weight_traction_bc_loss
    train_dataset = train_config.training_dataset
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

    ### Loss function
    def loss_func(
        ansatz: StandardAnsatz,
        pde_data: TrainingData1DCollocation,
        traction_bc_data: TrainingData1DTractionBC,
    ) -> tuple[Tensor, Tensor]:
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

        loss_pde = lambda_pde_loss * loss_func_pde(ansatz, pde_data)
        loss_traction_bc = lambda_traction_bc_loss * loss_func_traction_bc(
            ansatz, traction_bc_data
        )
        return loss_pde, loss_traction_bc

    ### Validation
    def validate_model(
        ansatz: StandardAnsatz, valid_dataloader: DataLoader
    ) -> tuple[float, float]:
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

    ### Training process
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=train_dataset.get_collate_func(),
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
    valid_hist_mae = []
    valid_hist_rl2 = []
    valid_epochs = []

    # Closure for LBFGS
    def loss_func_closure() -> float:
        optimizer.zero_grad()
        loss_pde, loss_traction_bc = loss_func(ansatz, batch_pde, batch_traction_bc)
        loss = loss_pde + loss_traction_bc
        loss.backward()
        return loss.item()

    print("Start training ...")
    for epoch in range(train_num_epochs):
        train_batches = iter(train_dataloader)
        loss_hist_pde_batches = []
        loss_hist_traction_bc_batches = []

        for batch_pde, batch_traction_bc in train_batches:
            ansatz.train()

            # Forward pass
            loss_pde, loss_traction_bc = loss_func(ansatz, batch_pde, batch_traction_bc)

            # Update parameters
            optimizer.step(loss_func_closure)

            loss_hist_pde_batches.append(loss_pde.detach().cpu().item())
            loss_hist_traction_bc_batches.append(loss_traction_bc.detach().cpu().item())

        mean_loss_pde = statistics.mean(loss_hist_pde_batches)
        mean_loss_traction_bc = statistics.mean(loss_hist_traction_bc_batches)
        loss_hist_pde.append(mean_loss_pde)
        loss_hist_traction_bc.append(mean_loss_traction_bc)

        if epoch % 1 == 0:
            mae, rl2 = validate_model(ansatz, valid_dataloader)
            valid_hist_mae.append(mae)
            valid_hist_rl2.append(rl2)
            valid_epochs.append(epoch)

    ### Postprocessing
    print("Postprocessing ...")
    history_plotter_config = HistoryPlotterConfig()

    plot_loss_history(
        loss_hists=[loss_hist_pde, loss_hist_traction_bc],
        loss_hist_names=["PDE", "Traction BC"],
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
