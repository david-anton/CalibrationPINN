import statistics
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from parametricpinn.ansatz import StandardAnsatz
from parametricpinn.data.dataset import (
    TrainingData2DCollocation,
    TrainingData2DTractionBC,
)
from parametricpinn.data.trainingdata_elasticity_2d import (
    SimplifiedDogBoneTrainingDataset2D,
)
from parametricpinn.data.validationdata_elasticity_2d import ValidationDataset2D
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.loaderssavers import PytorchModelSaver
from parametricpinn.postprocessing.plot import (
    HistoryPlotterConfig,
    plot_loss_history,
    plot_valid_history,
)
from parametricpinn.training.loss_2d.momentum_linearelasticity import (
    momentum_equation_func_factory,
    strain_energy_func_factory,
    stress_func_factory,
    traction_energy_func_factory,
    traction_func_factory,
)
from parametricpinn.training.metrics import mean_absolute_error, relative_l2_norm
from parametricpinn.types import Device, Tensor


@dataclass
class TrainingConfiguration:
    ansatz: StandardAnsatz
    material_model: str
    num_points_per_bc: int
    area_plate: float
    weight_pde_loss: float
    weight_energy_loss: float
    weight_traction_bc_loss: float
    weight_free_traction_bc_loss: float
    weight_dirichlet_bc_loss: float
    weight_symmetry_loss: float
    training_dataset: SimplifiedDogBoneTrainingDataset2D
    number_training_epochs: int
    training_batch_size: int
    validation_dataset: ValidationDataset2D
    validation_interval: int
    output_subdirectory: str
    project_directory: ProjectDirectory
    device: Device


def train_parametric_pinn(train_config: TrainingConfiguration) -> None:
    ansatz = train_config.ansatz
    material_model = train_config.material_model
    num_points_per_bc = train_config.num_points_per_bc
    area_plate = train_config.area_plate
    weight_pde_loss = train_config.weight_pde_loss
    weight_energy_loss = train_config.weight_energy_loss
    weight_traction_bc_loss = train_config.weight_traction_bc_loss
    weight_free_traction_bc_loss = train_config.weight_free_traction_bc_loss
    weight_dirichlet_bc_loss = train_config.weight_dirichlet_bc_loss
    weight_symmetry_loss = train_config.weight_symmetry_loss
    train_dataset = train_config.training_dataset
    train_num_epochs = train_config.number_training_epochs
    train_batch_size = train_config.training_batch_size
    valid_dataset = train_config.validation_dataset
    valid_batch_size = len(valid_dataset)
    valid_interval = train_config.validation_interval
    output_subdir = train_config.output_subdirectory
    project_directory = train_config.project_directory
    device = train_config.device

    loss_metric = torch.nn.MSELoss()

    ### Loss function
    momentum_equation_func = momentum_equation_func_factory(material_model)
    traction_func = traction_func_factory(material_model)
    stress_func = stress_func_factory(material_model)
    traction_energy_func = traction_energy_func_factory(material_model)
    strain_energy_func = strain_energy_func_factory(material_model)

    lambda_pde_loss = torch.tensor(weight_pde_loss, requires_grad=True).to(device)
    lambda_traction_bc_loss = torch.tensor(
        weight_traction_bc_loss, requires_grad=True
    ).to(device)
    lambda_free_traction_bc_loss = torch.tensor(
        weight_free_traction_bc_loss, requires_grad=True
    ).to(device)
    lambda_dirichlet_bc_loss = torch.tensor(
        weight_dirichlet_bc_loss, requires_grad=True
    ).to(device)
    lambda_energy_loss = torch.tensor(weight_energy_loss, requires_grad=True).to(device)
    lambda_symmetry_loss = torch.tensor(weight_symmetry_loss, requires_grad=True).to(
        device
    )

    def loss_func(
        ansatz: StandardAnsatz,
        collocation_data: TrainingData2DCollocation,
        traction_bc_data: TrainingData2DTractionBC,
    ) -> tuple[Tensor, Tensor, Tensor]:
        def loss_func_pde(
            ansatz: StandardAnsatz, collocation_data: TrainingData2DCollocation
        ) -> Tensor:
            x_coor = collocation_data.x_coor.to(device)
            x_E = collocation_data.x_E
            x_nu = collocation_data.x_nu
            x_param = torch.concat((x_E, x_nu), dim=1).to(device)
            volume_force = collocation_data.f.to(device)
            y_true = torch.zeros_like(x_coor).to(device)
            y = momentum_equation_func(ansatz, x_coor, x_param, volume_force)
            return loss_metric(y_true, y)

        def loss_func_traction_bc(
            ansatz: StandardAnsatz, traction_bc_data: TrainingData2DTractionBC
        ) -> Tensor:
            slice_0 = slice(0, num_points_per_bc)
            x_coor = traction_bc_data.x_coor[slice_0, :].to(device)
            x_E = traction_bc_data.x_E[slice_0, :]
            x_nu = traction_bc_data.x_nu[slice_0, :]
            x_param = torch.concat((x_E, x_nu), dim=1).to(device)
            normal = traction_bc_data.normal[slice_0, :].to(device)
            y_true = traction_bc_data.y_true[slice_0, :].to(device)
            y = traction_func(ansatz, x_coor, x_param, normal)
            return loss_metric(y_true, y)

        def loss_func_free_traction_bc(
            ansatz: StandardAnsatz, traction_bc_data: TrainingData2DTractionBC
        ) -> Tensor:
            slice_0 = slice(num_points_per_bc, len(traction_bc_data.x_coor))
            x_coor = traction_bc_data.x_coor[slice_0, :].to(device)
            x_E = traction_bc_data.x_E[slice_0, :]
            x_nu = traction_bc_data.x_nu[slice_0, :]
            x_param = torch.concat((x_E, x_nu), dim=1).to(device)
            normal = traction_bc_data.normal[slice_0, :].to(device)
            y_true = traction_bc_data.y_true[slice_0, :].to(device)
            y = traction_func(ansatz, x_coor, x_param, normal)
            return loss_metric(y_true, y)

        def loss_func_dirichlet_bc(ansatz: StandardAnsatz) -> Tensor:
            coor_x = -60.0
            coor_y_min = -15.0
            coor_y_max = 15.0
            youngs_modulus = 210000.0
            poissons_ratio = 0.3
            u_x = u_y = 0.0

            x_coor_x = torch.full(
                (num_points_per_bc, 1),
                coor_x,
                dtype=torch.float64,
                requires_grad=True,
            )
            x_coor_y = torch.linspace(
                coor_y_min,
                coor_y_max,
                steps=num_points_per_bc,
                dtype=torch.float64,
                requires_grad=True,
            ).reshape((-1, 1))
            x_coor = torch.concat((x_coor_x, x_coor_y), dim=1).to(device)
            x_E = torch.full(
                (num_points_per_bc, 1),
                youngs_modulus,
                dtype=torch.float64,
                requires_grad=True,
            )
            x_nu = torch.full(
                (num_points_per_bc, 1),
                poissons_ratio,
                dtype=torch.float64,
                requires_grad=True,
            )
            x_param = torch.concat((x_E, x_nu), dim=1).to(device)
            x = torch.concat((x_coor, x_param), dim=1).to(device)

            def _loss_func_dirichlet_bc() -> Tensor:
                y = ansatz(x)
                y_true = (
                    torch.tensor([u_x, u_y], requires_grad=True, dtype=torch.float64)
                    .repeat((num_points_per_bc, 1))
                    .to(device)
                )
                return loss_metric(y_true, y)

            return _loss_func_dirichlet_bc()
            # coor_x_max = 40
            # coor_y_min = -15
            # coor_y_max = 15
            # youngs_modulus = 210000.0
            # poissons_ratio = 0.3
            # u_x = 0.07

            # x_coor_x = torch.full(
            #     (num_points_per_bc, 1),
            #     coor_x_max,
            #     dtype=torch.float64,
            #     requires_grad=True,
            # )
            # x_coor_y = torch.linspace(
            #     coor_y_min,
            #     coor_y_max,
            #     steps=num_points_per_bc,
            #     dtype=torch.float64,
            #     requires_grad=True,
            # ).reshape((-1, 1))
            # x_coor = torch.concat((x_coor_x, x_coor_y), dim=1).to(device)
            # x_E = torch.full(
            #     (num_points_per_bc, 1),
            #     youngs_modulus,
            #     dtype=torch.float64,
            #     requires_grad=True,
            # )
            # x_nu = torch.full(
            #     (num_points_per_bc, 1),
            #     poissons_ratio,
            #     dtype=torch.float64,
            #     requires_grad=True,
            # )
            # x_param = torch.concat((x_E, x_nu), dim=1).to(device)
            # x = torch.concat((x_coor, x_param), dim=1).to(device)

            # def _loss_func_dirichlet_bc() -> Tensor:
            #     y = ansatz(x)[:, 0].reshape((-1, 1))
            #     y_true = torch.full((num_points_per_bc, 1), u_x).to(device)
            #     return loss_metric(y_true, y)

            # def _loss_func_sigma_xy_bc() -> Tensor:
            #     shear_stress_filter = (
            #         torch.tensor([[0.0, 1.0], [1.0, 0.0]])
            #         .repeat(num_points_per_bc, 1, 1)
            #         .to(device)
            #     )
            #     stress_tensors = stress_func(ansatz, x_coor, x_param)
            #     y = shear_stress_filter * stress_tensors
            #     y_true = (
            #         torch.tensor([[0.0, 0.0], [0.0, 0.0]])
            #         .repeat(num_points_per_bc, 1, 1)
            #         .to(device)
            #     )
            #     return loss_metric(y_true, y)

            # loss_dirichlet = _loss_func_dirichlet_bc()
            # loss_sigma_xy = _loss_func_sigma_xy_bc()
            # return loss_dirichlet + loss_sigma_xy

        def loss_func_energy(
            ansatz: StandardAnsatz,
            collocation_data: TrainingData2DCollocation,
            traction_bc_data: TrainingData2DTractionBC,
        ) -> Tensor:
            area = torch.tensor(area_plate)
            x_coor_int = collocation_data.x_coor.to(device)
            x_E_int = collocation_data.x_E
            x_nu_int = collocation_data.x_nu
            x_param_int = torch.concat((x_E_int, x_nu_int), dim=1).to(device)
            strain_energy = strain_energy_func(ansatz, x_coor_int, x_param_int, area)

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

        def loss_func_symmetry(
            ansatz: StandardAnsatz,
            traction_bc_data: TrainingData2DTractionBC,
        ) -> Tensor:
            # def _loss_func_dirichlet_bc() -> Tensor:
            #     slice_top = slice(num_points_per_bc, 2 * num_points_per_bc)
            #     x_coor_top = traction_bc_data.x_coor[slice_top, :].to(device)
            #     x_E_top = traction_bc_data.x_E[slice_top, :]
            #     x_nu_top = traction_bc_data.x_nu[slice_top, :]
            #     x_param_top = torch.concat((x_E_top, x_nu_top), dim=1).to(device)
            #     x_top = torch.concat((x_coor_top, x_param_top), dim=1).to(device)
            #     y_top = ansatz(x_top)

            #     slice_bottom = slice(2 * num_points_per_bc, 3 * num_points_per_bc)
            #     x_coor_bottom = traction_bc_data.x_coor[slice_bottom, :].to(device)
            #     x_E_bottom = traction_bc_data.x_E[slice_bottom, :]
            #     x_nu_bottom = traction_bc_data.x_nu[slice_bottom, :]
            #     x_param_bottom = torch.concat((x_E_bottom, x_nu_bottom), dim=1).to(
            #         device
            #     )
            #     x_bottom = torch.concat((x_coor_bottom, x_param_bottom), dim=1).to(
            #         device
            #     )
            #     y_bottom = ansatz(x_bottom)

            #     y = y_top + y_bottom
            #     y_true = torch.zeros_like(y, device=device)
            #     return loss_metric(y_true, y)

            # def _loss_func_sigma_xy_bc() -> Tensor:
            #     coor_x_min = -60.0
            #     coor_x_max = -40.0
            #     coor_y = 0.0
            #     youngs_modulus = 210000.0
            #     poissons_ratio = 0.3

            #     x_coor_x = torch.linspace(
            #         coor_x_min,
            #         coor_x_max,
            #         steps=num_points_per_bc,
            #         dtype=torch.float64,
            #         requires_grad=True,
            #     ).reshape((-1, 1))
            #     x_coor_y = torch.full(
            #         (num_points_per_bc, 1),
            #         coor_y,
            #         dtype=torch.float64,
            #         requires_grad=True,
            #     )
            #     x_coor = torch.concat((x_coor_x, x_coor_y), dim=1).to(device)
            #     x_E = torch.full(
            #         (num_points_per_bc, 1),
            #         youngs_modulus,
            #         dtype=torch.float64,
            #         requires_grad=True,
            #     )
            #     x_nu = torch.full(
            #         (num_points_per_bc, 1),
            #         poissons_ratio,
            #         dtype=torch.float64,
            #         requires_grad=True,
            #     )
            #     x_param = torch.concat((x_E, x_nu), dim=1).to(device)
            #     x = torch.concat((x_coor, x_param), dim=1).to(device)
            #     shear_stress_filter = (
            #         torch.tensor([[0.0, 1.0], [1.0, 0.0]])
            #         .repeat(num_points_per_bc, 1, 1)
            #         .to(device)
            #     )
            #     stress_tensors = stress_func(ansatz, x_coor, x_param)
            #     y = shear_stress_filter * stress_tensors
            #     y_true = (
            #         torch.tensor([[0.0, 0.0], [0.0, 0.0]])
            #         .repeat(num_points_per_bc, 1, 1)
            #         .to(device)
            #     )
            #     return loss_metric(y_true, y)

            def _loss_func_sigma_xy_bc() -> Tensor:
                coor_x = -60.0
                coor_y_min = -15.0
                coor_y_max = 15.0
                youngs_modulus = 210000.0
                poissons_ratio = 0.3

                x_coor_x = torch.full(
                    (num_points_per_bc, 1),
                    coor_x,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                x_coor_y = torch.linspace(
                    coor_y_min,
                    coor_y_max,
                    steps=num_points_per_bc,
                    dtype=torch.float64,
                    requires_grad=True,
                ).reshape((-1, 1))
                x_coor = torch.concat((x_coor_x, x_coor_y), dim=1).to(device)
                x_E = torch.full(
                    (num_points_per_bc, 1),
                    youngs_modulus,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                x_nu = torch.full(
                    (num_points_per_bc, 1),
                    poissons_ratio,
                    dtype=torch.float64,
                    requires_grad=True,
                )
                x_param = torch.concat((x_E, x_nu), dim=1).to(device)
                shear_stress_filter = (
                    torch.tensor([[0.0, 1.0], [1.0, 0.0]])
                    .repeat(num_points_per_bc, 1, 1)
                    .to(device)
                )
                stress_tensors = stress_func(ansatz, x_coor, x_param)
                y = shear_stress_filter * stress_tensors
                y_true = (
                    torch.tensor([[0.0, 0.0], [0.0, 0.0]])
                    .repeat(num_points_per_bc, 1, 1)
                    .to(device)
                )
                return loss_metric(y_true, y)

            # loss_dirichlet = _loss_func_dirichlet_bc()
            loss_sigma_xy = _loss_func_sigma_xy_bc()
            return loss_sigma_xy  # + loss_dirichlet

        loss_pde = lambda_pde_loss * loss_func_pde(ansatz, collocation_data)
        loss_traction_bc = lambda_traction_bc_loss * loss_func_traction_bc(
            ansatz, traction_bc_data
        )
        loss_free_traction_bc = (
            lambda_free_traction_bc_loss
            * loss_func_free_traction_bc(ansatz, traction_bc_data)
        )
        # loss_dirichlet_bc = lambda_dirichlet_bc_loss * loss_func_dirichlet_bc(ansatz)
        loss_dirichlet_bc = torch.tensor(0.0, device=device, requires_grad=True)
        loss_energy = lambda_energy_loss * loss_func_energy(
            ansatz, collocation_data, traction_bc_data
        )
        # loss_energy = torch.tensor(0.0, device=device, requires_grad=True)
        # loss_symmetry = lambda_symmetry_loss * loss_func_symmetry(
        #     ansatz, traction_bc_data
        # )
        loss_symmetry = torch.tensor(0.0, device=device, requires_grad=True)

        return (
            loss_pde,
            loss_traction_bc,
            loss_free_traction_bc,
            loss_dirichlet_bc,
            loss_energy,
            loss_symmetry,
        )

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
        shuffle=True,
        drop_last=False,
        collate_fn=train_dataset.get_collate_func(),
    )

    train_batches = iter(train_dataloader)
    for batch_collocation, batch_traction_bc in train_batches:
        box_length = 100
        box_height = 30
        fig_height = 4
        fig_length = (box_length / box_height) * fig_height + 1

        fig_collocation, ax_collocation = plt.subplots()
        fig_collocation.set_figheight(fig_height)
        fig_collocation.set_figwidth(fig_length)
        x_collocation = batch_collocation.x_coor[:, 0]
        y_collocation = batch_collocation.x_coor[:, 1]
        ax_collocation.scatter(x_collocation, y_collocation, edgecolors="none")
        save_path_collocation = project_directory.create_output_file_path(
            "scatter_collocation_points.pdf", output_subdir
        )
        fig_collocation.savefig(
            save_path_collocation,
            format="pdf",
            bbox_inches="tight",
            dpi=300,
        )

        fig_bc, ax_bc = plt.subplots()
        fig_bc.set_figheight(fig_height)
        fig_bc.set_figwidth(fig_length)
        x_bc = batch_traction_bc.x_coor[:num_points_per_bc, 0]
        y_bc = batch_traction_bc.x_coor[:num_points_per_bc, 1]
        x_bc_free = batch_traction_bc.x_coor[num_points_per_bc:, 0]
        y_bc_free = batch_traction_bc.x_coor[num_points_per_bc:, 1]
        ax_bc.scatter(x_bc, y_bc, edgecolors="none", c="r", label="traction BC")
        ax_bc.scatter(x_bc_free, y_bc_free, edgecolors="none", c="b", label="free BC")

        for i, coordinate in enumerate(batch_traction_bc.x_coor):
            coor_x = coordinate[0]
            coor_y = coordinate[1]
            norm_x = batch_traction_bc.normal[i, 0]
            norm_y = batch_traction_bc.normal[i, 1]
            ax_bc.arrow(coor_x, coor_y, norm_x, norm_y)

        ax_bc.legend()
        save_path_bc = project_directory.create_output_file_path(
            "scatter_bc_points.pdf", output_subdir
        )
        fig_bc.savefig(
            save_path_bc,
            format="pdf",
            bbox_inches="tight",
            dpi=300,
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

    # optimizer = torch.optim.Rprop(
    #     params=ansatz.parameters(),
    # )

    loss_hist_pde = []
    loss_hist_traction_bc = []
    loss_hist_free_traction_bc = []
    loss_hist_dirichlet_bc = []
    loss_hist_energy = []
    loss_hist_symmetry = []
    valid_hist_mae = []
    valid_hist_rl2 = []
    valid_epochs = []

    # Closure for LBFGS
    def loss_func_closure() -> float:
        optimizer.zero_grad()
        (
            loss_collocation,
            loss_traction_bc,
            loss_free_traction_bc,
            loss_dirichlet_bc,
            loss_energy,
            loss_symmetry,
        ) = loss_func(ansatz, batch_collocation, batch_traction_bc)
        loss = (
            loss_collocation
            + loss_traction_bc
            + loss_free_traction_bc
            + loss_dirichlet_bc
            + loss_energy
            + loss_symmetry
        )
        loss.backward(retain_graph=True)
        return loss.item()

    print("Start training ...")
    for epoch in range(train_num_epochs):
        train_batches = iter(train_dataloader)
        loss_hist_pde_batches = []
        loss_hist_traction_bc_batches = []
        loss_hist_free_traction_bc_batches = []
        loss_hist_dirichlet_bc_batches = []
        loss_hist_energy_batches = []
        loss_hist_symmetry_batches = []

        print(f"Number of batches: {len(train_batches)}")
        for batch_collocation, batch_traction_bc in train_batches:
            ansatz.train()

            # Forward pass
            (
                loss_pde,
                loss_traction_bc,
                loss_free_traction_bc,
                loss_dirichlet_bc,
                loss_energy,
                loss_symmetry,
            ) = loss_func(ansatz, batch_collocation, batch_traction_bc)

            # Update parameters
            # Adam
            # optimizer.zero_grad()
            # loss = loss_pde + loss_traction_bc + loss_free_traction_bc + loss_dirichlet_bc + loss_energy + loss_symmetry
            # loss.backward()
            # optimizer.step()

            # BFGS
            optimizer.step(loss_func_closure)

            loss_hist_pde_batches.append(loss_pde.detach().cpu().item())
            loss_hist_traction_bc_batches.append(loss_traction_bc.detach().cpu().item())
            loss_hist_free_traction_bc_batches.append(
                loss_free_traction_bc.detach().cpu().item()
            )
            loss_hist_dirichlet_bc_batches.append(
                loss_dirichlet_bc.detach().cpu().item()
            )
            loss_hist_energy_batches.append(loss_energy.detach().cpu().item())
            loss_hist_symmetry_batches.append(loss_symmetry.detach().cpu().item())

        mean_loss_pde = statistics.mean(loss_hist_pde_batches)
        mean_loss_traction_bc = statistics.mean(loss_hist_traction_bc_batches)
        mean_loss_free_traction_bc = statistics.mean(loss_hist_free_traction_bc_batches)
        mean_loss_dirichlet_bc = statistics.mean(loss_hist_dirichlet_bc_batches)
        mean_loss_energy = statistics.mean(loss_hist_energy_batches)
        mean_loss_symmetry = statistics.mean(loss_hist_symmetry_batches)
        loss_hist_pde.append(mean_loss_pde)
        loss_hist_traction_bc.append(mean_loss_traction_bc)
        loss_hist_free_traction_bc.append(mean_loss_free_traction_bc)
        loss_hist_dirichlet_bc.append(mean_loss_dirichlet_bc)
        loss_hist_energy.append(mean_loss_energy)
        loss_hist_symmetry.append(mean_loss_symmetry)

        print("##################################################")
        print(f"Epoch {epoch} / {train_num_epochs - 1}")
        print(f"PDE: \t\t {mean_loss_pde}")
        print(f"TRACTION BC: \t {mean_loss_traction_bc}")
        print(f"FREE TRACTION BC: \t {mean_loss_free_traction_bc}")
        print(f"DIRICHLET BC: \t {mean_loss_dirichlet_bc}")
        print(f"ENERGY: \t {mean_loss_energy}")
        print(f"SYMMETRY: \t {mean_loss_symmetry}")
        print("##################################################")
        if epoch % valid_interval == 0 or epoch == train_num_epochs:
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

    plot_loss_history(
        loss_hists=[
            loss_hist_pde,
            loss_hist_traction_bc,
            loss_hist_free_traction_bc,
            loss_hist_dirichlet_bc,
            loss_hist_energy,
            loss_hist_symmetry,
        ],
        loss_hist_names=[
            "PDE",
            "Traction BC",
            "Free Traction BC",
            "Dirichlet BC",
            "Energy",
            "Symmetry",
        ],
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
