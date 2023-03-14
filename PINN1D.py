# Standard library imports
from collections import namedtuple

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Local library imports


### Configuration
# Set up
length = 100.0
youngs_modulus = 210000.0
traction = 10.0
volume_force = 5.0
# Network
layer_sizes = [1, 16, 16, 1]
# Training
num_pde_points = 100
batch_size_train = 1
num_epochs = 20
# Validation
valid_interval = 1
num_valid_points = 1000
batch_size_valid = 1

# Default floating point dtype
torch.set_default_dtype(torch.float64)

# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using {device} device")

# Seed
torch.manual_seed(0)
np.random.seed(0)


### Solution
def calculate_displacements_solution(x):
    return (traction / youngs_modulus) * x + (volume_force / youngs_modulus) * (
        length * x - 1 / 2 * x**2
    )


### Model
class FFNN(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self._layer_sizes = layer_sizes
        self._activation = nn.Tanh

        layers = []
        for i in range(1, len(self._layer_sizes) - 1):
            fc_layer = nn.Linear(
                in_features=self._layer_sizes[i - 1],
                out_features=self._layer_sizes[i],
                bias=True,
            )
            activation = self._activation()
            layers.append(fc_layer)
            layers.append(activation)

        fc_layer_out = nn.Linear(
            in_features=self._layer_sizes[-2],
            out_features=self._layer_sizes[-1],
            bias=True,
        )
        layers.append(fc_layer_out)

        self._output = nn.Sequential(*layers)

    def forward(self, x):
        return self._output(x)


class InputNormalization(nn.Module):
    def __init__(self, min_inputs, max_inputs):
        super().__init__()
        self._min_inputs = min_inputs
        self._max_inputs = max_inputs
        self._input_range = max_inputs - min_inputs

    def forward(self, x):
        return (((x - self._min_inputs) / self._input_range) * 2.0) - 1.0


class OutputRenormalization(nn.Module):
    def __init__(self, min_outputs, max_outputs):
        super().__init__()
        self._min_outputs = min_outputs
        self._max_outputs = max_outputs
        self._output_range = max_outputs - min_outputs

    def forward(self, x):
        return (((x + 1) / 2) * self._output_range) + self._min_outputs


class NormalizedAnsatz(nn.Module):
    def __init__(self, network, min_input, max_input, min_output, max_output):
        super().__init__()
        self._network = network
        self._input_normalization = InputNormalization(
            min_inputs=min_input, max_inputs=max_input
        )
        self._output_renormalization = OutputRenormalization(
            min_outputs=min_output, max_outputs=max_output
        )
        self._range_inputs = max_input - min_input

    def _boundary_data(self, x):
        return -1.0

    def _distance_function(self, x):
        return x / self._range_inputs

    def forward(self, x):
        normalized_x = self._input_normalization(x)

        normalized_ansatz = self._boundary_data(x) + (
            self._distance_function(x) * self._network(normalized_x)
        )
        renomalized_ansatz = self._output_renormalization(normalized_ansatz)
        return renomalized_ansatz


### Data
PDEData = namedtuple("PDEData", ["x_pde", "y_pde_true"])
StressBCData = namedtuple("StressBCData", ["x_stress_bc", "y_stress_bc_true"])


class TrainingDataset(Dataset):
    def __init__(self, length, traction, num_pde_points):
        self._length = length
        self._traction = traction
        self._num_pde_points = num_pde_points
        self._num_bc_points = 1
        self._num_samples = 1
        self._samples_pde = []
        self._samples_stress_bc = []

        x_pde = torch.linspace(
            0.0, self._length, self._num_pde_points, requires_grad=True
        ).view([self._num_pde_points, 1])
        y_pde_true = torch.zeros_like(x_pde)
        sample_pde = PDEData(x_pde=x_pde, y_pde_true=y_pde_true)
        x_stress_bc = torch.full(
            (self._num_bc_points, 1), self._length, requires_grad=True
        )
        y_stress_bc_true = torch.full((self._num_bc_points, 1), self._traction)
        sample_stress_bc = StressBCData(
            x_stress_bc=x_stress_bc, y_stress_bc_true=y_stress_bc_true
        )

        self._samples_pde.append(sample_pde)
        self._samples_stress_bc.append(sample_stress_bc)

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        sample_pde = self._samples_pde[idx]
        sample_stress_bc = self._samples_stress_bc[idx]
        return sample_pde, sample_stress_bc


def collate_training_data(batch):
    x_pde_batch = []
    y_pde_true_batch = []
    x_stress_bc_batch = []
    y_stress_bc_true_batch = []

    for sample_pde, sample_stress_bc in batch:
        x_pde_batch.append(sample_pde.x_pde)
        y_pde_true_batch.append(sample_pde.y_pde_true)
        x_stress_bc_batch.append(sample_stress_bc.x_stress_bc)
        y_stress_bc_true_batch.append(sample_stress_bc.y_stress_bc_true)

    batch_pde = PDEData(
        x_pde=torch.concat(x_pde_batch, dim=0),
        y_pde_true=torch.concat(y_pde_true_batch, dim=0),
    )
    batch_stress_bc = StressBCData(
        x_stress_bc=torch.concat(x_stress_bc_batch, dim=0),
        y_stress_bc_true=torch.concat(y_stress_bc_true_batch, dim=0),
    )
    return batch_pde, batch_stress_bc


class ValidationDataset(Dataset):
    def __init__(self, length, num_points):
        self._length = length
        self._num_points = num_points
        self._num_samples = 1
        self._samples_x = []
        self._samples_y_true = []

        x = torch.linspace(0.0, self._length, self._num_points).view(
            [self._num_points, 1]
        )
        y_true = calculate_displacements_solution(x)
        self._samples_x.append(x)
        self._samples_y_true.append(y_true)

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        sample_x = self._samples_x[idx]
        sample_y_true = self._samples_y_true[idx]
        return sample_x, sample_y_true


### Loss function
def pinn_func(model, x):
    stress = stress_func(model, x)
    stress_x = torch.autograd.grad(
        stress,
        x,
        grad_outputs=torch.ones_like(stress),
        retain_graph=True,
        create_graph=True,
    )[0]
    return stress_x + volume_force


def stress_func(model, x):
    u = model(x)
    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]
    return youngs_modulus * u_x


loss_metric = torch.nn.MSELoss()


def loss_func(model, pde_data, stress_bc_data):
    def loss_func_pde(model, pde_data):
        x_pde = pde_data.x_pde.to(device)
        y_pde_true = pde_data.y_pde_true.to(device)
        y_pde = pinn_func(model, x_pde)
        return loss_metric(y_pde_true, y_pde)

    def loss_func_stress_bc(model, stress_bc_data):
        x_stress_bc = stress_bc_data.x_stress_bc.to(device)
        y_stress_bc_true = stress_bc_data.y_stress_bc_true.to(device)
        y_stress_bc = stress_func(model, x_stress_bc)
        return loss_metric(y_stress_bc_true, y_stress_bc)

    loss_pde = loss_func_pde(model, pde_data)
    loss_stress_bc = loss_func_stress_bc(model, stress_bc_data)
    return loss_pde, loss_stress_bc


### Validation
valid_metric_mae = torch.nn.L1Loss()


def validate_model(model, valid_dataloader):
    with torch.no_grad():
        model.eval()
        valid_batches = iter(valid_dataloader)
        mae_hist_batches = []
        batch_counter = 0

        for x, y_true in valid_batches:
            x = x.to(device)
            y_true = y_true.to(device)
            y = model(x)
            mae_batch = valid_metric_mae(y_true, y)
            mae_hist_batches.append(mae_batch.item())
            batch_counter += 1

        mean_mae = sum(mae_hist_batches) / batch_counter
    return mean_mae


### Plotting
def plot_loss_hist(loss_hist_pde, loss_hist_stress_bc, file_name):
    plt.plot(loss_hist_pde, label="loss PDE")
    plt.plot(loss_hist_stress_bc, label="loss stress BC")
    plt.yscale("log")
    plt.title("Loss history")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
    plt.savefig(file_name)
    plt.clf()


def plot_valid_hist(valid_epochs, valid_hist, valid_metric, file_name):
    plt.plot(valid_epochs, valid_hist, label=valid_metric)
    plt.yscale("log")
    plt.title("Validation history")
    plt.xlabel("Epoch")
    plt.legend(loc="best")
    plt.savefig(file_name)
    plt.clf()


def plot_displacements(file_name):
    num_samples = 1000
    coordinates = np.linspace(0.0, length, num_samples).reshape((num_samples, 1))
    solution = calculate_displacements_solution(coordinates)
    coordinates = torch.Tensor(coordinates)
    prediction_ansatz = model(coordinates).detach().numpy()
    plt.plot(coordinates, solution, label="solution")
    plt.plot(coordinates, prediction_ansatz, label="prediction")
    plt.xlabel("Coordinate")
    plt.ylabel("Displacements")
    plt.legend(loc="best")
    plt.savefig(file_name)
    plt.clf()


####################################################################################################
if __name__ == "__main__":
    min_coordinate = torch.Tensor([0.0])
    max_coordinate = torch.Tensor([length])
    min_displacement = torch.Tensor([0.0])
    max_displacement = calculate_displacements_solution(max_coordinate)

    network = FFNN(layer_sizes=layer_sizes)
    model = NormalizedAnsatz(
        network=network,
        min_input=min_coordinate,
        max_input=max_coordinate,
        min_output=min_displacement,
        max_output=max_displacement,
    ).to(device)

    train_dataset = TrainingDataset(
        length=length, traction=traction, num_pde_points=num_pde_points
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size_train,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_training_data,
    )

    valid_dataset = ValidationDataset(length=length, num_points=num_valid_points)
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size_valid,
        shuffle=False,
        drop_last=False,
    )

    optimizer = torch.optim.LBFGS(
        params=model.parameters(),
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
    valid_epochs = []

    # Closure for LBFGS
    def loss_func_closure():
        optimizer.zero_grad()
        loss_pde, loss_stress_bc = loss_func(model, batch_pde, batch_stress_bc)
        loss = loss_pde + loss_stress_bc
        loss.backward()
        return loss

    for epoch in range(num_epochs):
        train_batches = iter(train_dataloader)
        loss_hist_pde_batches = []
        loss_hist_stress_bc_batches = []
        batch_counter = 0

        for batch_pde, batch_stress_bc in train_batches:
            # Forward pass
            loss_pde, loss_stress_bc = loss_func(model, batch_pde, batch_stress_bc)
            loss = loss_pde + loss_stress_bc

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update parameters
            optimizer.step(loss_func_closure)

            loss_pde = loss_pde.cpu().item()
            loss_stress_bc = loss_stress_bc.cpu().item()

            loss_hist_pde_batches.append(loss_pde)
            loss_hist_stress_bc_batches.append(loss_stress_bc)
            batch_counter += 1

        mean_batch_loss_pde = sum(loss_hist_pde_batches) / batch_counter
        mean_batch_loss_stress_bc = sum(loss_hist_stress_bc_batches) / batch_counter
        loss_hist_pde.append(mean_batch_loss_pde)
        loss_hist_stress_bc.append(mean_batch_loss_stress_bc)

        if epoch % 1 == 0:
            mae = validate_model(model, valid_dataloader)
            valid_hist_mae.append(mae)
            valid_epochs.append(epoch)

    plot_loss_hist(
        loss_hist_pde=loss_hist_pde,
        loss_hist_stress_bc=loss_hist_stress_bc,
        file_name="loss_parametric_pinn.png",
    )

    plot_valid_hist(
        valid_epochs=valid_epochs,
        valid_hist=valid_hist_mae,
        valid_metric="mae",
        file_name="mae_parametric_pinn.png",
    )

    plot_displacements(file_name="displacements_parametric_pinn.png")
