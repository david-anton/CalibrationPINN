import torch
from torch.utils.data import Dataset

from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import CSVDataReader
from parametricpinn.types import Tensor


class ValidationDataset2D(Dataset):
    def __init__(
        self,
        input_subdir: str,
        num_points: int,
        num_samples: int,
        project_directory: ProjectDirectory
    ) -> None:
        self._input_subdir = input_subdir
        self._num_points = num_points
        self._num_samples = num_samples
        self._data_reader = CSVDataReader(project_directory)
        self._samples_x: list[Tensor] = []
        self._samples_y_true: list[Tensor] = []

        self._load_samples()

    def _load_samples(self) -> None:
        for i in range(self._num_samples):
            displacements = self._data_reader.read()
            parameters = self._data_reader.read()
            self._add_input_sample(displacements, parameters)
            self._add_output_sample(displacements)


    # def _add_input_sample(self, coordinates: Tensor, youngs_modulus: Tensor) -> None:
    #     x_coor = coordinates
    #     x_E = youngs_modulus.repeat(self._num_points, 1)
    #     x = torch.concat((x_coor, x_E), dim=1)
    #     self._samples_x.append(x)

    # def _add_output_sample(self, coordinates: Tensor, youngs_modulus: Tensor) -> None:
    #     y_true = calculate_displacements_solution_1D(
    #         coordinates=coordinates,
    #         length=self._geometry.length,
    #         youngs_modulus=youngs_modulus,
    #         traction=self._traction,
    #         volume_force=self._volume_force,
    #     )
    #     self._samples_y_true.append(cast(Tensor, y_true))

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        sample_x = self._samples_x[idx]
        sample_y_true = self._samples_y_true[idx]
        return sample_x, sample_y_true


# def collate_validation_data_1D(
#     batch: list[tuple[Tensor, Tensor]]
# ) -> tuple[Tensor, Tensor]:
#     x_batch = []
#     y_true_batch = []

#     for sample_x, sample_y_true in batch:
#         x_batch.append(sample_x)
#         y_true_batch.append(sample_y_true)

#     batch_x = torch.concat(x_batch, dim=0)
#     batch_y_true = torch.concat(y_true_batch, dim=0)
#     return batch_x, batch_y_true


# def create_validation_dataset_1D(
#     length: float,
#     traction: float,
#     volume_force: float,
#     min_youngs_modulus: float,
#     max_youngs_modulus: float,
#     num_points: int,
#     num_samples: int,
# ):
#     geometry = StretchedRod(length=length)
#     return ValidationDataset1D(
#         geometry=geometry,
#         traction=traction,
#         volume_force=volume_force,
#         min_youngs_modulus=min_youngs_modulus,
#         max_youngs_modulus=max_youngs_modulus,
#         num_points=num_points,
#         num_samples=num_samples,
#     )
