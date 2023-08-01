from pathlib import Path

import torch

from parametricpinn.io import ProjectDirectory
from parametricpinn.types import Device, Module


class PytorchModelSaver:
    def __init__(self, project_directory: ProjectDirectory) -> None:
        self._project_directory = project_directory

    def save(
        self,
        model: Module,
        file_name: str,
        subdir_name: str,
        device: Device,
        save_to_input_dir: bool = False,
    ) -> None:
        output_file_path = self._join_output_file_path(
            file_name, subdir_name, save_to_input_dir
        )
        self._detach_model_parameters(model)
        model.cpu()
        torch.save(model.state_dict(), output_file_path)
        model.to(device)

    def _join_output_file_path(
        self, file_name: str, subdir_name: str, save_to_input_dir: bool
    ) -> Path:
        if save_to_input_dir:
            return self._project_directory.create_input_file_path(
                file_name, subdir_name
            )
        else:
            return self._project_directory.create_output_file_path(
                file_name, subdir_name
            )

    def _detach_model_parameters(self, model: Module):
        for p in model.parameters():
            p.detach()
