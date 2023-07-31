from pathlib import Path
from typing import Optional

import torch

from parametricpinn.io import ProjectDirectory
from parametricpinn.types import Module


class PytorchModelLoader:
    def __init__(self, project_directory: ProjectDirectory) -> None:
        self._project_directory = project_directory

    def load(
        self,
        model: Module,
        file_name: str,
        subdir_name: Optional[str] = None,
        load_from_output_dir: bool = True,
    ) -> Module:
        input_file_path = self._join_input_file_path(
            file_name, subdir_name, load_from_output_dir
        )
        model.load_state_dict(
            torch.load(input_file_path, map_location=torch.device("cpu"))
        )
        return model

    def _join_input_file_path(
        self, file_name: str, subdir_name: Optional[str], load_from_output_dir: bool
    ) -> Path:
        if load_from_output_dir:
            return self._project_directory.create_output_file_path(
                file_name, subdir_name
            )
        else:
            return self._project_directory.create_input_file_path(
                file_name, subdir_name
            )
