from typing import Optional

import numpy as np

from parametricpinn.io import ProjectDirectory
from parametricpinn.types import NPArray


class DATDataReader:
    def __init__(self, project_directory: ProjectDirectory) -> None:
        self._project_directory = project_directory
        self._correct_file_ending = ".dat"

    def read(self, file_name: str, subdir_name: Optional[str] = None) -> NPArray:
        file_name = self._ensure_correct_file_ending(file_name)
        input_file_path = self._project_directory.get_input_file_path(
            file_name, subdir_name
        )
        return np.loadtxt(input_file_path)

    def _ensure_correct_file_ending(self, file_name: str) -> str:
        if file_name[-4:] == self._correct_file_ending:
            return file_name
        return file_name + self._correct_file_ending
