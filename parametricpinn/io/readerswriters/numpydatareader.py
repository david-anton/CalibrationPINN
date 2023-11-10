from typing import Literal, Optional

import numpy as np

from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters.utility import ensure_correct_file_ending
from parametricpinn.types import NPArray


class NumpyDataReader:
    def __init__(self, project_directory: ProjectDirectory) -> None:
        self._project_directory = project_directory
        self._correct_file_ending = ".npy"

    def read(
        self,
        file_name: str,
        allow_pickle: bool,
        encoding: Literal["ASCII", "latin1", "bytes"],
        subdir_name: Optional[str] = None,
    ) -> NPArray:
        file_name = ensure_correct_file_ending(
            file_name=file_name, file_ending=self._correct_file_ending
        )
        input_file_path = self._project_directory.get_input_file_path(
            file_name, subdir_name
        )
        return np.load(input_file_path, allow_pickle=allow_pickle, encoding=encoding)
