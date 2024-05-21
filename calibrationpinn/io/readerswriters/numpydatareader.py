from typing import Literal, Optional

import numpy as np

from calibrationpinn.io import ProjectDirectory
from calibrationpinn.io.readerswriters.utility import (
    ensure_correct_file_ending,
    join_input_file_path,
)
from calibrationpinn.types import NPArray


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
        read_from_output_dir: bool = False,
    ) -> NPArray:
        file_name = ensure_correct_file_ending(
            file_name=file_name, file_ending=self._correct_file_ending
        )
        input_file_path = join_input_file_path(
            file_name=file_name,
            subdir_name=subdir_name,
            project_directory=self._project_directory,
            read_from_output_dir=read_from_output_dir,
        )
        return np.load(input_file_path, allow_pickle=allow_pickle, encoding=encoding)
