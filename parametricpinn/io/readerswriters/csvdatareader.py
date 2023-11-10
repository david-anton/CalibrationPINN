from typing import Any, Optional

import pandas as pd

from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters.utility import ensure_correct_file_ending
from parametricpinn.types import NPArray


class CSVDataReader:
    def __init__(self, project_directory: ProjectDirectory) -> None:
        self._project_directory = project_directory
        self._correct_file_ending = ".csv"

    def read(
        self, file_name: str, subdir_name: Optional[str] = None, header: Any = 0
    ) -> NPArray:
        file_name = ensure_correct_file_ending(
            file_name=file_name, file_ending=self._correct_file_ending
        )
        input_file_path = self._project_directory.get_input_file_path(
            file_name, subdir_name
        )
        return pd.read_csv(input_file_path, header=header).to_numpy()
