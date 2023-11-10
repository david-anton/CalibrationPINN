import pandas as pd

from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters.utility import (
    ensure_correct_file_ending,
    join_output_file_path,
)
from parametricpinn.types import NPArray


class NumpyDataWriter:
    def __init__(self, project_directory: ProjectDirectory) -> None:
        self._project_directory = project_directory
        self._correct_file_ending = ".csv"

    def write(
        self,
        data: NPArray,
        file_name: str,
        subdir_name: str,
        save_to_input_dir: bool = False,
    ) -> None:
        file_name = ensure_correct_file_ending(
            file_name=file_name, file_ending=self._correct_file_ending
        )
        output_file_path = join_output_file_path(
            file_name=file_name,
            project_directory=self._project_directory,
            subdir_name=subdir_name,
            save_to_input_dir=save_to_input_dir,
        )
        pd.DataFrame(data).to_csv(output_file_path, header=False, index=False)
