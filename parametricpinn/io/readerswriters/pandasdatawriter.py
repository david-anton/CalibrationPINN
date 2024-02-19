from pathlib import Path
from typing import Union

from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters.utility import (
    ensure_correct_file_ending,
    join_output_file_path,
)
from parametricpinn.types import PDDataFrame


class PandasDataWriter:
    def __init__(self, project_directory: ProjectDirectory) -> None:
        self._project_directory = project_directory
        self._correct_file_ending = ".csv"

    def write(
        self,
        data: PDDataFrame,
        file_name: str,
        subdir_name: str,
        header: Union[bool, list[str], tuple[str, ...]] = False,
        index: bool = False,
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
        data.to_csv(output_file_path, header=header, index=index)
