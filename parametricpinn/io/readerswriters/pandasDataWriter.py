from pathlib import Path
from typing import Union

from parametricpinn.io import ProjectDirectory
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
        header: Union[bool, list[str]] = False,
        index: bool = False,
        save_to_input_dir: bool = False,
    ) -> None:
        file_name = self._ensure_correct_file_ending(file_name)
        output_file_path = self._join_output_file_path(
            file_name, subdir_name, save_to_input_dir
        )
        data.to_csv(output_file_path, header=header, index=index)

    def _ensure_correct_file_ending(self, file_name: str) -> str:
        if file_name[-4:] == self._correct_file_ending:
            return file_name
        return file_name + self._correct_file_ending

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
