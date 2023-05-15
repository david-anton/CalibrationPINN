from dataclasses import asdict
from pathlib import Path
from typing import Union

from parametricpinn.io import ProjectDirectory
from parametricpinn.types import DataClass


class DataclassWriter:
    def __init__(self, project_directory: ProjectDirectory) -> None:
        self._project_directory = project_directory
        self._correct_file_ending = ".txt"

    def write(
        self,
        data: DataClass,
        file_name: str,
        subdir_name: str,
        save_to_input_dir: bool = False,
    ) -> None:
        file_name = self._ensure_correct_file_ending(file_name)
        output_file_path = self._join_output_file_path(
            file_name, subdir_name, save_to_input_dir
        )
        data_dict = asdict(data)
        with open(output_file_path, "w") as f:
            for key, value in data_dict.items():
                f.write(f"{str(key)}: \t {str(value)}" + "\n")

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
