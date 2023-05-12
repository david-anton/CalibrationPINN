from dataclasses import asdict
from typing import Union

from parametricpinn.io import ProjectDirectory
from parametricpinn.types import DataClass


class DataclassWriter:
    def __init__(self, project_directory: ProjectDirectory) -> None:
        self._project_directory = project_directory
        self._correct_file_ending = ".txt"

    def write(self, data: DataClass, file_name: str, subdir_name: str) -> None:
        file_name = self._ensure_correct_file_ending(file_name)
        output_file_path = self._project_directory.create_output_file_path(
            file_name, subdir_name
        )
        data_dict = asdict(data)
        with open(output_file_path, "w") as f:
            for key, value in data_dict.items():
                f.write(f"{str(key)}: \t {str(value)}" + "\n")

    def _ensure_correct_file_ending(self, file_name: str) -> str:
        if file_name[-4:] == self._correct_file_ending:
            return file_name
        return file_name + self._correct_file_ending
