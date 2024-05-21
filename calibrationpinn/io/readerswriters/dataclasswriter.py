from dataclasses import asdict
from pathlib import Path

from calibrationpinn.io import ProjectDirectory
from calibrationpinn.io.readerswriters.utility import (
    ensure_correct_file_ending,
    join_output_file_path,
)
from calibrationpinn.types import DataClass


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
        file_name = ensure_correct_file_ending(
            file_name=file_name, file_ending=self._correct_file_ending
        )
        output_file_path = join_output_file_path(
            file_name=file_name,
            project_directory=self._project_directory,
            subdir_name=subdir_name,
            save_to_input_dir=save_to_input_dir,
        )
        data_dict = asdict(data)
        with open(output_file_path, "w") as f:
            for key, value in data_dict.items():
                f.write(f"{str(key)}: \t {str(value)}" + "\n")
