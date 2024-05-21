from pathlib import Path
from typing import Optional

from calibrationpinn.errors import DirectoryNotFoundError, FileNotFoundError
from calibrationpinn.settings import Settings


class ProjectDirectory:
    def __init__(self, settings: Settings) -> None:
        self._project_dir = settings.PROJECT_DIR
        self._output_subdir_name = settings.OUTPUT_SUBDIR
        self._input_subdir_name = settings.INPUT_SUBDIR
        self._output_subdir: Path
        self._input_subdir: Path
        self._initialize_project_subdir_paths()

    def create_output_file_path(
        self, file_name: str, subdir_name: Optional[str] = None
    ) -> Path:
        subdir_path = self._output_subdir
        if subdir_name is not None:
            subdir_path = self._join_path_to_subdir(self._output_subdir, subdir_name)
        return self._join_path_to_file(subdir_path, file_name)

    def create_input_file_path(
        self, file_name: str, subdir_name: Optional[str] = None
    ) -> Path:
        subdir_path = self._input_subdir
        if subdir_name is not None:
            subdir_path = self._join_path_to_subdir(self._input_subdir, subdir_name)
        return self._join_path_to_file(subdir_path, file_name)

    def get_output_file_path(
        self, file_name: str, subdir_name: Optional[str] = None
    ) -> Path:
        subdir_path = self._output_subdir
        if subdir_name is not None:
            subdir_path = self._join_path_to_existing_subdir(
                self._output_subdir, subdir_name
            )
        return self._join_path_to_existing_file(subdir_path, file_name)

    def get_input_file_path(
        self, file_name: str, subdir_name: Optional[str] = None
    ) -> Path:
        subdir_path = self._input_subdir
        if subdir_name is not None:
            subdir_path = self._join_path_to_existing_subdir(
                self._input_subdir, subdir_name
            )
        return self._join_path_to_existing_file(subdir_path, file_name)

    def _initialize_project_subdir_paths(self) -> None:
        self._output_subdir = self._join_project_subdir_paths(self._output_subdir_name)
        self._input_subdir = self._join_project_subdir_paths(self._input_subdir_name)

    def _join_project_subdir_paths(self, subdir_name: str) -> Path:
        subdir_path = self._project_dir / subdir_name
        return subdir_path

    def _join_path_to_subdir(self, dir_path: Path, subdir_name: str) -> Path:
        subdir_path = dir_path / subdir_name
        if not Path.is_dir(subdir_path):
            Path.mkdir(subdir_path, parents=True)
        return subdir_path

    def _join_path_to_existing_subdir(self, dir_path: Path, subdir_name: str) -> Path:
        subdir_path = dir_path / subdir_name
        if not Path.is_dir(subdir_path):
            raise DirectoryNotFoundError(subdir_path)
        return subdir_path

    def _join_path_to_file(self, dir_path: Path, file_name: str) -> Path:
        if not Path.is_dir(dir_path):
            raise DirectoryNotFoundError(dir_path)
        return dir_path / file_name

    def _join_path_to_existing_file(self, dir_path: Path, file_name: str) -> Path:
        file_path = dir_path / file_name
        if not Path.is_file(file_path):
            raise FileNotFoundError(file_path)
        return file_path
