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
    ) -> None:
        file_name = self._ensure_correct_file_ending(file_name)
        output_file_path = self._project_directory.get_output_file_path(
            file_name, subdir_name
        )
        data.to_csv(output_file_path, header=header, index=index)

    def _ensure_correct_file_ending(self, file_name: str) -> str:
        if file_name[-4:] == self._correct_file_ending:
            return file_name
        return file_name + self._correct_file_ending
