import os
import shutil
from pathlib import Path
from typing import Iterator

import pytest

from parametricpinn.errors import DirectoryNotFoundError, FileNotFoundError
from parametricpinn.io.projectdirectory import ProjectDirectory
from parametricpinn.settings import Settings


class FakeSetting(Settings):
    def __init__(self) -> None:
        self.PROJECT_DIR = Path() / "parametricpinn" / "io" / "tests"
        self.OUTPUT_SUBDIR = "test_output"
        self.INPUT_SUBDIR = "test_input"


settings = FakeSetting()


@pytest.fixture
def sut() -> Iterator[ProjectDirectory]:
    output_subdir_path = settings.PROJECT_DIR / settings.OUTPUT_SUBDIR
    input_subdir_path = settings.PROJECT_DIR / settings.INPUT_SUBDIR
    if not Path.is_dir(output_subdir_path):
        os.mkdir(output_subdir_path)
    if not Path.is_dir(input_subdir_path):
        os.mkdir(input_subdir_path)
    yield ProjectDirectory(settings)
    shutil.rmtree(output_subdir_path)
    shutil.rmtree(input_subdir_path)


# Path to output file
def test_create_output_file_path_without_additional_output_subdirectory(
    sut: ProjectDirectory,
) -> None:
    file_name = "output_file.txt"

    actual = sut.create_output_file_path(file_name=file_name)

    expected = settings.PROJECT_DIR / settings.OUTPUT_SUBDIR / file_name
    assert expected == actual


def test_create_output_file_path_with_additional_output_subdirectory(
    sut: ProjectDirectory,
) -> None:
    file_name = "output_file.txt"
    subdir_name = "output_subdirectory"

    actual = sut.create_output_file_path(
        file_name=file_name,
        subdir_name=subdir_name,
    )

    expected = settings.PROJECT_DIR / settings.OUTPUT_SUBDIR / subdir_name / file_name
    assert expected == actual


# Path to existing output file
def test_get_output_file_path_without_additional_output_subdirectory(
    sut: ProjectDirectory,
) -> None:
    file_name = "output_file.txt"
    expected = settings.PROJECT_DIR / settings.OUTPUT_SUBDIR / file_name
    open(expected, "w").close()

    actual = sut.get_output_file_path(file_name=file_name)

    assert expected == actual


def test_get_output_file_path_with_additional_output_subdirectory(
    sut: ProjectDirectory,
) -> None:
    file_name = "output_file.txt"
    subdir_name = "output_subdirectory"
    subdir_path = settings.PROJECT_DIR / settings.OUTPUT_SUBDIR / subdir_name
    Path.mkdir(subdir_path, parents=True)
    expected = subdir_path / file_name
    open(expected, "w").close()

    actual = sut.get_output_file_path(
        file_name=file_name,
        subdir_name=subdir_name,
    )

    assert expected == actual


# Path to input file
def test_get_input_file_path_without_additional_input_subdirectory(
    sut: ProjectDirectory,
) -> None:
    file_name = "input_file.txt"
    expected = settings.PROJECT_DIR / settings.INPUT_SUBDIR / file_name
    open(expected, "w").close()

    actual = sut.get_input_file_path(file_name=file_name)

    assert expected == actual


def test_get_input_file_path_with_additional_input_subdirectory(
    sut: ProjectDirectory,
) -> None:
    file_name = "input_file.txt"
    subdir_name = "input_subdirectory"
    subdir_path = settings.PROJECT_DIR / settings.INPUT_SUBDIR / subdir_name
    Path.mkdir(subdir_path, parents=True)
    expected = subdir_path / file_name
    open(expected, "w").close()

    actual = sut.get_input_file_path(file_name=file_name, subdir_name=subdir_name)

    assert expected == actual


# Errors
def test_get_output_file_path_for_not_existing_subdirectory(
    sut: ProjectDirectory,
) -> None:
    file_name = "output_file"
    subdir_name = "not_existing_output_subdirectory"

    with pytest.raises(DirectoryNotFoundError, match=subdir_name):
        sut.get_output_file_path(
            file_name=file_name,
            subdir_name=subdir_name,
        )


def test_get_output_file_path_for_not_existing_output_file(
    sut: ProjectDirectory,
) -> None:
    file_name = "not_existing_output_file"
    subdir_name = "output_subdirectory"
    subdir_path = settings.PROJECT_DIR / settings.OUTPUT_SUBDIR / subdir_name
    Path.mkdir(subdir_path, parents=True)

    with pytest.raises(FileNotFoundError, match=file_name):
        sut.get_output_file_path(
            file_name=file_name,
            subdir_name=subdir_name,
        )


def test_get_input_file_path_for_not_existing_subdirectory(
    sut: ProjectDirectory,
) -> None:
    file_name = "input_file"
    subdir_name = "not_existing_input_subdirectory"

    with pytest.raises(DirectoryNotFoundError, match=subdir_name):
        sut.get_input_file_path(
            file_name=file_name,
            subdir_name=subdir_name,
        )


def test_get_input_file_path_for_not_existing_input_file(
    sut: ProjectDirectory,
) -> None:
    file_name = "not_existing_input_file"
    subdir_name = "input_subdirectory"
    subdir_path = settings.PROJECT_DIR / settings.INPUT_SUBDIR / subdir_name
    Path.mkdir(subdir_path, parents=True)

    with pytest.raises(FileNotFoundError, match=file_name):
        sut.get_input_file_path(
            file_name=file_name,
            subdir_name=subdir_name,
        )
