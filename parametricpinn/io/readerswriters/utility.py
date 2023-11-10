from pathlib import Path

from parametricpinn.io import ProjectDirectory


def ensure_correct_file_ending(file_name: str, file_ending: str) -> str:
    length_file_ending = len(file_ending)
    if file_name[-length_file_ending:] == file_ending:
        return file_name
    return file_name + file_ending


def join_output_file_path(
    file_name: str,
    subdir_name: str,
    project_directory: ProjectDirectory,
    save_to_input_dir: bool,
) -> Path:
    if save_to_input_dir:
        return project_directory.create_input_file_path(file_name, subdir_name)
    else:
        return project_directory.create_output_file_path(file_name, subdir_name)
