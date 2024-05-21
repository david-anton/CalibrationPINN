from typing import TypeVar

from calibrationpinn.io import ProjectDirectory
from calibrationpinn.io.loaderssavers import ModuleType, PytorchModelLoader
from calibrationpinn.types import Device, Module


def freeze_model(model: Module) -> None:
    model.train(False)
    for parameters in model.parameters():
        parameters.requires_grad = False


def load_model(
    model: ModuleType,
    name_model_parameters_file: str,
    input_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
    load_from_output_dir=True,
) -> ModuleType:
    model_loader = PytorchModelLoader(project_directory)
    model = model_loader.load(
        model, name_model_parameters_file, input_subdir, load_from_output_dir
    ).to(device)
    freeze_model(model)
    return model
