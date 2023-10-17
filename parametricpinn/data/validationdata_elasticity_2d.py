from parametricpinn.data.dataset import ValidationDataset2D, ValidationDataset2DConfig


def create_validation_dataset(config: ValidationDataset2DConfig) -> ValidationDataset2D:
    return ValidationDataset2D(
        input_subdir=config.input_subdir,
        num_points=config.num_points,
        num_samples=config.num_samples,
        project_directory=config.project_directory,
    )
