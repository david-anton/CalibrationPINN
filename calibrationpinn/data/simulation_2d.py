from calibrationpinn.data.dataset import SimulationDataset2D, SimulationDataset2DConfig


def create_simulation_dataset(config: SimulationDataset2DConfig) -> SimulationDataset2D:
    return SimulationDataset2D(
        input_subdir=config.input_subdir,
        num_points=config.num_points,
        num_samples=config.num_samples,
        project_directory=config.project_directory,
        read_from_output_dir=config.read_from_output_dir,
    )
