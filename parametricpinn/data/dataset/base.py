import torch


def generate_uniform_parameter_list(
    min_parameter: float, max_parameter: float, num_samples: int
) -> list[float]:
    return torch.linspace(min_parameter, max_parameter, num_samples).tolist()
