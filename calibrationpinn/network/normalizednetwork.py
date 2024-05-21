import torch
import torch.nn as nn

from calibrationpinn.types import Module, Tensor


class InputNormalizer(nn.Module):
    def __init__(self, min_inputs: Tensor, max_inputs: Tensor) -> None:
        super().__init__()
        self._min_inputs = min_inputs
        self._max_inputs = max_inputs
        self._input_ranges = max_inputs - min_inputs
        self._atol = 1e-7

    def forward(self, x: Tensor) -> Tensor:
        denominator = self._input_ranges
        mask_division = torch.isclose(
            denominator,
            torch.zeros_like(denominator, device=denominator.device),
            atol=self._atol,
        )
        return torch.where(
            mask_division,
            torch.tensor([0.0], device=denominator.device),
            (((x - self._min_inputs) / denominator) * 2.0) - 1.0,
        )


class OutputRenormalizer(nn.Module):
    def __init__(self, min_outputs: Tensor, max_outputs: Tensor) -> None:
        super().__init__()
        self._min_outputs = min_outputs
        self._max_outputs = max_outputs
        self._output_ranges = max_outputs - min_outputs

    def forward(self, x: Tensor) -> Tensor:
        return (((x + 1) / 2) * self._output_ranges) + self._min_outputs


class NormalizedNetwork(nn.Module):
    def __init__(
        self,
        network: Module,
        input_normalizer: Module,
        output_renormalizer: Module,
    ) -> None:
        super().__init__()
        self._network = network
        self._input_normalizer = input_normalizer
        self._output_renormalizer = output_renormalizer

    def forward(self, x: Tensor) -> Tensor:
        normalized_x = self._input_normalizer(x)
        normalized_y = self._network(normalized_x)
        y = self._output_renormalizer(normalized_y)
        return y


def create_normalized_network(
    network: Module,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
) -> NormalizedNetwork:
    input_normalizer = InputNormalizer(min_inputs, max_inputs)
    output_renormalizer = OutputRenormalizer(min_outputs, max_outputs)
    return NormalizedNetwork(
        network=network,
        input_normalizer=input_normalizer,
        output_renormalizer=output_renormalizer,
    )
