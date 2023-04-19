import torch.nn as nn

from parametricpinn.network.normalization.inputnormlizer import InputNormalizer
from parametricpinn.network.normalization.outputrenormalizer import OutputRenormalizer
from parametricpinn.types import Module, Tensor


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
