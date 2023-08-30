from collections import OrderedDict

import torch
import torch.nn as nn

from parametricpinn.network.ffnn import FFNN

layer_sizes = [2, 4, 1]


def test_get_flattened_parameters() -> None:
    sut = FFNN(
        layer_sizes=layer_sizes, init_weights=nn.init.zeros_, init_bias=nn.init.zeros_
    )

    actual = sut.get_flattened_parameters()

    expected = torch.zeros(2 * 4 + 4 + 4 * 1 + 1)
    torch.testing.assert_close(actual, expected)


def test_get_and_set_flattened_parameters() -> None:
    sut = FFNN(layer_sizes=layer_sizes)

    flattened_parameters = torch.ones(2 * 4 + 4 + 4 * 1 + 1)
    sut.set_flattened_parameters(flattened_parameters=flattened_parameters)
    actual = sut.state_dict()

    expected = OrderedDict(
        [
            ("_output.0._fc_layer.weight", torch.ones((4, 2))),
            ("_output.0._fc_layer.bias", torch.ones((4,))),
            ("_output.1._fc_layer.weight", torch.ones((1, 4))),
            ("_output.1._fc_layer.bias", torch.ones((1,))),
        ]
    )
    torch.testing.assert_close(actual, expected)
