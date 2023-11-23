from parametricpinn.errors import UnvalidGPParametersError
from parametricpinn.types import Tensor, TensorSize


def validate_parameters_size(parameters: Tensor, valid_size: TensorSize) -> None:
    parameters_size = parameters.size()
    if parameters_size != valid_size:
        raise UnvalidGPParametersError(f"Parameter has unvalid size {parameters_size}")
