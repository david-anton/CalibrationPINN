from parametricpinn.types import Tensor


def repeat_tensor(tensor: Tensor, shape: tuple[int, ...]) -> Tensor:
    return tensor.repeat(shape)
