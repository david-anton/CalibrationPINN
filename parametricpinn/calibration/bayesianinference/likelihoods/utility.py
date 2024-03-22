import torch

from parametricpinn.types import Tensor


def logarithmic_sum_of_exponentials(log_probs: Tensor) -> Tensor:
    max_log_prob = torch.max(log_probs)
    return max_log_prob + torch.log(torch.sum(torch.exp(log_probs - max_log_prob)))
