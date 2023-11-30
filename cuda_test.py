import torch

is_cuda_available = torch.cuda.is_available()
print(f"Is CUDA availöable: {is_cuda_available}")
