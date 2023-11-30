import torch

is_cuda_available = torch.cuda.is_available()
print(f"Is CUDA available: {is_cuda_available}")

device_count = torch.cuda.device_count()
print(f"Device count: {device_count}")

current_device = torch.cuda.current_device()
print(f"Current device: {current_device}")
