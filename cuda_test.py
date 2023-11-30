import torch

device_count = torch.cuda.device_count()
is_cuda_available = torch.cuda.is_available()
current_device = torch.cuda.current_device()

print(f"Is CUDA available: {is_cuda_available}")
print(f"Device count: {device_count}")
print(f"Current device: {current_device}")
