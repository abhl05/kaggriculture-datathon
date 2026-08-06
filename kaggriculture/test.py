import torch

# 1. Check if the GPU is available
print("GPU Available: ", torch.cuda.is_available())

# 2. Check the number of GPUs detected
print("Number of GPUs: ", torch.cuda.device_count())

# 3. Get the name of the specific GPU (e.g., index 0)
if torch.cuda.is_available():
    print("GPU Name: ", torch.cuda.get_device_name(0))