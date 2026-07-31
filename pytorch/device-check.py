import torch

print('torch.cuda.is_available', torch.cuda.is_available(), torch.__version__)        # True
print('torch.cuda.device_count', torch.cuda.device_count())        # 2
print('torch.cuda.get_device_name', torch.cuda.get_device_name(0)) # 3090 ti

# Scalar
scalar = torch.tensor(7)
print('scalar', scalar, 'ndim', scalar.ndim, 'item', scalar.item())
