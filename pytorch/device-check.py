import torch

print('torch.cuda.is_available', torch.cuda.is_available(), torch.__version__)        # True
print('torch.cuda.device_count', torch.cuda.device_count())        # 2
print('torch.cuda.get_device_name', torch.cuda.get_device_name(0)) # 3090 ti


tensor = torch.tensor([1.0, 2.0, 3.0])
tensor = tensor.to(device='cuda:1')
tt = tensor * tensor
print('tensor * tensor', tt, 'ndim', tt.ndim, 'shape', tt.shape,
      'device', tensor.device)

matmul = tensor.matmul(tensor)
print('tensor.matmul(tensor)', matmul, 'ndim', matmul.ndim, 'shape', matmul.shape,
      'device', tensor.device)

tensor = tensor.cpu()
array = tensor.numpy()
print('tensor.numpy()', array, 'ndim', array.ndim, 'shape', array.shape,
      'device', tensor.device)


