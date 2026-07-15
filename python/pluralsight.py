# import random
import requests
import torch
x = torch.rand(5, 3)
# print(x, torch.__version__)
torch.set_default_dtype(torch.float64)

tensor_arr = torch.Tensor([[5, 3], [4, 3]]).type(torch.IntTensor)

print(torch.is_tensor(tensor_arr), tensor_arr, torch.numel(tensor_arr))

uninitialized = torch.Tensor(3, 5).type(torch.int64)
print(uninitialized)



