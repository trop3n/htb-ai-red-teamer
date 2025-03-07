import torch

# creating a tensor
x = torch.tensor([1.0, 2.0, 3.0])

# tensors can be moved to GPU if available
if torch.cuda.is_available():
    x = x.to('cuda')