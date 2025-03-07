import torch

# create a tensor with gradient tracking
x = torch.tensor([1.0, 2.0], [3.0, 4.0], requires_grad=True)

# perform operations
y = x * 2
z = y.sum()
z.backward() # compute gradients automatically
print(x.grad) # gradient of z w.r.t x: [[2,2], [2, 2]]