from __future__ import print_function
import torch
import numpy as np

def printLine(value):
    a = value
    for index in range(50):
        a += value
    print("\n", a, "\n")

x = torch.empty(5, 3)
print(x)
printLine("*")

x = torch.rand(5, 3)
print(x)
printLine("*")

x = torch.zeros(5, 3, dtype=torch.long)
print(x)
printLine("*")

x = torch.tensor([5.5, 3])
print(x)
printLine("*")

x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size
print(x.size())

# matrix addition
y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))
printLine("*")

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
printLine("*")

y.add_(x) # Any operation that mutates a tensor in-place is post-fixed with an _.
print(y)
printLine("*")

print(result.equal(x+y)) #True
print(result.equal(torch.add(x, y))) #True
print(result.eq(x+y)) #tensor([[True, True, True], [True, True, True], [True, True, True], [True, True, True], [True, True, True]])
print(result.eq_(x+y)) #tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])

print(x[:, 1]) #standard NumPy-like indexing with all bells and whistles
printLine("*")

x = torch.randn(4, 4)
y = x.view(16) # torch.view resizes/reshapes the tensor
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
printLine("*")

x = torch.randn(1)
print(x)
print(x.item()) # .item() get the value as a Python number
printLine("*")

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
printLine("*")

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

print("is cuda available: ", torch.cuda.is_available())

