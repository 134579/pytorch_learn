import torch

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])  # type: torch.Tensor
print(x)

x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)  # override dtype!
print(x)  # result has the same size

print(x.size())

## add
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
y.add_(x)
print(x[:, 1])

## reshape
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

## tensor to int/float
x = torch.randn(1)
print(x)
print(x.item())

## AUTOGRAD
x = torch.ones(2, 2, requires_grad=True)
print(x)
x.to("cuda")

y = x + 2

z = y * y * 3

out = z.mean()

## backprop
out.backward()  # type: torch.Tensor
print(x.grad)

### crazy
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)
print(x.grad)

### stop autograd
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

## reshape backward
# x = torch.randn([100000, 1], requires_grad=True, device='cuda')
import torch
x = torch.randn([10000000, 1], requires_grad=True, device='cuda')
y = x.view([1, 10000000])
z = y @ y.t()
w = y @ torch.one
z.backward()
print(torch.all(x.grad == 2 * x))
