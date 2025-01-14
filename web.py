from tinygrad.tensor import Tensor


a = Tensor([5])
b = Tensor([4])

print(a.add(b).tolist())
