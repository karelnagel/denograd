from tinygrad import Tensor

a = Tensor.rand([2])
b = Tensor.rand([2])

(a * b).tolist()