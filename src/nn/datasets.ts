// from tinygrad.tensor import Tensor
// from tinygrad.helpers import fetch
// from tinygrad.nn.state import tar_extract

import { Tensor } from '../tensor.ts'

// def mnist(device=None, fashion=False):
//   base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/" if fashion else "https://storage.googleapis.com/cvdf-datasets/mnist/"
//   def _mnist(file): return Tensor.from_url(base_url+file, gunzip=True)
//   return _mnist("train-images-idx3-ubyte.gz")[0x10:].reshape(-1,1,28,28).to(device), _mnist("train-labels-idx1-ubyte.gz")[8:].to(device), \
//          _mnist("t10k-images-idx3-ubyte.gz")[0x10:].reshape(-1,1,28,28).to(device), _mnist("t10k-labels-idx1-ubyte.gz")[8:].to(device)

export const mnist = (device = null, fashion?: boolean) => {
  return [new Tensor(), new Tensor(), new Tensor(), new Tensor()] as const
}
