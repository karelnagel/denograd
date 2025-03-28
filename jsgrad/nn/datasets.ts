import { range } from '../helpers/helpers.ts'
import { Tensor } from '../tensor.ts'
import { tar_extract } from './state.ts'

export const mnist = async (device = undefined, baseUrl?: string): Promise<[Tensor, Tensor, Tensor, Tensor]> => {
  const base_url = baseUrl || 'https://storage.googleapis.com/cvdf-datasets/mnist/'
  const _mnist = (file: string) => Tensor.from_url(base_url + file)
  return await Promise.all([
    _mnist('train-images-idx3-ubyte.gz').then((x) => x.get({ start: 0x10 }).reshape([-1, 1, 28, 28]).to(device)),
    _mnist('train-labels-idx1-ubyte.gz').then((x) => x.get({ start: 8 }).to(device)),
    _mnist('t10k-images-idx3-ubyte.gz').then((x) => x.get({ start: 0x10 }).reshape([-1, 1, 28, 28]).to(device)),
    _mnist('t10k-labels-idx1-ubyte.gz').then((x) => x.get({ start: 8 }).to(device)),
  ])
}

export const cifar = async (device = undefined) => {
  const tt = tar_extract(await Tensor.from_url('https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'))
  const train = Tensor.cat(range(1, 6).map((i) => tt[`cifar-10-batches-bin/data_batch_${i}.bin`].reshape([-1, 3073]).to(device)))
  const test = tt['cifar-10-batches-bin/test_batch.bin'].reshape([-1, 3073]).to(device)
  return [train.get({}, { start: 1 }).reshape([-1, 3, 32, 32]), train.get({}, 0), test.get({}, { start: 1 }).reshape([-1, 3, 32, 32]), test.get({}, { start: 0 })] as const
}
