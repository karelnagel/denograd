// import { Conv2d } from '../../src/nn/index.ts'
// import { Tensor } from '../../src/tensor.ts'
// import { compare } from '../helpers.ts'

// Deno.test(
//   'Conv2d.init',
//   compare(
//     [
//       [],
//     ],
//     () => {
//       const conv = new Conv2d(1, 1, 3)
//       const t = Tensor.rand([1, 1, 4, 4])
//       return conv.call(t)
//     },
//     [
//       'from tinygrad.nn import Conv2d',
//       'conv = Conv2d(1,1,3)',
//       't = tiny.Tensor.rand(1, 1, 4, 4)',
//       'out(conv(t))',
//     ],
//   ),
// )
