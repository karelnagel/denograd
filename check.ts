import { is_eq, Tensor } from './denograd/mod.ts'

const expect = (val1: any) => ({
  toEqual: (val2: any) => {
    if (!is_eq(val1, val2)) throw new Error(`${val1} !== ${val2}`)
  },
})
let res: any = await new Tensor([3, 4]).add(new Tensor([4, 9])).tolist()
expect(res).toEqual([7, 13])

res = await new Tensor([5, 5, 5, 5]).matmul(new Tensor([3, 3, 3, 3])).tolist()
expect(res).toEqual(60)

res = await Tensor.rand([3, 5]).tolist()
expect(res.length).toEqual(3)
expect(res[0].length).toEqual(5)

console.log('success')
