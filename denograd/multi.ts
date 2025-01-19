import { DType } from './dtype.ts'
import { NotImplemented } from './helpers.ts'
import { MathTrait, UOp } from './ops.ts'

export class MultiLazyBuffer extends MathTrait<MultiLazyBuffer> {
  dtype:DType
  constructor(lbs: UOp[], axis?: number, real?: boolean[]) {
    super()
    throw new NotImplemented()
  }
}
