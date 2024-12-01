import { DType } from './dtype.ts'
import { getEnumString, type Ops, UOp, UPat } from './ops.ts'

const getPyOpsStr = (op: Ops) => `tiny.ops.Ops.${getEnumString(op)}`
export const pyStr = (v: any, useList = false): string => {
  if (Array.isArray(v)) return v.length ? (useList ? `[${v.map((x) => pyStr(x)).join(', ')}]` : `(${v.map((x) => pyStr(x)).join(', ')},)`) : '()'
  if (v === null || typeof v === 'undefined') return 'None'
  if (typeof v === 'boolean') return v ? 'True' : 'False'
  if (typeof v === 'number') return v === Infinity ? 'inf' : v === -Infinity ? '-inf' : Number.isNaN(v) ? 'math.nan' : v.toString()
  if (typeof v === 'string') return `"${v}"`

  if (v instanceof UPat) {
    // if src is UPat[][] we use list, if UPat[] then tuple
    const src = Array.isArray(v._inSrc) ? (Array.isArray(v._inSrc.at(0)) ? pyStr(v._inSrc.at(0), true) : pyStr(v._inSrc)) : pyStr(v._inSrc)
    return `tiny.ops.UPat(op=${v.op ? `(${v.op?.map(getPyOpsStr)},)` : 'None'}, dtype=${pyStr(v.dtype)}, src=${src}, arg=${pyStr(v.arg)}, name=${pyStr(v.name)}, allow_any_len=${pyStr(v.allowedLen === -1)}, location=${
      pyStr(v.location)
    }, custom_early_reject=${pyStr(v.customEarlyReject)})`
  }
  if (v instanceof UOp) return `tiny.ops.UOp(op=${getPyOpsStr(v.op)}, dtype=${pyStr(v.dtype)}, src=${pyStr(v.src)}, arg=${pyStr(v.arg)})`
  if (v instanceof DType) return `tiny.ops.DType(${pyStr(v.priority)}, ${pyStr(v.itemsize)}, ${pyStr(v.name)}, ${pyStr(v.fmt)}, ${pyStr(v.count)}, ${pyStr(v._scalar)})`

  if (typeof v === 'function') return 'lambda x: x'
  if (typeof v === 'object') return `{${Object.entries(v).map((entry) => `"${entry[0]}":${pyStr(entry[1])}`).join(',')}}`
  throw new Error(`Invalid value: ${v}`)
}
