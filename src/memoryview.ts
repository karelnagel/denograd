import { all_int, listStr } from './helpers.ts'

export type FmtStr = keyof typeof MemoryView.ARRAYS

type MemoryViewOptions<F extends FmtStr> = {
  byteOffset?: number
  byteLength?: number
  fmt?: FmtStr
  shape?: number[]
}
function reshape1DToMultiD<T>(array: T[], shape: number[]): any {
  if (shape.length === 0) return array[0]
  if (shape.length === 1) {
    if (array.length !== shape[0]) throw new Error(`reshape1DToMultiD: shape mismatch. Expected length=${shape[0]}, got ${array.length}`)
    return array
  }

  const [head, ...tail] = shape
  const chunkSize = tail.reduce((acc, dim) => acc * dim, 1)
  const result = []
  let offset = 0
  for (let i = 0; i < head; i++) {
    const chunk = array.slice(offset, offset + chunkSize)
    offset += chunkSize
    // Recursively reshape the chunk
    result.push(reshape1DToMultiD(chunk, tail))
  }
  return result
}

type Arr<T extends FmtStr> = InstanceType<typeof MemoryView.ARRAYS[T]>
type Const<T extends FmtStr> = T extends 'q' | 'Q' ? bigint : T extends '?' ? boolean : number
export class MemoryView<F extends FmtStr = 'B'> {
  buffer: ArrayBuffer
  byteOffset: number
  byteLength: number
  format: F = 'B' as F

  // New fields for multi-dimensional logic:
  shape: number[] = []
  toString = () => `new MemoryView(new ${this.asTypedArray.constructor.name}(${listStr(this.to1DList())}), {byteOffset:${this.byteOffset}, byteLength:${this.byteLength}, fmt:'${this.format}', shape:${listStr(this.shape)}})`
  // Just like your code; constructor with shape support added at the end:
  constructor(input: MemoryView<F>, opts?: MemoryViewOptions<F>)
  constructor(input: number, opts?: MemoryViewOptions<F>)
  constructor(input: Const<F>[], opts?: MemoryViewOptions<F>)
  constructor(input: Arr<F>, opts?: MemoryViewOptions<F>)
  constructor(input: ArrayBuffer, opts?: MemoryViewOptions<F>)
  constructor(input: number | Const<F>[] | ArrayBuffer | ArrayBufferView | MemoryView<F>, opts: MemoryViewOptions<F> = {}) {
    // using opt.fmt over
    this.format = (opts.fmt || (input instanceof MemoryView && input.format) || Object.entries(MemoryView.ARRAYS).find(([k, v]) => input instanceof v)?.[0] || 'B') as F
    if (input instanceof MemoryView) {
      this.buffer = input.buffer
      this.byteOffset = opts.byteOffset || input.byteOffset
      this.byteLength = opts.byteLength || input.byteLength
    } else if (typeof input === 'number') {
      this.buffer = new ArrayBuffer(input)
      this.byteOffset = opts.byteOffset || 0
      this.byteLength = opts.byteLength || input
    } else if (Array.isArray(input)) {
      const typed = new MemoryView.ARRAYS[this.format](input as any)
      this.buffer = typed.buffer
      this.byteOffset = opts.byteOffset || typed.byteOffset
      this.byteLength = opts.byteLength || typed.byteLength
    } else if (input instanceof ArrayBuffer) {
      this.buffer = input
      this.byteOffset = opts.byteOffset || opts.byteOffset || 0
      this.byteLength = opts.byteLength || (input.byteLength - this.byteOffset)
    } else {
      this.buffer = input.buffer
      this.byteOffset = opts.byteOffset || input.byteOffset
      this.byteLength = opts.byteLength || input.byteLength
    }
    this.setShape(opts.shape)
  }
  private setShape = (shape?: number[]) => {
    if (shape?.length) {
      if (!all_int(shape)) throw new Error(`Shape can only have ints: [${shape}]`)
      if (shape.reduce((acc, dim) => acc * dim, 1) !== this.length) throw new Error(`Provided shape [${shape}] does not match total elements = ${this.length}`)
      this.shape = shape.slice()
    } else if (this.length === 1) this.shape = []
    else this.shape = [this.length]
  }
  get strides(): number[] {
    const strides = new Array(this.shape.length)
    let acc = 1
    for (let i = this.shape.length - 1; i >= 0; i--) {
      strides[i] = acc
      acc *= this.shape[i]
    }
    return strides
  }

  /** Reinterpret the underlying type. */
  cast = <NewF extends FmtStr>(fmt: NewF, shape?: number[]): MemoryView<NewF> => {
    this.format = fmt as any
    this.setShape(shape)
    return this as any
  }

  /** The number of typed elements (1D) in this MemoryView. */
  get length() {
    return this.byteLength / this.BYTES_PER_ELEMENT
  }

  /** Returns the size of one element in bytes. */
  get BYTES_PER_ELEMENT() {
    return MemoryView.ARRAYS[this.format].BYTES_PER_ELEMENT
  }

  /** A typed-array “view” into the memory (creates on-the-fly). */
  get asTypedArray(): Arr<F> {
    return new MemoryView.ARRAYS[this.format](this.buffer, this.byteOffset, this.length) as Arr<F>
  }

  /** Return the underlying bytes as a Uint8Array. */
  toBytes = (): Uint8Array => {
    return new Uint8Array(this.buffer, this.byteOffset, this.byteLength)
  }

  /** Access multi-dimensional elements by index. */
  getValue(...indices: number[]): Const<F> {
    if (indices.length !== this.shape.length) throw new Error(`Expected ${this.shape.length} indices; got ${indices.length}`)
    let offset = 0
    for (let i = 0; i < indices.length; i++) {
      if (indices[i] < 0 || indices[i] >= this.shape[i]) {
        throw new RangeError(`Index ${i} = ${indices[i]} is out of bounds for dimension size ${this.shape[i]}`)
      }
      offset += indices[i] * this.strides[i]
    }
    if (this.isBoolean) return Boolean(this.asTypedArray[offset]) as Const<F>
    return this.asTypedArray[offset] as Const<F>
  }

  /** Set multi-dimensional element. */
  setValue(value: Const<F>, ...indices: number[]) {
    if (indices.length !== this.shape.length) throw new Error(`Expected shape=${listStr(this.shape)} indices; got ${listStr(indices)}`)
    let offset = 0
    for (let i = 0; i < indices.length; i++) {
      if (indices[i] < 0 || indices[i] >= this.shape[i]) throw new RangeError(`Index ${i} = ${indices[i]} is out of bounds for dimension size ${this.shape[i]}`)
      offset += indices[i] * this.strides[i]
    }
    this.asTypedArray[offset] = value as number // or cast to BigInt, depending on format
  }

  /** Flatten back to a single dimension. */
  flat() {
    this.shape = [this.length]
    return this
  }

  /** Reshape to a new multi-dimensional shape (must have same total # of elements). */
  reshape(newShape: number[]) {
    const size = newShape.reduce((acc, dim) => acc * dim, 1)
    if (size !== this.length) {
      throw new Error(
        `reshape: new shape [${newShape.join(', ')}] has size ${size}, but total elements = ${this.length}`,
      )
    }
    this.shape = newShape.slice()
    return this
  }

  /** Just like your existing "slice" but now it's 1D slicing. */
  slice = (begin: number, end?: number) => {
    // Handle negative indices
    if (begin < 0) begin = this.length + begin
    if (end !== undefined && end < 0) end = this.length + end

    // Clamp indices to valid range
    begin = Math.max(0, Math.min(begin, this.length))
    if (end !== undefined) {
      end = Math.max(0, Math.min(end, this.length))
    }

    const itemBegin = begin * this.BYTES_PER_ELEMENT
    const itemEnd = end !== undefined ? end * this.BYTES_PER_ELEMENT : this.byteLength
    const newByteOffset = this.byteOffset + itemBegin
    const newByteLength = itemEnd - itemBegin
    return new MemoryView(this.buffer, { byteOffset: newByteOffset, byteLength: newByteLength / this.BYTES_PER_ELEMENT }).cast(this.format)
  }

  /**
   * Sets typed values from another array/MemoryView into our memory,
   * beginning at element offset = `offset`.
   */
  set = (array: number[] | ArrayBufferView | MemoryView, offset = 0) => {
    const source = array instanceof MemoryView ? array.asTypedArray : Array.isArray(array) ? new MemoryView(array).asTypedArray : array
    this.asTypedArray.set(source as any, offset)
    return this
  }

  /** If 'q' or 'Q', we treat them as 64-bit bigints in JS. */
  get isBigInt() {
    return this.format === 'q' || this.format === 'Q'
  }
  /** If '?' we store booleans (0/1) in a Uint8Array. */
  get isBoolean() {
    return this.format === '?'
  }

  /**
   * Convert to a list of JS numbers (1D). Throws if the format is one of the bigint types.
   */
  to1DList = (): Const<F>[] => {
    if (this.isBoolean) return Array.from(this.asTypedArray as Uint8Array, (x) => Boolean(x)) as Const<F>[]
    return Array.from(this.asTypedArray as Uint8Array) as Const<F>[]
  }
  toList = (): Const<F>[] => {
    return reshape1DToMultiD(this.to1DList(), this.shape)
  }

  // All typed array constructors mapped to "format" keys.
  static ARRAYS = {
    'b': Int8Array,
    'B': Uint8Array,
    'h': Int16Array,
    'H': Uint16Array,
    'i': Int32Array,
    'I': Uint32Array,
    'q': BigInt64Array,
    'Q': BigUint64Array,
    'e': Float16Array,
    'f': Float32Array,
    'd': Float64Array,
    '?': Uint8Array,
  }
}
