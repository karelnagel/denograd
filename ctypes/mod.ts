type DenoFnType = Deno.ToNativeParameterTypes<[Deno.NativeType]>[number] | void

export abstract class Type<T extends DenoFnType> {
  constructor(public _value: T) {}
  abstract get buffer(): ArrayBuffer
  abstract fromBuffer(buf: ArrayBuffer): typeof this
  get value() {
    return this._value
  }
  get alignment() {
    return this.buffer.byteLength
  }
  ptr() {
    const buf = this.buffer
    return new Pointer<typeof this>(Deno.UnsafePointer.of(buf))
  }
}

// UINTS
export class U8 extends Type<number> {
  get buffer(): ArrayBuffer {
    return new Uint8Array([this.value]).buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    this._value = new Uint8Array(buf)[0]
    return this
  }
}
export class U16 extends Type<number> {
  get buffer(): ArrayBuffer {
    return new Uint16Array([this.value]).buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    this._value = new Uint16Array(buf)[0]
    return this
  }
}
export class U32 extends Type<number> {
  get buffer(): ArrayBuffer {
    return new Uint32Array([this.value]).buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    this._value = new Uint32Array(buf)[0]
    return this
  }
}
export class U64 extends Type<bigint> {
  get buffer(): ArrayBuffer {
    return new BigUint64Array([this.value]).buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    this._value = new BigUint64Array(buf)[0]
    return this
  }
}
export class Size extends U64 {}

// INTS
export class I8 extends Type<number> {
  get buffer(): ArrayBuffer {
    return new Int8Array([this.value]).buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    this._value = new Int8Array(buf)[0]
    return this
  }
}
export class I16 extends Type<number> {
  get buffer(): ArrayBuffer {
    return new Int16Array([this.value]).buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    this._value = new Int16Array(buf)[0]
    return this
  }
}
export class I32 extends Type<number> {
  get buffer(): ArrayBuffer {
    return new Int32Array([this.value]).buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    this._value = new Int32Array(buf)[0]
    return this
  }
}
export class I64 extends Type<bigint> {
  get buffer(): ArrayBuffer {
    return new BigInt64Array([this.value]).buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    this._value = new BigInt64Array(buf)[0]
    return this
  }
}

// FLOATS
export class F32 extends Type<number> {
  get buffer(): ArrayBuffer {
    return new Float32Array([this.value]).buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    this._value = new Float32Array(buf)[0]
    return this
  }
}
export class F64 extends Type<number> {
  get buffer(): ArrayBuffer {
    return new Float64Array([this.value]).buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    this._value = new Float64Array(buf)[0]
    return this
  }
}

// ENUM
export abstract class Enum<T extends Type<any>, K extends DenoFnType = T['value']> extends Type<K> {
  constructor(public item: T) {
    super(item.value)
  }
  get buffer(): ArrayBuffer {
    return this.item.buffer
  }
  override fromBuffer(buf: ArrayBuffer): this {
    this.item.fromBuffer(buf)
    return this
  }
}

// POINTER
export class Pointer<T extends Type<any> | null> extends Type<Deno.PointerValue> {
  constructor(value: Deno.PointerValue) {
    super(value)
  }
  get buffer(): ArrayBuffer {
    return new BigUint64Array([Deno.UnsafePointer.value(this._value)]).buffer
  }
  fromBuffer(buf: ArrayBuffer) {
    this._value = Deno.UnsafePointer.create(new BigUint64Array(buf)[0])
    return this
  }
  override ptr(): any {
    throw new Error("Can't call .ptr() on pointer")
  }
  load(type: T): T {
    if (type === null) return null as T
    const buf = type.buffer
    return type.fromBuffer(Deno.UnsafePointerView.getArrayBuffer(this.value as any, buf.byteLength, 0)) as T
  }
}

// STRUCT
const getOffset = (offset: number, alignment: number) => Math.ceil(offset / alignment) * alignment

export abstract class Struct<T extends Type<any>[]> extends Type<BufferSource> {
  items: T
  constructor(...items: T) {
    super(new Uint8Array())
    this.items = items
  }
  override get value() {
    return this.buffer
  }
  override get alignment(): number {
    return Math.max(...this.items.map((x) => x.alignment))
  }
  get buffer(): ArrayBuffer {
    let offsets: number[] = [], offset = 0

    for (const item of this.items) {
      const alignedOffset = getOffset(offset, item.alignment)
      offsets.push(alignedOffset)
      offset = alignedOffset + item.buffer.byteLength
    }

    const result = new Uint8Array(getOffset(offset, this.alignment))
    for (const [i, item] of this.items.entries()) result.set(new Uint8Array(item.buffer), offsets[i])

    return result.buffer
  }

  override fromBuffer(buf: ArrayBuffer): this {
    let offset = 0

    for (const item of this.items) {
      const alignedOffset = getOffset(offset, item.alignment)
      const size = item.buffer.byteLength
      item.fromBuffer(buf.slice(alignedOffset, alignedOffset + size))
      offset = alignedOffset + size
    }

    return this
  }
}

// FUNCTION
export class Function extends Type<Deno.PointerValue> {
  get buffer(): ArrayBuffer {
    return new Float32Array([]).buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    // this._value = new Float32Array(buf)[0]
    return this
  }
}

export class Void extends Type<void> {
  get buffer(): ArrayBuffer {
    return new Uint8Array().buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    return this
  }
}
