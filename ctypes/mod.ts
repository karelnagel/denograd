type DenoFnType = Deno.ToNativeParameterTypes<[Deno.NativeType]>[number]

export abstract class Type<T extends DenoFnType> {
  constructor(public _value: T) {}
  abstract get buffer(): ArrayBuffer
  abstract fromBuffer(buf: ArrayBuffer): typeof this
  get value() {
    return this._value
  }
  ptr() {
    const buf = this.buffer
    return new Pointer<typeof this>(Deno.UnsafePointer.of(buf))
  }
}

// UINTS
export class U8 extends Type<number> {
  get buffer(): ArrayBuffer {
    const view = new DataView(new ArrayBuffer(1))
    view.setUint8(0, this.value)
    return view.buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    const view = new DataView(buf)
    this._value = view.getUint8(0)
    return this
  }
}
export class U16 extends Type<number> {
  get buffer(): ArrayBuffer {
    const view = new DataView(new ArrayBuffer(2))
    view.setUint16(0, this.value)
    return view.buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    const view = new DataView(buf)
    this._value = view.getUint16(0)
    return this
  }
}
export class U32 extends Type<number> {
  get buffer(): ArrayBuffer {
    const view = new DataView(new ArrayBuffer(4))
    view.setUint32(0, this.value)
    return view.buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    const view = new DataView(buf)
    this._value = view.getUint32(0)
    return this
  }
}
export class U64 extends Type<bigint> {
  get buffer(): ArrayBuffer {
    const view = new DataView(new ArrayBuffer(8))
    view.setBigUint64(0, this.value)
    return view.buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    const view = new DataView(buf)
    this._value = view.getBigUint64(0)
    return this
  }
}

// INTS
export class I8 extends Type<number> {
  get buffer(): ArrayBuffer {
    const view = new DataView(new ArrayBuffer(1))
    view.setInt8(0, this.value)
    return view.buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    const view = new DataView(buf)
    this._value = view.getInt8(0)
    return this
  }
}
export class I16 extends Type<number> {
  get buffer(): ArrayBuffer {
    const view = new DataView(new ArrayBuffer(2))
    view.setInt16(0, this.value)
    return view.buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    const view = new DataView(buf)
    this._value = view.getInt16(0)
    return this
  }
}
export class I32 extends Type<number> {
  get buffer(): ArrayBuffer {
    const view = new DataView(new ArrayBuffer(4))
    view.setInt32(0, this.value)
    return view.buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    const view = new DataView(buf)
    this._value = view.getInt32(0)
    return this
  }
}
export class I64 extends Type<bigint> {
  get buffer(): ArrayBuffer {
    const view = new DataView(new ArrayBuffer(8))
    view.setBigInt64(0, this.value)
    return view.buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    const view = new DataView(buf)
    this._value = view.getBigInt64(0)
    return this
  }
}

// FLOATS
export class F32 extends Type<number> {
  get buffer(): ArrayBuffer {
    const view = new DataView(new ArrayBuffer(4))
    view.setFloat32(0, this.value)
    return view.buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    const view = new DataView(buf)
    this._value = view.getFloat32(0)
    return this
  }
}
export class F64 extends Type<number> {
  get buffer(): ArrayBuffer {
    const view = new DataView(new ArrayBuffer(8))
    view.setFloat64(0, this.value)
    return view.buffer
  }
  override fromBuffer(buf: ArrayBuffer) {
    const view = new DataView(buf)
    this._value = view.getFloat64(0)
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
export class Pointer<T extends Type<any>> extends Type<Deno.PointerValue> {
  constructor(value: Deno.PointerValue) {
    super(value)
  }
  get buffer(): ArrayBuffer {
    const view = new DataView(new ArrayBuffer(8))
    view.setBigUint64(0, 0n) //Todo get the actual pointer value
    return view.buffer
  }
  fromBuffer(buf: ArrayBuffer) {
    const view = new DataView(buf)
    this._value = view.getBigUint64(0) // TODO
    return this
  }
  override ptr(): any {
    throw new Error("Can't call .ptr() on pointer")
  }
  load(type: T): T {
    const buf = type.buffer
    return type.fromBuffer(Deno.UnsafePointerView.getArrayBuffer(this.value as any, buf.byteLength, 0))
  }
}

// STRUCT
function concatArrayBuffers(buffers: ArrayBuffer[]) {
  const totalLength = buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0)

  let offset = 0, result = new Uint8Array(totalLength)
  for (const buffer of buffers) {
    result.set(new Uint8Array(buffer), offset)
    offset += buffer.byteLength
  }

  return result.buffer
}

export abstract class Struct<T extends Type<any>[]> extends Type<BufferSource> {
  items: T
  constructor(...items: T) {
    super(new Uint8Array())
    this.items = items
  }
  override get value() {
    return this.buffer
  }
  get buffer() {
    return concatArrayBuffers(this.items.map((x) => x.buffer))
  }
  override fromBuffer(buf: ArrayBuffer): typeof this {
    let offset = 0
    for (const item of this.items) {
      const len = item.buffer.byteLength
      item.fromBuffer(buf.slice(offset, offset+len))
      offset += len
    }
    return this
  }
}
