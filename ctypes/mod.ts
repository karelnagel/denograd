export type DenoFnType = Deno.ToNativeParameterTypes<[Deno.NativeType]>[number] | void

export class Type<NativeValue extends DenoFnType, Value = NativeValue, SetNativeValue = NativeValue, SetValue = Value> {
  buffer: ArrayBuffer
  constructor(
    buffer?: ArrayBuffer,
    public offset: number = 0,
    public byteLength: number = 0,
    public alignment: number = 0,
  ) {
    this.buffer = buffer ?? new ArrayBuffer(this.byteLength)
  }
  get bytes() {
    return new Uint8Array(this.buffer, this.offset, this.byteLength)
  }
  protected _value(): Value {
    throw new Error()
  }
  get value() {
    return this._value()
  }
  protected _set(val: SetValue): void {
    throw new Error()
  }
  set(val: SetValue) {
    this._set(val)
    return this
  }
  protected _native = (): NativeValue => this._value() as any
  get native(): NativeValue {
    return this._native()
  }
  protected _setNative = (val: SetNativeValue) => this._set(val as any)
  setNative(val: SetNativeValue) {
    this._setNative(val)
    return this
  }
  ptr(): Pointer<typeof this> {
    return new Pointer().setNative(Deno.UnsafePointer.offset(Deno.UnsafePointer.of(this.buffer) as any, this.offset))
  }
  /** Doesn't change the underlying buffer */
  loadFromPtr(ptr: Pointer<typeof this>, offset = 0): typeof this {
    if (ptr.value) Deno.UnsafePointerView.copyInto(ptr.native as any, new Uint8Array(this.buffer, this.offset, this.byteLength), offset)
    return this
  }
  /** Changes the buffer to the pointed buffer */
  replaceWithPtr(ptr: Pointer<typeof this>, offset = 0): typeof this {
    if (ptr.value) this.buffer = Deno.UnsafePointerView.getArrayBuffer(ptr.native as any, this.byteLength, offset)
    return this
  }
  toString = () => {
    const val = this.value
    if (typeof val === 'object') return `${this.constructor.name}({ ${Object.entries(val!).map(([k, v]) => `${k}: ${v}`).join(', ')} })`
    else return `${this.constructor.name}(${val})`
  };
  [Symbol.for('nodejs.util.inspect.custom')](_depth: number, _options: any) {
    return this.toString()
  }
}
// UINTS
export class U8 extends Type<number> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 1, 1)
  }
  protected override _value = () => new Uint8Array(this.buffer, this.offset)[0]
  protected override _set = (val: number) => new Uint8Array(this.buffer, this.offset).set([val])
  static new = (val: number) => new U8().set(val)
}

export class U16 extends Type<number> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 2, 2)
  }
  protected override _value = () => new Uint16Array(this.buffer, this.offset)[0]
  protected override _set = (val: number) => new Uint16Array(this.buffer, this.offset).set([val])
  static new = (val: number) => new U16().set(val)
}

export class U32 extends Type<number> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 4, 4)
  }
  protected override _value = () => new Uint32Array(this.buffer, this.offset)[0]
  protected override _set = (val: number) => new Uint32Array(this.buffer, this.offset).set([val])
  static new = (val: number) => new U32().set(val)
}

export class U64 extends Type<bigint> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 8, 8)
  }
  protected override _value = () => new BigUint64Array(this.buffer, this.offset)[0]
  protected override _set = (val: bigint) => new BigUint64Array(this.buffer, this.offset).set([val])
  static new = (val: bigint) => new U64().set(val)
}
export class Size extends U64 {
  static override new = (val: bigint) => new Size().set(val)
}
// INTS
export class I8 extends Type<number> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 1, 1)
  }
  protected override _value = () => new Int8Array(this.buffer, this.offset)[0]
  protected override _set = (val: number) => new Int8Array(this.buffer, this.offset).set([val])
  static new = (val: number) => new I8().set(val)
}

export class I16 extends Type<number> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 2, 2)
  }
  protected override _value = () => new Int16Array(this.buffer, this.offset)[0]
  protected override _set = (val: number) => new Int16Array(this.buffer, this.offset).set([val])
  static new = (val: number) => new I16().set(val)
}

export class I32 extends Type<number> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 4, 4)
  }
  protected override _value = () => new Int32Array(this.buffer, this.offset)[0]
  protected override _set = (val: number) => new Int32Array(this.buffer, this.offset).set([val])
  static new = (val: number) => new I32().set(val)
}

export class I64 extends Type<bigint> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 8, 8)
  }
  protected override _value = () => new BigInt64Array(this.buffer, this.offset)[0]
  protected override _set = (val: bigint) => new BigInt64Array(this.buffer, this.offset).set([val])
  static new = (val: bigint) => new I64().set(val)
}

// FLOATS
export class F32 extends Type<number> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 4, 4)
  }
  protected override _value = () => new Float32Array(this.buffer, this.offset)[0]
  protected override _set = (val: number) => new Float32Array(this.buffer, this.offset).set([val])
  static new = (val: number) => new F32().set(val)
}

export class F64 extends Type<number> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 8, 8)
  }
  protected override _value = () => new Float64Array(this.buffer, this.offset)[0]
  protected override _set = (val: number) => new Float64Array(this.buffer, this.offset).set([val])
  static new = (val: number) => new F64().set(val)
}

// STRUCT
export class Struct<Value extends Record<string, Type<any>>> extends Type<ArrayBuffer, Value, ArrayBuffer | Deno.PointerValue, Partial<Value>> {
  protected override _set(val: Partial<Value>) {
    for (const [k, v] of Object.entries(val)) {
      this.value[k].set(v.value)
    }
  }
  protected override _native = () => this.buffer
  protected override _setNative = (val: ArrayBuffer | Deno.PointerValue) => {
    if (val instanceof ArrayBuffer) this.buffer = val
    else this.replaceWithPtr(new Pointer().setNative(val))
  }
}

// POINTER
export class Pointer<Value extends Type<any, any>> extends Type<Deno.PointerValue, bigint> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 8, 8)
  }
  protected override _value(): bigint {
    return new BigUint64Array(this.buffer, this.offset)[0]
  }
  protected override _set(val: bigint) {
    new BigUint64Array(this.buffer, this.offset).set([val])
  }
  protected override _native = () => Deno.UnsafePointer.create(this.value)
  protected override _setNative = (val: Deno.PointerValue) => this.buffer = new BigUint64Array([Deno.UnsafePointer.value(val)]).buffer
  static new = (val: bigint = 0n) => new Pointer().set(val)
}

// VOID
export class Void extends Type<void, bigint> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 8, 8)
  }
  protected override _value(): bigint {
    return new BigUint64Array(this.buffer, this.offset)[0]
  }
  protected override _set(val: bigint) {
    new BigUint64Array(this.buffer, this.offset).set([val])
  }
  protected override _native = () => {}
  protected override _setNative = (val: void) => {}
  static new = (val: bigint = 0n) => new Void().set(val)
}

// FUNCTION
export class Function extends Type<Deno.PointerValue, () => void> {
}
