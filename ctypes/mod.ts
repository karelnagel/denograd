export type DenoFnType = Deno.ToNativeParameterTypes<[Deno.NativeType]>[number] | void

export class Type<NativeValue extends DenoFnType, Value = NativeValue, SetValue = Value> {
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
  protected _setNative = (val: NativeValue) => this._set(val as any)
  setNative(val: NativeValue) {
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
}
// UINTS
export class U8 extends Type<number> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 1, 1)
  }
  protected override _value = () => new Uint8Array(this.buffer, this.offset)[0]
  protected override _set = (val: number) => new Uint8Array(this.buffer, this.offset).set([val])
}
export const u8 = (val: number) => new U8().set(val)

export class U16 extends Type<number> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 2, 2)
  }
  protected override _value = () => new Uint16Array(this.buffer, this.offset)[0]
  protected override _set = (val: number) => new Uint16Array(this.buffer, this.offset).set([val])
}
export const u16 = (val: number) => new U16().set(val)

export class U32 extends Type<number> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 4, 4)
  }
  protected override _value = () => new Uint32Array(this.buffer, this.offset)[0]
  protected override _set = (val: number) => new Uint32Array(this.buffer, this.offset).set([val])
}
export const u32 = (val: number) => new U32().set(val)

export class U64 extends Type<bigint> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 8, 8)
  }
  protected override _value = () => new BigUint64Array(this.buffer, this.offset)[0]
  protected override _set = (val: bigint) => new BigUint64Array(this.buffer, this.offset).set([val])
}
export const u64 = (val: bigint) => new U64().set(val)

export class Struct<Value extends Record<string, Type<any>>> extends Type<ArrayBuffer, Value, Partial<Value>> {
  protected override _set(val: Partial<Value>) {
    for (const [k, v] of Object.entries(val)) {
      this.value[k].set(v.value)
    }
  }
  protected override _native = () => this.buffer
  protected override _setNative = (val: ArrayBuffer) => this.buffer = val
}
export class Pointer<Value extends Type<any, any>> extends Type<Deno.PointerValue, bigint> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 8, 8)
  }
  protected override _value(): bigint {
    return new BigUint64Array(this.buffer, this.offset)[0]
  }
  protected override _set(val: bigint) {
    new BigUint64Array(this.buffer, this.offset, this.byteLength).set([val])
  }
  protected override _native = () => Deno.UnsafePointer.create(this.value)
  protected override _setNative = (val: Deno.PointerValue) => this.buffer = new BigUint64Array([Deno.UnsafePointer.value(val)]).buffer
}
export const ptr = <T extends Type<any>>(type?: T) => type ? type.ptr() : new Pointer()
export class Void extends Pointer<any> {}