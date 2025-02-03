// deno-lint-ignore-file no-explicit-any no-control-regex camelcase
import { Env } from './env/index.ts'
import type { MathTrait } from './ops.ts'

// Python Map/Set implementations
export class DefaultMap<K, V> extends Map<K, V> {
  constructor(values: [K, V][] | undefined, private defaultFn: () => V) {
    super(values)
  }
  override get(key: K): V {
    let res = super.get(key)
    if (res !== undefined) return res

    res = this.defaultFn()
    this.set(key, res)
    return res
  }
}
// in JS [1] !== [1], so this is for Maps where key needs to be array
type WeakArrayKey = { key: string } | ArrayKey[]
type ArrayKey = { key: string } | ArrayKey[] | string | ConstType | undefined

export const get_key = (...args: ArrayKey[]): string => {
  const stringify = (item: ArrayKey): string => {
    if (Array.isArray(item)) return `[${item.map(stringify).join(',')}]`
    if (item === undefined) return 'undefined'
    if (typeof item === 'string') return `"${item}"`
    if (typeof item === 'bigint') return `${item}n`
    if (typeof item === 'number' || typeof item === 'boolean') return `${item}`
    if (typeof item?.key === 'string') return `${item.key}`
    throw new Error(`No stringifier for ${item}, type: ${typeof item}, args:${JSON.stringify(args)}`)
  }
  return hash(stringify(args))
}
export class ArrayMap<K extends ArrayKey, V, Internal extends [any, any] = [K, V]> {
  map: Map<string, Internal>
  constructor(values?: Internal[]) {
    this.map = new Map(values?.map(([key, value]) => [get_key(key), [key, value] as Internal]))
  }
  get size() {
    return this.map.size
  }
  get = (key: K): V | undefined => this.map.get(get_key(key))?.[1]
  set = (key: K, value: V): void => void this.map.set(get_key(key), [key, value] as Internal)
  has = (key: K): boolean => this.get(key) !== undefined
  entries = (): [K, V][] => [...this.map.values()]
  keys = () => this.entries().map((e) => e[0])
  values = () => this.entries().map((e) => e[1])
  delete = (k: K) => this.map.delete(get_key(k))
  clear = () => this.map.clear()
  setDefault = (key: K, defaultValue: V) => {
    const res = this.get(key)
    if (res !== undefined) return res
    this.set(key, defaultValue)
    return defaultValue
  }
}

// When key is garbage collected then the item is removed from the map
export class WeakKeyMap<K extends WeakArrayKey, V extends any> extends ArrayMap<K, V, [WeakRef<K>, V]> {
  private finalizationGroup = new FinalizationRegistry<string>((key) => this.map.delete(key))
  constructor(values?: [K, V][]) {
    super(values?.map(([k, v]) => [new WeakRef(k), v]))
  }
  override get = (key: K): V | undefined => {
    const res = this.map.get(get_key(key))
    if (res === undefined) return undefined
    if (res[0].deref() !== undefined) return res[1]

    // if key is gone then return undefined + delete from list
    this.map.delete(get_key(key))
    return undefined
  }
  override set = (key: K, value: V) => {
    this.map.set(get_key(key), [new WeakRef(key), value])
    this.finalizationGroup.register(key, get_key(key))
  }
  override entries = (): [K, V][] => {
    const res: [K, V][] = []
    for (const [id, [k, v]] of this.map.entries()) {
      const derefed = k.deref()
      derefed ? res.push([derefed, v]) : this.map.delete(id)
    }
    return res
  }
}

// When value is garbage collected then the item is removed from the map
export class WeakValueMap<K extends ArrayKey, V extends object> extends ArrayMap<K, V, [K, WeakRef<V>]> {
  private finalizationGroup = new FinalizationRegistry<string>((key) => this.map.delete(key))
  constructor(values?: [K, V][]) {
    super(values?.map(([k, v]) => [k, new WeakRef(v)]))
  }
  override get = (key: K): V | undefined => {
    const res = this.map.get(get_key(key))
    if (res === undefined) return undefined
    const derefed = res[1].deref()
    if (derefed !== undefined) return derefed

    // if value is gone, remove from map
    this.map.delete(get_key(key))
    return undefined
  }
  override set = (key: K, value: V) => {
    this.map.set(get_key(key), [key, new WeakRef(value)])
    this.finalizationGroup.register(value, get_key(key))
  }
  override entries = (): [K, V][] => {
    const res: [K, V][] = []
    for (const [id, [k, v]] of this.map.entries()) {
      const derefed = v.deref()
      derefed ? res.push([k, derefed]) : this.map.delete(id)
    }
    return res
  }
}

export class NotImplemented extends Error {
  constructor() {
    super('Not implemented!')
  }
}

export const max = <T extends ConstType>(v: T[]) => typeof v[0] !== 'bigint' ? Math.max(...v as number[]) : v.reduce((max, curr) => curr > max ? curr : max)
export const min = <T extends ConstType>(v: T[]) => typeof v[0] !== 'bigint' ? Math.min(...v as number[]) : v.reduce((min, curr) => curr < min ? curr : min)
export const abs = <T extends ConstType>(x: T) => typeof x !== 'bigint' ? Math.abs(Number(x)) : x < 0n ? -x : x
export const trunc = <T extends ConstType>(x: T) => typeof x !== 'bigint' ? Math.trunc(Number(x)) : x
export const sqrt = <T extends ConstType>(x: T) => typeof x !== 'bigint' ? Math.sqrt(Number(x)) : bigint_sqrt(x)
export const sin = <T extends ConstType>(x: T) => typeof x !== 'bigint' ? Math.sin(Number(x)) : bigint_sin(x)

// no idea, AI generated
export const bigint_sqrt = (v: bigint) => {
  let x = v
  let y = (x + 1n) / 2n
  while (y < x) {
    x = y
    y = (x + v / x) / 2n
  }
  return x
}
// no idea, AI generated
export const bigint_sin = (v: bigint) => {
  const PI = 3141592653589793238n
  const TWO_PI = PI * 2n

  let normalized = v % TWO_PI
  if (normalized > PI) normalized -= TWO_PI
  else if (normalized < -PI) normalized += TWO_PI

  const x = normalized
  const x2 = (x * x) / (10n ** 20n) // Scale down for precision
  const x3 = (x2 * x) / (10n ** 20n)
  const x5 = x3 * x2
  const x7 = x5 * x2
  const x9 = x7 * x2
  const x11 = x9 * x2

  return Number(x / (10n ** 10n) - x3 / 6n + x5 / 120n - x7 / 5040n + x9 / 362880n - x11 / 39916800n) / 10 ** 10
}
export const next = <A extends any>(arr: Iterator<A>, def: A): A => {
  const { value, done } = arr.next()
  return done ? def : value
}
export const int_to_bytes = (int: number) => {
  const hash = new Uint8Array(4)
  new DataView(hash.buffer).setInt32(0, int, false)
  return hash
}
export const bytes_to_bigint = (bytes: Uint8Array) => {
  let result = BigInt(0)
  for (const byte of bytes) result = (result << BigInt(8)) + BigInt(byte)
  return result
}
export const isInf = (x: number) => x === Infinity || x === -Infinity
export abstract class Enum {
  constructor(public readonly name: string, public readonly value: number) {}
  toString = () => `${this.constructor.name}.${this.name}`;
  [Symbol.for('nodejs.util.inspect.custom')](_depth: number, _options: any) {
    return `${this.constructor.name}.${this.name}`
  }

  [Symbol.toPrimitive](hint: 'string' | 'number' | 'default') {
    if (hint === 'number') return this.value
    if (hint === 'string') return this.name
    return this.value
  }
}

export const random_id = () => (Math.random() * 100000000).toFixed(0)
export function hash(input: string) {
  let hash = 0xdeadbeef
  for (let i = 0; i < input.length; i++) {
    hash = ((hash << 5) + hash) + input.charCodeAt(i)
    hash = hash & hash // Convert to 32bit integer
  }
  return (hash >>> 0).toString(16).padStart(8, '0')
}

export const string_to_bytes = (text: string) => new TextEncoder().encode(text)
export const bytes_to_string = (bytes: Uint8Array) => new TextDecoder().decode(bytes)
export const bytes_to_hex = (arr: Uint8Array) => Array.from(arr).map((byte) => byte.toString(16).padStart(2, '0')).join('')
;(BigInt.prototype as any).toJSON = function () {
  return this.toString()
}

export function product<T extends any[][], Out extends any[] = T>(...arrays: T): Out
export function product<T extends any[], Out extends any[] = T[]>(array: T, repeat?: number): Out
export function product<T extends any[][], Out extends any[] = T>(...args: [...T] | [T, number]): Out {
  let arrays: T[]
  if (args.length === 2 && typeof args[1] === 'number') {
    // Handle repeat case
    const [array, repeat] = args as any
    arrays = Array(repeat).fill(array) as T[]
  } else {
    arrays = args as T[]
  }

  if (arrays.length === 0) return [[]] as Out

  return arrays.reduce<any>((acc, arr) => {
    const result = []
    for (const a of acc) {
      for (const b of arr) result.push([...a, b])
    }
    return result
  }, [[]])
}

export const isinstance = <T extends abstract new (...args: any) => any | NumberConstructor | BooleanConstructor>(
  instance: any,
  classType: T | NumberConstructor | BooleanConstructor | ArrayConstructor | StringConstructor,
): instance is InstanceType<T> => {
  if (classType === Number && typeof instance === 'number') return true
  if (classType === Boolean && typeof instance === 'boolean') return true
  if (classType === String && typeof instance === 'string') return true
  if (classType === Array && Array.isArray(instance)) return true
  return Array.isArray(classType) ? classType.some((t) => instance instanceof t) : instance instanceof classType
}

export const divmod = (a: number, b: number) => [Math.floor(a / b), a % b] as [number, number]
export function* counter(start = 0) {
  let current = start
  while (true) yield current++
}
export const list_str = (x?: any[]): string => Array.isArray(x) ? `[${x.map(list_str).join(', ')}]` : typeof x === 'string' ? `'${x}'` : `${x}`
export const entries = <K extends string, V extends any>(object: Record<K, V>) => Object.entries(object) as [K, V][]
export const is_less_than = (a: any, b: any): boolean => {
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return a.length < b.length
    for (const [ai, bi] of zip(a, b)) if (ai !== bi) return is_less_than(ai, bi)
  }
  if (typeof a === 'object' && typeof b === 'object' && 'lt' in a && 'lt' in b) {
    return a.lt(b)
  }
  return a < b
}
export type ConstType<This = never> = number | bigint | boolean | This
export const isConst = (x: any): x is ConstType => ['number', 'bigint', 'boolean'].includes(typeof x)
export const is_eq = (one: any, two: any): boolean => {
  if (Array.isArray(one) && Array.isArray(two)) return one.length === two.length && one.every((o, i) => is_eq(o, two[i]))
  // deno-lint-ignore eqeqeq
  if (isConst(one) && isConst(two)) return one == two
  return one === two
}
export const intersection = <T>(...sets: Set<T>[]): Set<T> => sets.reduce((acc, set) => new Set([...acc].filter((item) => set.has(item))))

export function set_default<K, V>(map: Map<K, V>, key: K, defaultValue: V): V {
  if (map.has(key)) return map.get(key)!
  map.set(key, defaultValue)
  return defaultValue
}

type Iterableify<T> = { [K in keyof T]: Iterable<T[K]> }
export function zip<T extends Array<any>>(...toZip: Iterableify<T>): T[] {
  const iterators = toZip.map((i) => [...i])
  const minLength = Math.min(...iterators.map((i) => i.length))
  return range(minLength).map((i) => iterators.map((arr) => arr[i]) as T)
}

export function range(start: number, stop?: number, step = 1): number[] {
  const result: number[] = []
  if (stop === undefined) {
    stop = start
    start = 0
  }
  if (!Number.isFinite(start) || !Number.isFinite(stop)) throw new Error(`Range should be finite`)
  if (step > 0) { for (let i = start; i < stop; i += step) result.push(i) }
  else if (step < 0) { for (let i = start; i > stop; i += step) result.push(i) }
  return result
}
export const tuple = <T extends any[]>(...t: T) => t
export const assert = (condition: boolean): condition is true => {
  if (!condition) throw new Error()
  return condition
}
export function permutations<T>(arr: T[], length: number = arr.length): T[][] {
  if (length === 1) return arr.map((item) => [item])

  const result: T[][] = []
  arr.forEach((item, i) => {
    const remaining = arr.slice(0, i).concat(arr.slice(i + 1))
    permutations(remaining, length - 1).forEach((perm) => result.push([item, ...perm]))
  })
  return result
}

export function is_subset<T>(main: Set<T>, subset: Set<T>): boolean {
  for (const elem of subset) if (!main.has(elem)) return false
  return true
}

export function math_gcd(...numbers: number[]): number {
  function gcdTwo(a: number, b: number): number {
    while (b !== 0) {
      const temp = b
      b = a % b
      a = temp
    }
    return Math.abs(a)
  }
  if (numbers.length === 0) throw new Error('At least one number must be provided')
  return numbers.reduce((acc, num) => gcdTwo(acc, num))
}
// TINYGRAD CODE
// NOTE: it returns int 1 if x is empty regardless of the type of x
export const OSX = Env.PLATFORM === 'darwin'
export const CI = !!Env.env.get('CI')

if (Env.PLATFORM === 'win32') Env.writeStdout(string_to_bytes(''))

// TODO: probably should just filter out duplicates + use isEq
export const dedup = <T>(x: T[]): T[] => [...new Set(x)] // retains list order

export const argsort = <T>(x: T[]) => range(x.length).sort((a, b) => x[a] < x[b] ? -1 : x[a] > x[b] ? 1 : 0)
export const all_same = <T>(items: T[]) => items.every((x) => is_eq(x, items[0]))
export const isInt = (x: any): x is number => Number.isInteger(x)
export const all_int = (t: any[]): t is number[] => t.every((s) => Number.isInteger(s))
export const colored = (st: string, color?: string, background = false) => {
  if (!color) return st
  const colors = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
  const code = 10 * (background ? 1 : 0) + 60 * (color === color.toUpperCase() ? 1 : 0) + 30 + colors.indexOf(color.toLowerCase())
  return `\u001b[${code}m${st}\u001b[0m`
}
export const colorize_float = (x: number) => colored(x.toFixed(2).padStart(7) + 'x', x < 0.75 ? 'green' : x > 1.15 ? 'red' : 'yellow')
export const memsize_to_str = (_b: number) => [tuple(1e9, 'GB'), tuple(1e6, 'MB'), tuple(1e3, 'KB'), tuple(1, 'B')].filter(([d]) => _b > d).map(([d, pr]) => `${(_b / d).toFixed(2)} ${pr}`)[0]
export const ansistrip = (s: string) => s.replace(/\x1b\[(K|.*?m)/g, '')
export const ansilen = (s: string) => ansistrip(s).length
export const make_tuple = (x: number | number[], cnt: number): number[] => Array.isArray(x) ? [...x] : Array(cnt).fill(x)
export const flatten = <T>(l: T[][]): T[] => l.flat()
export const fully_flatten = (l: any): any[] => {
  if (Array.isArray(l) || (l && typeof l === 'object' && 'length' in l && !('length' in String.prototype))) {
    const flattened: any[] = []
    if ('shape' in l && l.shape?.length === 0) {
      flattened.push(l[0])
    } else {
      for (let i = 0; i < l.length; i++) {
        flattened.push(...fully_flatten(l[i]) as any)
      }
    }
    return flattened
  }
  return [l]
}

// def fromimport(mod, frm): return getattr(__import__(mod, fromlist=[frm]), frm)
export const strip_parens = (s: string) => s[0] === '(' && s[s.length - 1] === ')' && s.slice(1, -1).indexOf('(') <= s.slice(1, -1).indexOf(')') ? s.slice(1, -1) : s
export const round_up = (num: number, amt: number) => Math.ceil(num / amt) * amt
export const lo32 = (x: any) => Number(BigInt(x) & 0xFFFFFFFFn) // Any is sint
export const hi32 = (x: any) => Number(BigInt(x) >> 32n) // any is sint
export const data64 = (data: number): [number, number] => [Math.floor(data / Math.pow(2, 32)), data >>> 0] // TODO:make work with sint
export const data64Le = (data: number): [number, number] => [data >>> 0, Math.floor(data / Math.pow(2, 32))] // TODO:make work with sint
export const getbits = (value: number, start: number, end: number) => (value >> start) & ((1 << end - start + 1) - 1)
export const i2u = (bits: number, value: number) => value >= 0 ? value : (1 << bits) + value
export const merge_dicts = <T extends string, U = any>(ds: Record<T, U>[]): Record<T, U> => {
  const kvs = new Set(ds.flatMap((d) => Object.entries(d))) as Set<[T, U]>
  const keys = new Set(Array.from(kvs).map((kv) => kv[0]))
  if (kvs.size !== keys.size) throw new Error(`cannot merge, ${Array.from(kvs)} contains different values for the same key`)
  return Object.fromEntries(Array.from(kvs)) as Record<T, U>
}
export function merge_maps<K, V>(maps: Iterable<Map<K, V>>): Map<K, V> {
  const resultMap = new Map<K, V>()
  for (const map of maps) {
    if (!(map instanceof Map)) continue
    for (const [key, value] of map.entries()) {
      if (resultMap.has(key) && resultMap.get(key) !== value) throw new Error(`Cannot merge, key "${key}" has conflicting values.`)
      resultMap.set(key, value)
    }
  }

  return resultMap
}
export function merge_sets<V>(sets: Iterable<Set<V>>): Set<V> {
  const resultSet = new Set<V>()
  for (const set of sets) for (const value of set) resultSet.add(value)
  return resultSet
}

export const partition = <T>(itr: T[], fn: (x: T) => boolean): [T[], T[]] => itr.reduce(([a, b], s) => fn(s) ? [[...a, s], b] : [a, [...b, s]], [[], []] as [T[], T[]])
export const get_single_element = <T>(x: T[]): T => {
  if (x.length !== 1) throw new Error(`list ${x} must only have 1 element`)
  return x[0]
}
export const unwrap = <T>(x: T | undefined): T => x!
export const getChild = (obj: any, key: string): any => key.split('.').reduce((current, k) => !isNaN(Number(k)) ? current[Number(k)] : current[k], obj)

export const word_wrap = (x: string, wrap = 80): string => x.length <= wrap || x.slice(0, wrap).includes('\n') ? x : x.slice(0, wrap) + '\n' + word_wrap(x.slice(wrap), wrap)
export const to_function_name = (s: string) => ansistrip(s).split('').map((c) => /[A-Za-z0-9_]/.test(c) ? c : c.charCodeAt(0).toString(16).toUpperCase().padStart(2, '0')).join('')
export const get_env = (key: string, defaultVal = '') => Env.env.get(key) || defaultVal
export const get_number_env = (key: string, defaultVal?: number) => Number(Env.env.get(key) || defaultVal)
export const temp = (x?: string): string => `${Env.tmpdir()}/${x || random_id()}`

// TODO JIT should be automatic
export const DEBUG = get_number_env('DEBUG', 0), IMAGE = get_number_env('IMAGE', 0), BEAM = get_number_env('BEAM', 0), NOOPT = get_number_env('NOOPT', 0), JIT = get_number_env('JIT', 1)
export const WINO = get_number_env('WINO', 0), CAPTURING = get_number_env('CAPTURING', 1), TRACEMETA = get_number_env('TRACEMETA', 1)
export const USE_TC = get_number_env('TC', 1), TC_OPT = get_number_env('TC_OPT', 0), AMX = get_number_env('AMX', 0), TRANSCENDENTAL = get_number_env('TRANSCENDENTAL', 1)
export const FUSE_ARANGE = get_number_env('FUSE_ARANGE', 0), FUSE_CONV_BW = get_number_env('FUSE_CONV_BW', 0)
export const SPLIT_REDUCEOP = get_number_env('SPLIT_REDUCEOP', 1), NO_MEMORY_PLANNER = get_number_env('NO_MEMORY_PLANNER', 0), RING = get_number_env('RING', 1)
export const PICKLE_BUFFERS = get_number_env('PICKLE_BUFFERS', 1), PROFILE = get_env('PROFILE', get_env('VIZ')), LRU = get_number_env('LRU', 1)
export const CACHELEVEL = get_number_env('CACHELEVEL', 2)

export class Metadata {
  key: string
  static cache = new WeakValueMap<string, Metadata>()
  constructor(public name: string, public caller: string, public backward = false) {
    this.key = get_key(name, caller, backward)
    return Metadata.cache.setDefault(this.key, this)
  }
  // def __hash__(self): return hash(self.name)
  //   def __repr__(self): return str(self) + (f" - {self.caller}" if self.caller else "")
  //   def __str__(self): return self.name + (" bw" if self.backward else "")
}

class ContextVar<T> {
  static _cache: Record<string, ContextVar<any>> = {}
  initialValue!: string
  key!: string
  value!: T
  tokens: Record<string, T> = {}
  constructor(key: string, value: T) {
    const cached = ContextVar._cache[key]
    if (cached) return cached
    this.key = key
    this.value = value
  }
  get = () => this.value
  set = (v: T): string => {
    const token = random_id()
    this.value = v
    this.tokens[token] = v
    return token
  }
  reset = (token: string) => {
    this.value = this.tokens[token]
  }
}
export const _METADATA = new ContextVar<Metadata | undefined>('_METADATA', undefined)
// **************** global state Counters ****************

export class GlobalCounters {
  static global_ops = 0
  static global_mem = 0
  static time_sum_s = 0
  static kernel_count = 0
  static mem_used = 0 // NOTE: this is not reset
  static reset = () => [GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.time_sum_s, GlobalCounters.kernel_count] = [0, 0, 0, 0]
}

// **************** timer and profiler ****************

export class Timing {
  //   def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled
  //   def __enter__(self): self.st = time.perf_counter_ns()
  //   def __exit__(self, *exc):
  //     self.et = time.perf_counter_ns() - self.st
  //     if self.enabled: print(f"{self.prefix}{self.et*1e-6:6.2f} ms"+(self.on_exit(self.et) if self.on_exit else ""))
}
export const _format_fcn = (fcn: any[]) => `${fcn[0]}:${fcn[1]}:${fcn[2]}`
export class Profiling {
  //   def __init__(self, enabled=True, sort='cumtime', frac=0.2, fn=None, ts=1):
  //     self.enabled, self.sort, self.frac, self.fn, self.time_scale = enabled, sort, frac, fn, 1e3/ts
  //   def __enter__(self):
  //     import cProfile
  //     self.pr = cProfile.Profile()
  //     if self.enabled: self.pr.enable()
  //   def __exit__(self, *exc):
  //     if self.enabled:
  //       self.pr.disable()
  //       if self.fn: self.pr.dump_stats(self.fn)
  //       import pstats
  //       stats = pstats.Stats(self.pr).strip_dirs().sort_stats(self.sort)
  //       for fcn in stats.fcn_list[0:int(len(stats.fcn_list)*self.frac)]:    # type: ignore[attr-defined]
  //         (_primitive_calls, num_calls, tottime, cumtime, callers) = stats.stats[fcn]    # type: ignore[attr-defined]
  //         scallers = sorted(callers.items(), key=lambda x: -x[1][2])
  //         print(f"n:{num_calls:8d}  tm:{tottime*self.time_scale:7.2f}ms  tot:{cumtime*self.time_scale:7.2f}ms",
  //               colored(_format_fcn(fcn).ljust(50), "yellow"),
  //               colored(f"<- {(scallers[0][1][2]/tottime)*100:3.0f}% {_format_fcn(scallers[0][0])}", "BLACK") if scallers else '')
}
// *** universal database cache ***

const cache_dir = get_env('XDG_CACHE_HOME', OSX ? `${Env.homedir()}/Library/Caches` : `${Env.homedir()}/.cache`)
export const CACHEDB = get_env('CACHEDB', `${cache_dir}/tinygrad/cache.db`)

// VERSION = 16
const _db_connection = undefined
export const db_connection = () => {
  //   global _db_connection
  //   if _db_connection is None:
  //     os.makedirs(CACHEDB.rsplit(os.sep, 1)[0], exist_ok=True)
  //     _db_connection = sqlite3.connect(CACHEDB, timeout=60, isolation_level="IMMEDIATE")
  // another connection has set it already or is in the process of setting it
  // that connection will lock the database
  //     with contextlib.suppress(sqlite3.OperationalError): _db_connection.execute("PRAGMA journal_mode=WAL").fetchone()
  //     if DEBUG >= 7: _db_connection.set_trace_callback(print)
  //   return _db_connection
}
export const diskcache_clear = () => {
  //   cur = db_connection().cursor()
  //   drop_tables = cur.execute("SELECT 'DROP TABLE IF EXISTS ' || quote(name) || ';' FROM sqlite_master WHERE type = 'table';").fetchall()
  //   cur.executescript("\n".join([s[0] for s in drop_tables] + ["VACUUM;"]))
}
export const diskcache_get = (table: string, key: any): any | undefined => {
  //   if CACHELEVEL == 0: return None
  //   if isinstance(key, (str,int)): key = {"key": key}
  //   conn = db_connection()
  //   cur = conn.cursor()
  //   try:
  //     res = cur.execute(f"SELECT val FROM '{table}_{VERSION}' WHERE {' AND '.join([f'{x}=?' for x in key.keys()])}", tuple(key.values()))
  //   except sqlite3.OperationalError:
  //     return None  # table doesn't exist
  //   if (val:=res.fetchone()) is not None: return pickle.loads(val[0])
  return undefined
}
// _db_tables = set()
export const diskcache_put = (table: string, key: any, val: any, prepickled = false) => {
  //   if CACHELEVEL == 0: return val
  //   if isinstance(key, (str,int)): key = {"key": key}
  //   conn = db_connection()
  //   cur = conn.cursor()
  //   if table not in _db_tables:
  //     TYPES = {str: "text", bool: "integer", int: "integer", float: "numeric", bytes: "blob"}
  //     ltypes = ', '.join(f"{k} {TYPES[type(key[k])]}" for k in key.keys())
  //     cur.execute(f"CREATE TABLE IF NOT EXISTS '{table}_{VERSION}' ({ltypes}, val blob, PRIMARY KEY ({', '.join(key.keys())}))")
  //     _db_tables.add(table)
  //   cur.execute(f"REPLACE INTO '{table}_{VERSION}' ({', '.join(key.keys())}, val) VALUES ({', '.join(['?']*len(key.keys()))}, ?)", tuple(key.values()) + (pickle.dumps(val), ))  # noqa: E501
  //   conn.commit()
  //   cur.close()
  //   return val
}
export const diskcache = (func: any) => {
  //   def wrapper(*args, **kwargs) -> bytes:
  //     table, key = f"cache_{func.__name__}", hashlib.sha256(pickle.dumps((args, kwargs))).hexdigest()
  //     if (ret:=diskcache_get(table, key)): return ret
  //     return diskcache_put(table, key, func(*args, **kwargs))
  //   return wrapper
}
// *** http support ***
export const CAPTURE_PROCESS_REPLAY = get_env('RUN_PROCESS_REPLAY') || get_env('CAPTURE_PROCESS_REPLAY')

export const _ensure_downloads_dir = (): string => {
  throw new NotImplemented()
  // if we are on a tinybox, use the raid array
  //   if pathlib.Path("/etc/tinybox-release").is_file():
  // try creating dir with sudo
  //     if not (downloads_dir := pathlib.Path("/raid/downloads")).exists():
  //       subprocess.run(["sudo", "mkdir", "-p", downloads_dir], check=True)
  //       subprocess.run(["sudo", "chown", "tiny:root", downloads_dir], check=True)
  //       subprocess.run(["sudo", "chmod", "775", downloads_dir], check=True)
  //     return downloads_dir
  //   return pathlib.Path(_cache_dir) / "tinygrad" / "downloads"
}
// def fetch(url:str, name:Optional[Union[pathlib.Path, str]]=None, subdir:Optional[str]=None, gunzip:bool=False,
//           allow_caching=not getenv("DISABLE_HTTP_CACHE")) -> pathlib.Path:
//   if url.startswith(("/", ".")): return pathlib.Path(url)
//   if name is not None and (isinstance(name, pathlib.Path) or '/' in name): fp = pathlib.Path(name)
//   else:
//     fp = _ensure_downloads_dir() / (subdir or "") / \
//       ((name or hashlib.md5(url.encode('utf-8')).hexdigest()) + (".gunzip" if gunzip else ""))
//   if not fp.is_file() or not allow_caching:
//     with urllib.request.urlopen(url, timeout=10) as r:
//       assert r.status == 200
//       length = int(r.headers.get('content-length', 0)) if not gunzip else None
//       progress_bar = tqdm(total=length, unit='B', unit_scale=True, desc=f"{url}", disable=CI)
//       (path := fp.parent).mkdir(parents=True, exist_ok=True)
//       readfile = gzip.GzipFile(fileobj=r) if gunzip else r
//       with tempfile.NamedTemporaryFile(dir=path, delete=False) as f:
//         while chunk := readfile.read(16384): progress_bar.update(f.write(chunk))
//         f.close()
//         progress_bar.update(close=True)
//         if length and (file_size:=os.stat(f.name).st_size) < length: raise RuntimeError(f"fetch size incomplete, {file_size} < {length}")
//         pathlib.Path(f.name).rename(fp)
//   return fp

// *** Exec helpers

export const cpu_time_execution = <Args extends any[]>(fn: (...args: Args) => Promise<void>) => {
  return async (...args: Args) => {
    const st = performance.now()
    await fn(...args)
    return performance.now() - st
  }
}

export const cpu_objdump = (lib: Uint8Array, objdumpTool = 'objdump') => {
  const outputFile = temp('temp_output.so')
  Env.writeFileSync(outputFile, lib)
  try {
    const output = Env.execSync(objdumpTool, { args: ['-d', outputFile] })
    console.log(output)
  } finally {
    Env.removeSync(outputFile)
  }
}

export const capstone_flatdump = (lib: Uint8Array) => {
  throw new NotImplemented()
  // import capstone
  // match platform.machine():
  //   case 'x86_64': cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
  //   case 'aarch64' | 'arm64': cs = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
  //   case machine: raise NotImplementedError(f"Capstone disassembly isn't supported for {machine}")
  // for instr in cs.disasm(lib, 0):
  //   print(f"{instr.address:#08x}: {instr.mnemonic}\t{instr.op_str}")
}
export function replace<T>(instance: T, replacement: Partial<T>): T {
  const newInstance = Object.create(Object.getPrototypeOf(instance))
  Object.assign(newInstance, instance, replacement)
  return newInstance
}
// *** universal support for code object pickling

// def _reconstruct_code(*args): return types.CodeType(*args)
// def _serialize_code(code:types.CodeType):
//   args = inspect.signature(types.CodeType).parameters  # NOTE: this works in Python 3.10 and up
//   return _reconstruct_code, tuple(code.__getattribute__('co_'+x.replace('codestr', 'code').replace('constants', 'consts')) for x in args)
// copyreg.pickle(types.CodeType, _serialize_code)

// def _serialize_module(module:types.ModuleType): return importlib.import_module, (module.__name__,)
// copyreg.pickle(types.ModuleType, _serialize_module)

export type Slice = { start?: number; stop?: number; step?: number }
/**
 * A Python-like slice function for arrays in TypeScript.
 * @param arr - The array you want to slice
 * @param options - An object containing start, stop, and step
 *   - start?: number (defaults to 0 if step > 0, or last index if step < 0)
 *   - stop?: number (defaults to arr.length if step > 0, or -arr.length - 1 if step < 0)
 *   - step?: number (defaults to 1)
 * @returns A new array sliced according to the options
 */
export function slice<T>(arr: T[], { start, stop, step }: Slice = {}): T[] {
  const len = arr.length

  // 1) Default step is 1 if not provided
  step ??= 1

  // 2) Zero step is invalid
  if (step === 0) {
    throw new Error('slice step cannot be 0')
  }

  // 3) Derive default "start" and "stop" based on sign of step
  if (step > 0) {
    // Forward slicing
    if (start === undefined) start = 0
    if (stop === undefined) stop = len // up to the end
  } else {
    // Negative step => backward slicing
    if (start === undefined) start = len - 1
    if (stop === undefined) {
      /**
       * In Python, if stop is omitted with a negative step (e.g. arr[-2::-2]),
       * it effectively means "keep going until index < 0" in normal terms,
       * so we can pick a 'virtual' negative stop that becomes -1 after conversion.
       *
       * Example: If array length = 4, then -4 - 1 = -5
       * later we do (arr.length + -5) = -1
       */
      stop = -len - 1
    }
  }

  // 4) Convert negative start/stop to positive indices
  //    i.e. -1 => arr.length - 1
  if (start < 0) start = len + start
  if (stop < 0) stop = len + stop

  const result: T[] = []

  // 5) Collect values
  if (step > 0) {
    // forward
    for (let i = start; i < stop && i < len; i += step) {
      // i >= 0 check can be omitted if you want to allow e.g. i < 0
      if (i >= 0) {
        result.push(arr[i])
      }
    }
  } else {
    // backward
    for (let i = start; i > stop && i >= 0; i += step) {
      // i < len check can be omitted if you want to be safe,
      // but typically i won't exceed len in normal usage
      if (i < len) {
        result.push(arr[i])
      }
    }
  }

  return result
}

// DECORATORS

export function cache<This extends object, Args extends any[], Return>(
  target: (this: This, ...args: Args) => Return,
  ctx: ClassGetterDecoratorContext<This, Return> | ClassMethodDecoratorContext<This, (this: This, ...args: Args) => Return>,
) {
  const instanceCaches = new WeakMap<This, Record<string, Return>>()
  const staticCache: Record<string, Return> = {}
  return function (this: This, ...args: Args): Return {
    const key = get_key(String(ctx.name), args)
    let cache = ctx.static ? staticCache : (instanceCaches.get(this) || instanceCaches.set(this, {}).get(this)!)
    if (key in cache) return cache[key]
    const res = target.call(this, ...args)
    cache[key] = res
    return res
  }
}
export function cache_fn<Args extends any[], Return>(fn: (...args: Args) => Return) {
  const cache: Record<string, Return> = {}
  return (...args: Args) => {
    const key = get_key(args)
    if (key in cache) return cache[key]
    const res = fn(...args)
    cache[key] = res
    return res
  }
}

export function debug<Args extends any[], Return>(target: (...args: Args) => Return, _context: ClassMemberDecoratorContext) {
  const name = String(_context.name)
  return (...args: Args): Return => {
    console.log(`'${name}' called: ${args}`)
    const start = performance.now()
    const res = target(...args)
    console.log(`'${name}' returned in ${performance.now() - start}ms: ${res}`)
    return res
  }
}

export const measure_time = async <T>(name: string, fn: () => Promise<T> | T) => {
  const s = performance.now()
  const res = await fn()
  console.log(`${name} took ${Math.round(performance.now() - s) / 1000}s and returned ${res}`)
  return res
}

export type Math = ConstType | MathTrait<any>
type Return<A, B> = A extends MathTrait<any> ? A : B extends MathTrait<any> ? B : A extends bigint ? bigint : B extends bigint ? bigint : number
const _meta = (mathFn: (a: MathTrait<MathTrait<any>>, b: Math, reverse: boolean) => MathTrait<any>, numberFn: (a: number, b: number) => number, bigintFn?: (a: bigint, b: bigint) => bigint) => {
  if (!bigintFn) bigintFn = numberFn as unknown as (a: bigint, b: bigint) => bigint
  return <A extends Math, B extends Math>(a: A, b: B): Return<A, B> => {
    if (typeof a === 'string') throw new Error(`a is string, a=${a}`)
    if (typeof b === 'string') throw new Error(`b is string, b=${b}`)
    if (!isConst(a)) return mathFn(a, b, false) as Return<A, B>
    else if (!isConst(b)) return mathFn(b, a, true) as Return<A, B>
    else if (typeof a === 'bigint' || typeof b === 'bigint') return bigintFn(BigInt(a), BigInt(b)) as Return<A, B>
    else return numberFn(Number(a), Number(b)) as Return<A, B>
  }
}

export const add = _meta((a, b, r) => a.add(b, r), (a, b) => a + b)
export const sub = _meta((a, b, r) => a.sub(b, r), (a, b) => a - b)
export const mul = _meta((a, b, r) => a.mul(b, r), (a, b) => a * b)
export const div = _meta((a, b, r) => a.div(b, r), (a, b) => a / b)
export const idiv = _meta((a, b, r) => a.idiv(b, r), (a, b) => Math.floor(a / b), (a, b) => a / b)
export const neg = <A extends Math>(a: A): Return<A, A> => ((!isConst(a)) ? a.neg() : typeof a === 'bigint' ? a * -1n : Number(a) * -1)
export const mod = _meta((a, b, r) => a.mod(b, r), (a, b) => a % b)

export const and = _meta((a, b, r) => a.bitwise_and(b, r), (a, b) => a & b)
export const or = _meta((a, b, r) => a.bitwise_or(b, r), (a, b) => a | b)
export const xor = _meta((a, b, r) => a.xor(b, r), (a, b) => a ^ b)
export const lshift = _meta((a, b, r) => a.lshift(b, r), (a, b) => a << b)
export const rshift = _meta((a, b, r) => a.rshift(b, r), (a, b) => a >> b)

export const lt = _meta((a, b, r) => !r ? a.lt(b) : a.const_like(b as any).lt(a), (a, b) => Number(a < b), (a, b) => BigInt(a < b))
export const gt = _meta((a, b, r) => !r ? a.gt(b) : a.const_like(b as any).gt(a), (a, b) => Number(a > b), (a, b) => BigInt(a > b))
export const le = _meta((a, b, r) => !r ? a.le(b) : a.const_like(b as any).le(a), (a, b) => Number(a <= b), (a, b) => BigInt(a <= b))
export const ge = _meta((a, b, r) => !r ? a.ge(b) : a.const_like(b as any).ge(a), (a, b) => Number(a >= b), (a, b) => BigInt(a >= b))
export const ne = _meta((a, b, r) => !r ? a.ne(b) : a.const_like(b as any).ne(a), (a, b) => Number(a !== b), (a, b) => BigInt(a !== b))
export const eq = _meta((a, b, r) => !r ? a.eq(b) : a.const_like(b as any).eq(a), (a, b) => Number(a === b), (a, b) => BigInt(a === b))

export const polyN = <T extends Math>(x: T, p: Math[]): T => p.reduce((acc, c) => add(mul(acc, x), c), 0) as T
export const prod = <T extends Math>(x: T[]): T => x.reduce((acc, curr) => mul(acc, curr) as T, 1 as T)
export const sum = <T extends Math>(x: T[]): T => x.reduce((acc, curr) => add(acc, curr) as T, 0 as T)
export const ceildiv = <A extends Math, B extends Math>(num: A, amt: B): Return<A, B> => neg(idiv(num, neg(amt))) as Return<A, B>
export const pow = _meta((a, b, r) => (a as any).pow(b, r), (a, b) => a ** b)

export function* pairwise<T>(iterable: Iterable<T>): Generator<[T, T], void, unknown> {
  let previous: T | undefined
  let hasPrevious = false

  for (const item of iterable) {
    if (!hasPrevious) {
      previous = item
      hasPrevious = true
    } else {
      yield [previous as T, item]
      previous = item
    }
  }
}

export function* accumulate<T>(iterable: Iterable<T>, func?: (acc: T, val: T) => T, initialValue?: T): Generator<T, void, unknown> {
  const operation = func ?? ((a, b) => (a as unknown as number) + (b as unknown as number)) as (acc: T, val: T) => T
  let accumulator: T
  let started = false

  for (const value of iterable) {
    if (!started) {
      // If an initialValue is provided, set the accumulator to that,
      // then apply the first item. Otherwise, use the first item as the initial accumulator.
      if (initialValue !== undefined) {
        accumulator = operation(initialValue, value)
      } else {
        accumulator = value
      }
      started = true
      yield accumulator
    } else {
      accumulator = operation(accumulator!, value)
      yield accumulator
    }
  }
}
