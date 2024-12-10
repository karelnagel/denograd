// deno-lint-ignore-file no-explicit-any no-control-regex camelcase
import path from 'node:path'
import process from 'node:process'
import os from 'node:os'
import { unlinkSync, writeFileSync } from 'node:fs'
import { execSync } from 'node:child_process'

// GENERAL HELPERS
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

export const len = (x: { length: number } | null | undefined): number => isNone(x) ? 0 : x.length

export const divmod = (a: number, b: number) => [Math.floor(a / b), a % b] as [number, number]
export function* counter(start = 0) {
  let current = start
  while (true) yield current++
}
export const listStr = (x?: null | any[]): string => Array.isArray(x) ? `[${x.map((x) => Array.isArray(x) ? listStr(x) : x).join(', ')}]` : `${x}`
export const entries = <K extends string, V extends any>(object: Record<K, V>) => Object.entries(object) as [K, V][]
export const isLessThan = (a: any, b: any): boolean => {
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return a.length < b.length
    for (const [ai, bi] of zip(a, b)) if (ai !== bi) return isLessThan(ai, bi)
  }
  if (typeof a === 'object' && typeof b === 'object' && 'lt' in a && 'lt' in b) {
    return a.lt(b)
  }
  return a < b
}
export const isEq = (one: any, two: any): boolean => {
  if (Array.isArray(one) && Array.isArray(two)) return one.length === two.length && one.every((o, i) => isEq(o, two[i]))
  // if (typeof one === 'object' && typeof two === 'object') return JSON.stringify(one) === JSON.stringify(two)//should not be needed after having cahces for classes
  return one == two // this uses == instead of ===, cause in python 0==False, but in js 0!==false, but 0==false
}
export const intersection = <T>(...sets: Set<T>[]): Set<T> => sets.reduce((acc, set) => new Set([...acc].filter((item) => set.has(item))))

type LT = { lt: (x: any) => boolean }
type ADD = { add: (x: any) => ADD }

const defaultSum = <T extends ADD | number>(a: T, b: T): T => (typeof a !== 'number' ? a.add(b) : typeof b !== 'number' ? b.add(a) : a + b) as T
export const sum = <T extends ADD | number>(items: T[], fn = (a: T, b: T) => defaultSum(a, b)): T => items.reduce((acc, x) => fn(acc, x))

const defaultLt = <T extends LT | number>(a: T, b: T) => (typeof a !== 'number' ? a.lt(b) : typeof b !== 'number' ? b.lt(a) : a < b)

export const sorted = <T extends LT | number>(items: T[], reverse = false, fn = (a: T, b: T) => defaultLt(a, b)) => !reverse ? items.toSorted((a, b) => fn(a, b) ? -1 : 1) : items.toSorted((a, b) => fn(a, b) ? 1 : -1)
export const max = <T extends LT | number>(items: T[], fn = (a: T, b: T) => defaultLt(a, b)) => sorted(items, true, fn)[0]
export const min = <T extends LT | number>(items: T[], fn = (a: T, b: T) => defaultLt(a, b)) => sorted(items, false, fn)[0]

export const all = (items: boolean[]) => items.every((x) => x)
export const any = (items: boolean[]) => items.some((x) => x)

export function setDefault<K, V>(map: Map<K, V>, key: K, defaultValue: V): V {
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

export const isNone = <T>(x: T | null | undefined): x is null | undefined => x === undefined || x === null
export const isNotNone = <T>(x: T | null | undefined): x is T => x !== undefined && x !== null

export const setMap = <K, V>(map: Map<K, V>, key: K, fn: (x: V) => V) => {
  const newVal = fn(map.get(key)!)
  map.set(key, newVal)
  return newVal
}
export function range(start: number, stop?: number, step = 1): number[] {
  const result: number[] = []
  if (stop === undefined) {
    stop = start
    start = 0
  }
  if (step > 0) { for (let i = start; i < stop; i += step) result.push(i) }
  else if (step < 0) { for (let i = start; i > stop; i += step) result.push(i) }
  return result
}
export const d = <T extends any[]>(...t: T) => t
export const assert = (condition: boolean, message?: string): condition is true => {
  if (!condition) throw new Error(message)
  return condition
}
export const raise = (msg?: string) => {
  throw new Error(msg)
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
export const resolvePromise = <T>(promise: Promise<T>): T => {
  let result
  promise.then((x) => result = x)
  while (result === undefined) {
    // Wait for promise to resolve
  }
  return result
}

export function isSubset<T>(main: Set<T>, subset: Set<T>): boolean {
  for (const elem of subset) if (!main.has(elem)) return false
  return true
}

export function mathGcd(...numbers: number[]): number {
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
export const prod = (x: number[]) => x.reduce((acc, curr) => acc * curr, 1)

export const OSX = process.platform === 'darwin'
export const CI = !!process.env.CI

if (process.platform === 'win32') process.stdout.write('')

export const dedup = <T>(x: T[]): T[] => [...new Set(x)] // retains list order
export const argfix = (...x: any[]) => {
  if (x.length && (Array.isArray(x[0]))) {
    if (x.length !== 1) throw new Error(`bad arg ${x}`)
    return [...x[0]]
  }
  return x
}

export const argsort = <T>(x: T[]) => range(x.length).sort((a, b) => x[a] < x[b] ? -1 : x[a] > x[b] ? 1 : 0)
export const all_same = <T>(items: T[]) => items.every((x) => isEq(x, items[0]))
export const isInt = (x: any): x is number => Number.isInteger(x)
export const all_int = (t: any[]): t is number[] => t.every((s) => Number.isInteger(s))
export const colored = (st: string, color?: string, background = false) => {
  if (!color) return st
  const colors = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
  const code = 10 * (background ? 1 : 0) + 60 * (color === color.toUpperCase() ? 1 : 0) + 30 + colors.indexOf(color.toLowerCase())
  return `\u001b[${code}m${st}\u001b[0m`
}
export const colorizeFloat = (x: number) => colored(x.toFixed(2).padStart(7) + 'x', x < 0.75 ? 'green' : x > 1.15 ? 'red' : 'yellow')
export const memsizeToStr = (_b: number) => [d(1e9, 'GB'), d(1e6, 'MB'), d(1e3, 'KB'), d(1, 'B')].filter(([d]) => _b > d).map(([d, pr]) => `${(_b / d).toFixed(2)} ${pr}`)[0]
export const ansistrip = (s: string) => s.replace(/\x1b\[(K|.*?m)/g, '')
export const ansilen = (s: string) => ansistrip(s).length
export const makeTuple = (x: number | number[], cnt: number): number[] => Array.isArray(x) ? [...x] : Array(cnt).fill(x)
export const flatten = <T>(l: T[][]): T[] => l.flat()
export const fullyFlatten = <T>(l: any): T[] => {
  if (Array.isArray(l) || (l && typeof l === 'object' && 'length' in l && !('length' in String.prototype))) {
    const flattened: T[] = []
    if ('shape' in l && l.shape?.length === 0) {
      flattened.push(l[0])
    } else {
      for (let i = 0; i < l.length; i++) {
        flattened.push(...fullyFlatten(l[i]) as any)
      }
    }
    return flattened
  }
  return [l]
}

// TODO
// def fromimport(mod, frm): return getattr(__import__(mod, fromlist=[frm]), frm)
export const stripParens = (s: string) => s[0] === '(' && s[s.length - 1] === ')' && s.slice(1, -1).indexOf('(') <= s.slice(1, -1).indexOf(')') ? s.slice(1, -1) : s
export const ceildiv = (num: number, amt: number): number => {
  const ret = -(Math.floor(-num / amt))
  return Number.isInteger(ret) ? ret : Math.floor(ret)
}
export const round_up = (num: number, amt: number) => Math.ceil(num / amt) * amt
export const data64 = (data: number): [number, number] => [Math.floor(data / Math.pow(2, 32)), data >>> 0] // TODO:make work with sint
export const data64Le = (data: number): [number, number] => [data >>> 0, Math.floor(data / Math.pow(2, 32))] // TODO:make work with sint
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
export const unwrap = <T>(x: T | undefined): T => x!
export const getChild = (obj: any, key: string): any => key.split('.').reduce((current, k) => !isNaN(Number(k)) ? current[Number(k)] : current[k], obj)

export const wordWrap = (x: string, wrap = 80): string => x.length <= wrap || x.slice(0, wrap).includes('\n') ? x : x.slice(0, wrap) + '\n' + wordWrap(x.slice(wrap), wrap)
export const polyN = (x: number, p: number[]): number => p.reduce((acc, c) => acc * x + c, 0)
export const to_function_name = (s: string): string => s.split('').map((c) => (c.match(/[a-zA-Z0-9_]/) ? c : c.charCodeAt(0).toString(16))).join('')
export const getEnv = (key: string, defaultVal = '') => process.env[key] || defaultVal
export const getNumberEnv = (key: string, defaultVal?: number) => Number(process.env[key] || defaultVal)
export const temp = (x: string): string => path.join(os.tmpdir(), x)

export const [DEBUG, IMAGE, BEAM, NOOPT, JIT] = [getNumberEnv('DEBUG', 0), getNumberEnv('IMAGE', 0), getNumberEnv('BEAM', 0), getNumberEnv('NOOPT', 0), getNumberEnv('JIT', 1)]
export const [WINO, CAPTURING, TRACEMETA] = [getNumberEnv('WINO', 0), getNumberEnv('CAPTURING', 1), getNumberEnv('TRACEMETA', 1)]
export const [PROFILE, PROFILEPATH] = [getNumberEnv('PROFILE', 0), getEnv('PROFILEPATH', temp('tinygrad_profile.json'))]
export const [USE_TC, TC_OPT, AMX, TRANSCENDENTAL] = [getNumberEnv('TC', 1), getNumberEnv('TC_OPT', 0), getNumberEnv('AMX', 0), getNumberEnv('TRANSCENDENTAL', 1)]
export const [FUSE_ARANGE, FUSE_CONV_BW, LAZYCACHE] = [getNumberEnv('FUSE_ARANGE', 0), getNumberEnv('FUSE_CONV_BW', 0), getNumberEnv('LAZYCACHE', 1)]
export const [SPLIT_REDUCEOP, NO_MEMORY_PLANNER, RING] = [getNumberEnv('SPLIT_REDUCEOP', 1), getNumberEnv('NO_MEMORY_PLANNER', 0), getNumberEnv('RING', 1)]

// @dataclass(frozen=True)
export class Metadata {
  name: string
  caller: string
  backward: boolean
  constructor(name: string, caller: string, backward = false) {
    this.name = name, this.caller = caller, this.backward = backward
  }
  // def __hash__(self): return hash(self.name)
  //   def __repr__(self): return str(self) + (f" - {self.caller}" if self.caller else "")
  //   def __str__(self): return self.name + (" bw" if self.backward else "")
}
export const _METADATA: Metadata | undefined = undefined
// # **************** global state Counters ****************

export class GlobalCounters {
  static global_ops = 0
  static global_mem = 0
  static time_sum_s = 0
  static kernel_count = 0
  static mem_used = 0 // NOTE: this is not reset
  static reset = () => [GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.time_sum_s, GlobalCounters.kernel_count] = [0, 0, 0, 0]
}

// # **************** timer and profiler ****************

// class Timing(contextlib.ContextDecorator):
//   def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled
//   def __enter__(self): self.st = time.perf_counter_ns()
//   def __exit__(self, *exc):
//     self.et = time.perf_counter_ns() - self.st
//     if self.enabled: print(f"{self.prefix}{self.et*1e-6:6.2f} ms"+(self.on_exit(self.et) if self.on_exit else ""))

// def _format_fcn(fcn): return f"{fcn[0]}:{fcn[1]}:{fcn[2]}"
// class Profiling(contextlib.ContextDecorator):
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

// # *** universal database cache ***

const cacheDir = getEnv('XDG_CACHE_HOME', path.resolve(OSX ? path.join(os.homedir(), 'Library', 'Caches') : path.join(os.homedir(), '.cache')))
export const CACHEDB = getEnv('CACHEDB', path.resolve(path.join(cacheDir, 'tinygrad', 'cache.db')))
export const CACHELEVEL = getNumberEnv('CACHELEVEL', 2)

// VERSION = 16
// _db_connection = None
export const db_connection = () => {
  //   global _db_connection
  //   if _db_connection is None:
  //     os.makedirs(CACHEDB.rsplit(os.sep, 1)[0], exist_ok=True)
  //     _db_connection = sqlite3.connect(CACHEDB, timeout=60, isolation_level="IMMEDIATE")
  //     # another connection has set it already or is in the process of setting it
  //     # that connection will lock the database
  //     with contextlib.suppress(sqlite3.OperationalError): _db_connection.execute("PRAGMA journal_mode=WAL").fetchone()
  //     if DEBUG >= 7: _db_connection.set_trace_callback(print)
  //   return _db_connection
}
export const diskcache_clear = () => {
  //   cur = db_connection().cursor()
  //   drop_tables = cur.execute("SELECT 'DROP TABLE IF EXISTS ' || quote(name) || ';' FROM sqlite_master WHERE type = 'table';").fetchall()
  //   cur.executescript("\n".join([s[0] for s in drop_tables] + ["VACUUM;"]))
}
export const diskcache_get = (table: string, key: any): any | null => {
  //   if CACHELEVEL == 0: return None
  //   if isinstance(key, (str,int)): key = {"key": key}
  //   conn = db_connection()
  //   cur = conn.cursor()
  //   try:
  //     res = cur.execute(f"SELECT val FROM '{table}_{VERSION}' WHERE {' AND '.join([f'{x}=?' for x in key.keys()])}", tuple(key.values()))
  //   except sqlite3.OperationalError:
  //     return None  # table doesn't exist
  //   if (val:=res.fetchone()) is not None: return pickle.loads(val[0])
  return null
}
// _db_tables = set()
export const diskcache_put = (table: string, key: any, val: any) => {
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
// # *** http support ***

// def _ensure_downloads_dir() -> pathlib.Path:
//   # if we are on a tinybox, use the raid array
//   if pathlib.Path("/etc/tinybox-release").is_file():
//     # try creating dir with sudo
//     if not (downloads_dir := pathlib.Path("/raid/downloads")).exists():
//       subprocess.run(["sudo", "mkdir", "-p", downloads_dir], check=True)
//       subprocess.run(["sudo", "chown", "tiny:root", downloads_dir], check=True)
//       subprocess.run(["sudo", "chmod", "775", downloads_dir], check=True)
//     return downloads_dir
//   return pathlib.Path(_cache_dir) / "tinygrad" / "downloads"

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

// # *** Exec helpers

export const cpuTimeExecution = (cb: () => void, enable: boolean) => {
  let st = 0
  if (enable) st = performance.now()
  cb()
  if (enable) return performance.now() - st
}
export const cpuObjdump = (lib: bytes, objdumpTool = 'objdump') => {
  const outputFile = temp('temp_output.so')
  writeFileSync(outputFile, lib)
  try {
    const output = execSync(`${objdumpTool} -d ${outputFile}`, { encoding: 'utf-8' })
    console.log(output)
  } finally {
    unlinkSync(outputFile)
  }
}

// # *** ctypes helpers
export class bytes {
  toString = () => ''
  get length() {
    return 3
  }
}
export class bytearray {
  constructor(i: number) {}
}
export class memoryview {
  constructor(obj?: c_char | bytearray) {}
  get length() {
    return 0
  }
  get nbytes() {
    return 0
  }
  cast = (format: string, shape?: number[]) => new memoryview()
  slice = (from: number, to?: number) => new memoryview()
}

export class c_char {
  from_buffer = (mv: memoryview) => {}
  mul = (other: number) => new c_char()
  call = () => {}
}
export class c_ubyte {
  mul = (other: number) => new c_ubyte()
  call = () => {}
  fromAddress = (ptr: number) => {}
}
type CData = any

export class ctypes {
  static cast = (obj: any, type: any) => {
    return { contents: [new c_char()] }
  }
  static addressof = (obj: CData): number => {
    return 0
  }
  static memmove = (dst: any, src: any, count: number) => {}
  static POINTER = (type: c_char) => new c_char()
  static create_string_buffer = (init: bytes) => {}
  static c_char = new c_char()
  static c_uint8 = new c_ubyte()
  static CDLL = (file: string) => {
    return { get: (name: string) => (...args: any[]) => {} }
  }
}

// # TODO: make this work with read only memoryviews (if possible)
export const from_mv = (mv: memoryview, to_type = ctypes.c_char): c_char[] => {
  return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type.mul(mv.length))).contents
}
export const flat_mv = (mv: memoryview) => mv.length === 0 ? mv : mv.cast('B', [mv.nbytes])

// # *** universal support for code object pickling

// def _reconstruct_code(*args): return types.CodeType(*args)
// def _serialize_code(code:types.CodeType):
//   args = inspect.signature(types.CodeType).parameters  # NOTE: this works in Python 3.10 and up
//   return _reconstruct_code, tuple(code.__getattribute__('co_'+x.replace('codestr', 'code').replace('constants', 'consts')) for x in args)
// copyreg.pickle(types.CodeType, _serialize_code)

// def _serialize_module(module:types.ModuleType): return importlib.import_module, (module.__name__,)
// copyreg.pickle(types.ModuleType, _serialize_module)
