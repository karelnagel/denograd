// deno-lint-ignore-file no-explicit-any no-control-regex
import process from 'node:process'

// GENERAL HELPERS
export const range = (i: int) => Array.from({ length: i }, (_, i) => i)
export const sorted = <T>(x: T[], key?: (x: T) => keyof T): T[] => {
    if (key) {
        return [...x].sort((a, b) => {
            const ka: any = key(a)
            const kb: any = key(b)
            return ka < kb ? -1 : ka > kb ? 1 : 0
        })
    }
    return [...x].sort()
}
export const d = <T extends any[]>(...t: T) => t

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
export const allSame = <T>(items: T[]) => items.every((x) => x === items[0])
export const allInt = (t: any[]): t is int[] => t.every((s) => Number.isInteger(s))
export const colored = (st: str, color?: str, background = false) => {
    if (!color) return st
    const colors = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
    const code = 10 * (background ? 1 : 0) + 60 * (color === color.toUpperCase() ? 1 : 0) + 30 + colors.indexOf(color.toLowerCase())
    return `\u001b[${code}m${st}\u001b[0m`
}
export const colorizeFloat = (x: number) => colored(x.toFixed(2).padStart(7) + 'x', x < 0.75 ? 'green' : x > 1.15 ? 'red' : 'yellow')
export const memsizeToStr = (_b: int) => [d(1e9, 'GB'), d(1e6, 'MB'), d(1e3, 'KB'), d(1, 'B')].filter(([d]) => _b > d).map(([d, pr]) => `${(_b / d).toFixed(2)} ${pr}`)[0]
export const ansistrip = (s: str) => s.replace(/\x1b\[(K|.*?m)/g, '')
export const ansilen = (s: str) => ansistrip(s).length
export const makeTuple = (x: int | int[], cnt: int): int[] => Array.isArray(x) ? [...x] : Array(cnt).fill(x)
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
export const stripParens = (s: str) => s[0] === '(' && s[s.length - 1] === ')' && s.slice(1, -1).indexOf('(') <= s.slice(1, -1).indexOf(')') ? s.slice(1, -1) : s
export const ceildiv = (num: number, amt: number): number => {
    const ret = -(Math.floor(-num / amt))
    return Number.isInteger(ret) ? ret : Math.floor(ret)
}
export const roundUp = (num: int, amt: int): int => Math.ceil(num / amt) * amt
export const data64 = (data: int): [int, int] => [
    Math.floor(data / Math.pow(2, 32)), // correctly get upper 32 bits
    data >>> 0, // get lower 32 bits with unsigned right shift
]
export const data64_le = (data: int): [int, int] => [
    data >>> 0, // get lower 32 bits with unsigned right shift
    Math.floor(data / Math.pow(2, 32)), // correctly get upper 32 bits
]
export const mergeDicts = <T extends string, U = any>(ds: Record<T, U>[]): Record<T, U> => {
    const kvs = new Set(ds.flatMap((d) => Object.entries(d))) as Set<[T, U]>
    const keys = new Set(Array.from(kvs).map((kv) => kv[0]))
    if (kvs.size !== keys.size) throw new Error(`cannot merge, ${Array.from(kvs)} contains different values for the same key`)
    return Object.fromEntries(Array.from(kvs)) as Record<T, U>
}
export const partition = <T>(itr: Iterable<T>, fxn: (x: T) => boolean): [T[], T[]] => {
    const a: T[] = []
    const b: T[] = []
    for (const s of itr) (fxn(s) ? a : b).push(s)
    return [a, b]
}
// def unwrap(x:Optional[T]) -> T:
//   assert x is not None
//   return x
// def get_child(obj, key):
//   for k in key.split('.'):
//     if k.isnumeric(): obj = obj[int(k)]
//     elif isinstance(obj, dict): obj = obj[k]
//     else: obj = getattr(obj, k)
//   return obj
// def word_wrap(x, wrap=80): return x if len(x) <= wrap else (x[0:wrap] + "\n" + word_wrap(x[wrap:], wrap))

// # for length N coefficients `p`, returns p[0] * x**(N-1) + p[1] * x**(N-2) + ... + p[-2] * x + p[-1]
// def polyN(x:T, p:List[float]) -> T: return functools.reduce(lambda acc,c: acc*x+c, p, 0.0)  # type: ignore

// @functools.lru_cache(maxsize=None)
// def to_function_name(s:str): return ''.join([c if c in (str.ascii_letters+str.digits+'_') else f'{ord(c):02X}' for c in ansistrip(s)])
// @functools.lru_cache(maxsize=None)
// def getenv(key:str, default=0): return type(default)(os.getenv(key, default))
// def temp(x:str) -> str: return (pathlib.Path(tempfile.gettempdir()) / x).as_posix()

// class Context(contextlib.ContextDecorator):
//   stack: ClassVar[List[dict[str, int]]] = [{}]
//   def __init__(self, **kwargs): self.kwargs = kwargs
//   def __enter__(self):
//     Context.stack[-1] = {k:o.value for k,o in ContextVar._cache.items()} # Store current state.
//     for k,v in self.kwargs.items(): ContextVar._cache[k].value = v # Update to new temporary state.
//     Context.stack.append(self.kwargs) # Store the temporary state so we know what to undo later.
//   def __exit__(self, *args):
//     for k in Context.stack.pop(): ContextVar._cache[k].value = Context.stack[-1].get(k, ContextVar._cache[k].value)

// class ContextVar:
//   _cache: ClassVar[Dict[str, ContextVar]] = {}
//   value: int
//   key: str
//   def __init__(self, key, default_value):
//     assert key not in ContextVar._cache, f"attempt to recreate ContextVar {key}"
//     ContextVar._cache[key] = self
//     self.value, self.key = getenv(key, default_value), key
//   def __bool__(self): return bool(self.value)
//   def __ge__(self, x): return self.value >= x
//   def __gt__(self, x): return self.value > x
//   def __lt__(self, x): return self.value < x

// DEBUG, IMAGE, BEAM, NOOPT, JIT = ContextVar("DEBUG", 0), ContextVar("IMAGE", 0), ContextVar("BEAM", 0), ContextVar("NOOPT", 0), ContextVar("JIT", 1)
// WINO, CAPTURING, TRACEMETA = ContextVar("WINO", 0), ContextVar("CAPTURING", 1), ContextVar("TRACEMETA", 1)
// PROFILE, PROFILEPATH = ContextVar("PROFILE", 0), ContextVar("PROFILEPATH", temp("tinygrad_profile.json"))
// USE_TC, TC_OPT, AMX, TRANSCENDENTAL = ContextVar("TC", 1), ContextVar("TC_OPT", 0), ContextVar("AMX", 0), ContextVar("TRANSCENDENTAL", 1)
// FUSE_ARANGE, FUSE_CONV_BW, LAZYCACHE = ContextVar("FUSE_ARANGE", 0), ContextVar("FUSE_CONV_BW", 0), ContextVar("LAZYCACHE", 1)
// SPLIT_REDUCEOP, NO_MEMORY_PLANNER, RING = ContextVar("SPLIT_REDUCEOP", 1), ContextVar("NO_MEMORY_PLANNER", 0), ContextVar("RING", 1)

// @dataclass(frozen=True)
// class Metadata:
//   name: str
//   caller: str
//   backward: bool = False
//   def __hash__(self): return hash(self.name)
//   def __repr__(self): return str(self) + (f" - {self.caller}" if self.caller else "")
//   def __str__(self): return self.name + (" bw" if self.backward else "")
// _METADATA: contextvars.ContextVar[Optional[Metadata]] = contextvars.ContextVar("_METADATA", default=None)

// # **************** global state Counters ****************

// class GlobalCounters:
//   global_ops: ClassVar[int] = 0
//   global_mem: ClassVar[int] = 0
//   time_sum_s: ClassVar[float] = 0.0
//   kernel_count: ClassVar[int] = 0
//   mem_used: ClassVar[int] = 0   # NOTE: this is not reset
//   @staticmethod
//   def reset(): GlobalCounters.global_ops, GlobalCounters.global_mem, GlobalCounters.time_sum_s, GlobalCounters.kernel_count = 0,0,0.0,0

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

// _cache_dir: str = getenv("XDG_CACHE_HOME", os.path.expanduser("~/Library/Caches" if OSX else "~/.cache"))
// CACHEDB: str = getenv("CACHEDB", os.path.abspath(os.path.join(_cache_dir, "tinygrad", "cache.db")))
// CACHELEVEL = getenv("CACHELEVEL", 2)

// VERSION = 16
// _db_connection = None
// def db_connection():
//   global _db_connection
//   if _db_connection is None:
//     os.makedirs(CACHEDB.rsplit(os.sep, 1)[0], exist_ok=True)
//     _db_connection = sqlite3.connect(CACHEDB, timeout=60, isolation_level="IMMEDIATE")
//     # another connection has set it already or is in the process of setting it
//     # that connection will lock the database
//     with contextlib.suppress(sqlite3.OperationalError): _db_connection.execute("PRAGMA journal_mode=WAL").fetchone()
//     if DEBUG >= 7: _db_connection.set_trace_callback(print)
//   return _db_connection

// def diskcache_clear():
//   cur = db_connection().cursor()
//   drop_tables = cur.execute("SELECT 'DROP TABLE IF EXISTS ' || quote(name) || ';' FROM sqlite_master WHERE type = 'table';").fetchall()
//   cur.executescript("\n".join([s[0] for s in drop_tables] + ["VACUUM;"]))

// def diskcache_get(table:str, key:Union[Dict, str, int]) -> Any:
//   if CACHELEVEL == 0: return None
//   if isinstance(key, (str,int)): key = {"key": key}
//   conn = db_connection()
//   cur = conn.cursor()
//   try:
//     res = cur.execute(f"SELECT val FROM '{table}_{VERSION}' WHERE {' AND '.join([f'{x}=?' for x in key.keys()])}", tuple(key.values()))
//   except sqlite3.OperationalError:
//     return None  # table doesn't exist
//   if (val:=res.fetchone()) is not None: return pickle.loads(val[0])
//   return None

// _db_tables = set()
// def diskcache_put(table:str, key:Union[Dict, str, int], val:Any):
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

// def diskcache(func):
//   def wrapper(*args, **kwargs) -> bytes:
//     table, key = f"cache_{func.__name__}", hashlib.sha256(pickle.dumps((args, kwargs))).hexdigest()
//     if (ret:=diskcache_get(table, key)): return ret
//     return diskcache_put(table, key, func(*args, **kwargs))
//   return wrapper

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

// def cpu_time_execution(cb, enable):
//   if enable: st = time.perf_counter()
//   cb()
//   if enable: return time.perf_counter()-st

// def cpu_objdump(lib, objdump_tool='objdump'):
//   with tempfile.NamedTemporaryFile(delete=True) as f:
//     pathlib.Path(f.name).write_bytes(lib)
//     print(subprocess.check_output([objdump_tool, '-d', f.name]).decode('utf-8'))

// # *** ctypes helpers

// # TODO: make this work with read only memoryviews (if possible)
// def from_mv(mv:memoryview, to_type=ctypes.c_char):
//   return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type * len(mv))).contents
// def to_mv(ptr, sz) -> memoryview: return memoryview(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * sz)).contents).cast("B")
// def mv_address(mv:memoryview): return ctypes.addressof(ctypes.c_char.from_buffer(mv))
// def to_char_p_p(options: List[bytes], to_type=ctypes.c_char): return (ctypes.POINTER(to_type) * len(options))(*[ctypes.cast(ctypes.create_str_buffer(o), ctypes.POINTER(to_type)) for o in options])  # noqa: E501
// @functools.lru_cache(maxsize=None)
// def init_c_struct_t(fields: Tuple[Tuple[str, ctypes._SimpleCData], ...]):
//   class CStruct(ctypes.Structure):
//     _pack_, _fields_ = 1, fields
//   return CStruct
// def init_c_var(ctypes_var, creat_cb): return (creat_cb(ctypes_var), ctypes_var)[1]
// def flat_mv(mv:memoryview): return mv if len(mv) == 0 else mv.cast("B", shape=(mv.nbytes,))

// # *** universal support for code object pickling

// def _reconstruct_code(*args): return types.CodeType(*args)
// def _serialize_code(code:types.CodeType):
//   args = inspect.signature(types.CodeType).parameters  # NOTE: this works in Python 3.10 and up
//   return _reconstruct_code, tuple(code.__getattribute__('co_'+x.replace('codestr', 'code').replace('constants', 'consts')) for x in args)
// copyreg.pickle(types.CodeType, _serialize_code)

// def _serialize_module(module:types.ModuleType): return importlib.import_module, (module.__name__,)
// copyreg.pickle(types.ModuleType, _serialize_module)
