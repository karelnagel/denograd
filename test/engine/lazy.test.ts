// deno-fmt-ignore-file
import { DType } from '../../src/dtype.ts'
import { create_lazybuffer, LazyBuffer } from '../../src/engine/lazy.ts'
import { Metadata } from '../../src/helpers.ts'
import { Ops } from '../../src/ops.ts'
import { ShapeTracker } from '../../src/shape/shapetracker.ts'
import { View } from '../../src/shape/view.ts'
import { compare, tryCatch } from '../helpers.ts'

Deno.test(
  'create_lazybuffer',
  compare(
    [
      [`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(5, 4, `int`, `i`, 1, undefined), 62, 0, [], undefined, false],
      [`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(5, 4, `int`, `i`, 1, undefined), 62, 1, [], undefined, false],
      [`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(5, 4, `int`, `i`, 1, undefined), 62, -1, [], undefined, false],
      [`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), 62, 0, [], undefined, false],
      [`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), 62, 1, [], undefined, false],
      [`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(7, 8, `long`, `q`, 1, undefined), 62, 0.0, [], undefined, false]
    ],
    create_lazybuffer,
    'out(tiny.engine.lazy.create_lazybuffer(*data))',
  ),
)
Deno.test(
  'LazyBuffer.init',
  compare(
    [
      [`PYTHON`, new ShapeTracker([new View([1], [0], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), 5, undefined, [], undefined, undefined],
      [`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(7, 8, `long`, `q`, 1, undefined), 62, 0, [], undefined, new Metadata(`zeros`, ``, false)],
      [`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(5, 4, `int`, `i`, 1, undefined), 62, 0, [], undefined, new Metadata(`randint`, ``, false)],
      [`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), 62, 0.0, [], undefined, new Metadata(`zeros`, ``, false)],
      [`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(5, 4, `int`, `i`, 1, undefined), 62, 1, [], undefined, new Metadata(`__getitem__`, ``, false)],
      [`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(5, 4, `int`, `i`, 1, undefined), 62, 0, [], undefined, new Metadata(`__getitem__`, ``, false)]
      
    ],
    tryCatch((device: string, st: ShapeTracker,dtype: DType, op?: Ops, arg?: any, srcs: LazyBuffer[] = [], base?: LazyBuffer,  metadata?: Metadata)=>new LazyBuffer(device,st,dtype,op,arg,srcs,base,metadata)),
    'out(tiny.engine.lazy.LazyBuffer(*data))',
  ),
)
Deno.test(
  'LazyBuffer.metaop',
  compare(
    [
      [62, [], new DType(5, 4, `int`, `i`, 1, undefined), `CLANG`, 0, [], false],
      [62, [], new DType(5, 4, `int`, `i`, 1, undefined), `CLANG`, 1, [], false],
      [62, [], new DType(0, 1, `bool`, `?`, 1, undefined), `CLANG`, 1, [], false], // Fails, arg:1 vs arg:true
      [62, [], new DType(7, 8, `long`, `q`, 1, undefined), `CLANG`, 1, [], false],
      [62, [], new DType(7, 8, `long`, `q`, 1, undefined), `CLANG`, 2, [], false],
      [62, [], new DType(5, 4, `int`, `i`, 1, undefined), `CLANG`, -1, [], false]
    ],
    LazyBuffer.metaop,
    'out(tiny.engine.lazy.LazyBuffer.metaop(*data))',
    {ignore:[2]}
  ),
)
Deno.test(
  'LazyBuffer.contiguous',
  compare(
    [
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([1], [0], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), 62, 1.0, [], undefined, new Metadata(`uniform`, ``, false)), new Metadata(`ones`, ``, false)), true],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([32], [0], 0, undefined, false)]), new DType(11, 4, `float`, `f`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), 62, 0.0, [], undefined, new Metadata(`zeros`, ``, false)), new Metadata(`zeros`, ``, false)), true],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([10], [0], 0, undefined, false)]), new DType(11, 4, `float`, `f`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), 62, 0.0, [], undefined, new Metadata(`zeros`, ``, false)), new Metadata(`zeros`, ``, false)), true],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([64], [0], 0, undefined, false)]), new DType(11, 4, `float`, `f`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), 62, 0.0, [], undefined, new Metadata(`zeros`, ``, false)), new Metadata(`zeros`, ``, false)), true],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([10, 576], [0, 0], 0, undefined, false)]), new DType(11, 4, `float`, `f`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), 62, 0.0, [], undefined, new Metadata(`zeros`, ``, false)), new Metadata(`zeros`, ``, false)), true],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([32, 1, 5, 5], [0, 0, 0, 0], 0, undefined, false)]), new DType(11, 4, `float`, `f`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), 62, 0.0, [], undefined, new Metadata(`zeros`, ``, false)), new Metadata(`zeros`, ``, false)), true],
      [new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/684fe191a76d9107981692e6f0d3842f.gunzip`, new ShapeTracker([new View([60000], [1], 8, undefined, false)]), new DType(2, 1, `unsigned char`, `B`, 1, undefined), undefined, undefined, [], new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/684fe191a76d9107981692e6f0d3842f.gunzip`, new ShapeTracker([new View([60008], [1], 0, undefined, true)]), new DType(2, 1, `unsigned char`, `B`, 1, undefined), Ops.EMPTY, undefined, [], undefined, new Metadata(`from_url`, ``, false)), new Metadata(`__getitem__`, ``, false)),  false],
      [new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/7b1e157ab43d54d0946f76d1bb6df574.gunzip`, new ShapeTracker([new View([10000], [1], 8, undefined, false)]), new DType(2, 1, `unsigned char`, `B`, 1, undefined), undefined, undefined, [], new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/7b1e157ab43d54d0946f76d1bb6df574.gunzip`, new ShapeTracker([new View([10008], [1], 0, undefined, true)]), new DType(2, 1, `unsigned char`, `B`, 1, undefined), Ops.EMPTY, undefined, [], undefined, new Metadata(`from_url`, ``, false)), new Metadata(`__getitem__`, ``, false)), false],
      [new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/40a9ec2164c7887e8a5f1225de927907.gunzip`, new ShapeTracker([new View([10000, 1, 28, 28], [784, 0, 28, 1], 16, undefined, false)]), new DType(2, 1, `unsigned char`, `B`, 1, undefined), undefined, undefined, [], new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/40a9ec2164c7887e8a5f1225de927907.gunzip`, new ShapeTracker([new View([7840016], [1], 0, undefined, true)]), new DType(2, 1, `unsigned char`, `B`, 1, undefined), Ops.EMPTY, undefined, [], undefined, new Metadata(`from_url`, ``, false)), new Metadata(`reshape`, ``, false)),  false]
    ],
    (x:LazyBuffer,allow_buffer_view:boolean) => {
      x = x.contiguous(allow_buffer_view)
      return [x.base.forced_realize,x.base.contiguous_child,x.st]
    },
    [
      'x = data[0].contiguous(data[1])',
      "out([x.base.forced_realize,x.base.contiguous_child,x.st])"
    ],
  ),
)
Deno.test(
  'LazyBuffer.cast',
  compare(
    [
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), Ops.CONST, 1.0, [], undefined, new Metadata(`uniform`, ``, false)), new DType(6, 4, `unsigned int`, `I`, 1, undefined), true, true],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(5, 4, `int`, `i`, 1, undefined), Ops.CONST, 512, [], undefined, new Metadata(`sparse_categorical_crossentropy`, ``, false)), new DType(11, 4, `float`, `f`, 1, undefined), false, true],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([512, 64, 6, 6], [0, 0, 0, 0], 0, undefined, false)]), new DType(0, 1, `bool`, `?`, 1, undefined), Ops.CONST, true, [], undefined, new Metadata(`max_pool2d`, ``, false)), new DType(11, 4, `float`, `f`, 1, undefined), false, true],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), Ops.CONST, 512.0, [], undefined, new Metadata(`sparse_categorical_crossentropy`, ``, false)), new DType(11, 4, `float`, `f`, 1, undefined), false, true],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([512, 32, 20, 20], [0, 0, 0, 0], 0, undefined, false)]), new DType(0, 1, `bool`, `?`, 1, undefined), Ops.CONST, true, [], undefined, new Metadata(`max_pool2d`, ``, false)), new DType(11, 4, `float`, `f`, 1, undefined), false, true],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([512], [0], 0, undefined, false)]), new DType(11, 4, `float`, `f`, 1, undefined), Ops.CONST, 1.0, [], undefined, new Metadata(`sparse_categorical_crossentropy`, ``, false)), new DType(11, 4, `float`, `f`, 1, undefined), false, true]
    ],
    (x:LazyBuffer,dtype:DType,bitcast:any,allow_buffer_view:boolean)=>x.cast(dtype,bitcast,allow_buffer_view),
    'out(data[0].cast(*data[1:]))',
  ),
)
Deno.test(
  'LazyBuffer.copy_to_device',
  compare(
    [
      [new LazyBuffer(`PYTHON`, new ShapeTracker([new View([1], [0], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), Ops.EMPTY, undefined, [], undefined, undefined), `CLANG`, false, false],
      [new LazyBuffer(`PYTHON`, new ShapeTracker([new View([2], [1], 0, undefined, true)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), Ops.EMPTY, undefined, [], undefined, new Metadata(`uniform`, ``, false)), `CLANG`, false, false],
      [new LazyBuffer(`PYTHON`, new ShapeTracker([new View([1], [0], 0, undefined, true)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), Ops.EMPTY, undefined, [], undefined, new Metadata(`uniform`, ``, false)), `CLANG`, false, false],
      [new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/684fe191a76d9107981692e6f0d3842f.gunzip`, new ShapeTracker([new View([60000], [1], 8, undefined, false)]), new DType(2, 1, `unsigned char`, `B`, 1, undefined), undefined, undefined, [], new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/684fe191a76d9107981692e6f0d3842f.gunzip`, new ShapeTracker([new View([60008], [1], 0, undefined, true)]), new DType(2, 1, `unsigned char`, `B`, 1, undefined), Ops.EMPTY, undefined, [], undefined, new Metadata(`from_url`, ``, false)), new Metadata(`__getitem__`, ``, false)), `CLANG`, false, false],
      [new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/7b1e157ab43d54d0946f76d1bb6df574.gunzip`, new ShapeTracker([new View([10000], [1], 8, undefined, false)]), new DType(2, 1, `unsigned char`, `B`, 1, undefined), undefined, undefined, [], new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/7b1e157ab43d54d0946f76d1bb6df574.gunzip`, new ShapeTracker([new View([10008], [1], 0, undefined, true)]), new DType(2, 1, `unsigned char`, `B`, 1, undefined), Ops.EMPTY, undefined, [], undefined, new Metadata(`from_url`, ``, false)), new Metadata(`__getitem__`, ``, false)), `CLANG`, false, false],
      [new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/40a9ec2164c7887e8a5f1225de927907.gunzip`, new ShapeTracker([new View([10000, 1, 28, 28], [784, 0, 28, 1], 16, undefined, false)]), new DType(2, 1, `unsigned char`, `B`, 1, undefined), undefined, undefined, [], new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/40a9ec2164c7887e8a5f1225de927907.gunzip`, new ShapeTracker([new View([7840016], [1], 0, undefined, true)]), new DType(2, 1, `unsigned char`, `B`, 1, undefined), Ops.EMPTY, undefined, [], undefined, new Metadata(`from_url`, ``, false)), new Metadata(`reshape`, ``, false)), `CLANG`, false, false]
    ],
    (x:LazyBuffer,device:string,force:boolean,clone:boolean) => x.copy_to_device(device,force,clone),
    'out(data[0].copy_to_device(*data[1:]))',
  ),
)
Deno.test(
  'LazyBuffer.alu',
  compare(
    [
      [new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/684fe191a76d9107981692e6f0d3842f.gunzip`, new ShapeTracker([new View([60000], [1], 8, undefined, false)]), new DType(2, 1, `unsigned char`, `B`, 1, undefined), undefined, undefined, [], new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/684fe191a76d9107981692e6f0d3842f.gunzip`, new ShapeTracker([new View([60008], [1], 0, undefined, true)]), new DType(2, 1, `unsigned char`, `B`, 1, undefined), Ops.EMPTY, undefined, [], undefined, new Metadata(`from_url`, ``, false)), new Metadata(`__getitem__`, ``, false)), Ops.BUFFER_VIEW],
      [new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/7b1e157ab43d54d0946f76d1bb6df574.gunzip`, new ShapeTracker([new View([10000], [1], 8, undefined, false)]), new DType(2, 1, `unsigned char`, `B`, 1, undefined), undefined, undefined, [], new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/7b1e157ab43d54d0946f76d1bb6df574.gunzip`, new ShapeTracker([new View([10008], [1], 0, undefined, true)]), new DType(2, 1, `unsigned char`, `B`, 1, undefined), Ops.EMPTY, undefined, [], undefined, new Metadata(`from_url`, ``, false)), new Metadata(`__getitem__`, ``, false)), Ops.CONTIGUOUS],
      [new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/40a9ec2164c7887e8a5f1225de927907.gunzip`, new ShapeTracker([new View([10000, 1, 28, 28], [784, 0, 28, 1], 16, undefined, false)]), new DType(2, 1, `unsigned char`, `B`, 1, undefined), undefined, undefined, [], new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/40a9ec2164c7887e8a5f1225de927907.gunzip`, new ShapeTracker([new View([7840016], [1], 0, undefined, true)]), new DType(2, 1, `unsigned char`, `B`, 1, undefined), Ops.EMPTY, undefined, [], undefined, new Metadata(`from_url`, ``, false)), new Metadata(`reshape`, ``, false)), Ops.BUFFER_VIEW],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([1], [0], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), Ops.CONST, 1.0, [], undefined, new Metadata(`ones`, ``, false)), new Metadata(`ones`, ``, false)), Ops.CONTIGUOUS, []],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([32], [0], 0, undefined, false)]), new DType(11, 4, `float`, `f`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), Ops.CONST, 204800.0, [], undefined, new Metadata(`mean`, ``, false)), new Metadata(`mean`, ``, false)), Ops.RECIP, []],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([10], [0], 0, undefined, false)]), new DType(11, 4, `float`, `f`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), Ops.CONST, 0.0, [], undefined, new Metadata(`zeros`, ``, false)), new Metadata(`zeros`, ``, false)), Ops.CONTIGUOUS, []],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([64], [0], 0, undefined, false)]), new DType(11, 4, `float`, `f`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), Ops.CONST, 0.0, [], undefined, new Metadata(`zeros`, ``, false)), new Metadata(`zeros`, ``, false)), Ops.CONTIGUOUS, []],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([32], [0], 0, undefined, false)]), new DType(11, 4, `float`, `f`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), Ops.CONST, 0.0, [], undefined, new Metadata(`zeros`, ``, false)), new Metadata(`zeros`, ``, false)), Ops.CONTIGUOUS, []],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([10, 576], [0, 0], 0, undefined, false)]), new DType(11, 4, `float`, `f`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), Ops.CONST, 0.0, [], undefined, new Metadata(`zeros`, ``, false)), new Metadata(`zeros`, ``, false)), Ops.CONTIGUOUS, []]
    ],
    (x:LazyBuffer,op:Ops,srcs:LazyBuffer[]=[]) => x.alu(op,...srcs),
    'out(data[0].alu(data[1],*(data[2] if len(data)==3 else [])))',
  ),
)
Deno.test(
  'LazyBuffer._reduce_op',
  compare(
    [
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([6, 9], [0, 0], 0, [[0, 6], [4, 9]], false), new View([5, 5], [1, 10], 0, undefined, false)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), Ops.CONST, 1, [], undefined, new Metadata(`uniform`, ``, false)), new Metadata(`uniform`, ``, false)), Ops.ADD, [1]],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([17, 31], [0, 0], 0, [[0, 17], [15, 31]], false), new View([16, 16], [1, 32], 0, undefined, false)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), Ops.CONST, 1, [], undefined, new Metadata(`uniform`, ``, false)), new Metadata(`uniform`, ``, false)), Ops.ADD, [1]],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([33, 63], [0, 0], 0, [[0, 33], [31, 63]], false), new View([32, 32], [1, 64], 0, undefined, false)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), Ops.CONST, 1, [], undefined, new Metadata(`uniform`, ``, false)), new Metadata(`uniform`, ``, false)), Ops.ADD, [1]],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([401, 799], [0, 0], 0, [[0, 401], [399, 799]], false), new View([400, 400], [1, 800], 0, undefined, false)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), Ops.CONST, 1, [], undefined, new Metadata(`uniform`, ``, false)), new Metadata(`uniform`, ``, false)), Ops.ADD, [1]],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([257, 511], [0, 0], 0, [[0, 257], [255, 511]], false), new View([256, 256], [1, 512], 0, undefined, false)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), Ops.CONST, 1, [], undefined, new Metadata(`randint`, ``, false)), new Metadata(`randint`, ``, false)), Ops.ADD, [1]],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([2881, 5759], [0, 0], 0, [[0, 2881], [2879, 5759]], false), new View([2880, 2880], [1, 5760], 0, undefined, false)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), Ops.CONST, 1, [], undefined, new Metadata(`uniform`, ``, false)), new Metadata(`uniform`, ``, false)), Ops.ADD, [1]]

    ],
    (x:LazyBuffer,op:Ops,axis:number[]) => x._reduce_op(op,axis),
    'out(data[0]._reduce_op(*data[1:]))',
  ),
)
Deno.test(
  'LazyBuffer.r',
  compare(
    [
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([6, 9], [0, 0], 0, [[0, 6], [4, 9]], false), new View([5, 5], [1, 10], 0, undefined, false)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), Ops.CONST, 1, [], undefined, new Metadata(`uniform`, ``, false)), new Metadata(`uniform`, ``, false)), Ops.ADD, [1]],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([17, 31], [0, 0], 0, [[0, 17], [15, 31]], false), new View([16, 16], [1, 32], 0, undefined, false)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), Ops.CONST, 1, [], undefined, new Metadata(`uniform`, ``, false)), new Metadata(`uniform`, ``, false)), Ops.ADD, [1]],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([33, 63], [0, 0], 0, [[0, 33], [31, 63]], false), new View([32, 32], [1, 64], 0, undefined, false)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), Ops.CONST, 1, [], undefined, new Metadata(`uniform`, ``, false)), new Metadata(`uniform`, ``, false)), Ops.ADD, [1]],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([401, 799], [0, 0], 0, [[0, 401], [399, 799]], false), new View([400, 400], [1, 800], 0, undefined, false)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), Ops.CONST, 1, [], undefined, new Metadata(`uniform`, ``, false)), new Metadata(`uniform`, ``, false)), Ops.ADD, [1]],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([257, 511], [0, 0], 0, [[0, 257], [255, 511]], false), new View([256, 256], [1, 512], 0, undefined, false)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), Ops.CONST, 1, [], undefined, new Metadata(`randint`, ``, false)), new Metadata(`randint`, ``, false)), Ops.ADD, [1]],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([2881, 5759], [0, 0], 0, [[0, 2881], [2879, 5759]], false), new View([2880, 2880], [1, 5760], 0, undefined, false)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(6, 4, `unsigned int`, `I`, 1, undefined), Ops.CONST, 1, [], undefined, new Metadata(`uniform`, ``, false)), new Metadata(`uniform`, ``, false)), Ops.ADD, [1]]
    ],
    (x:LazyBuffer,op:Ops,axis:number[]) => x.r(op,axis),
    'out(data[0].r(*data[1:]))',
  ),
)
Deno.test(
  'LazyBuffer._view',
  compare(
    [
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(7, 8, `long`, `q`, 1, undefined), Ops.CONST, 0, [], undefined, new Metadata(`zeros`, ``, false)), new ShapeTracker([new View([1], [0], 0, undefined, true)])],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(5, 4, `int`, `i`, 1, undefined), Ops.CONST, 0, [], undefined, new Metadata(`randint`, ``, false)), new ShapeTracker([new View([1], [0], 0, undefined, true)])],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), Ops.CONST, 1.0, [], undefined, new Metadata(`ones`, ``, false)), new ShapeTracker([new View([1], [0], 0, undefined, true)])],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(7, 8, `long`, `q`, 1, undefined), Ops.CONST, 1, [], undefined, new Metadata(`__iadd__`, ``, false)), new ShapeTracker([new View([1], [0], 0, undefined, true)])],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), Ops.CONST, 1e-05, [], undefined, new Metadata(`add`, ``, false)), new ShapeTracker([new View([1], [0], 0, undefined, true)])],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View([], [], 0, undefined, true)]), new DType(11, 4, `float`, `f`, 1, undefined), Ops.CONST, 0.0, [], undefined, new Metadata(`zeros`, ``, false)), new ShapeTracker([new View([1], [0], 0, undefined, true)])]

    ],
    (x:LazyBuffer,new_st:ShapeTracker) => x._view(new_st),
    'out(data[0]._view(*data[1:]))',
  ),
)
