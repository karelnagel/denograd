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
      [`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), 62, 0, [], undefined, false],
      [`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), 62, 1, [], undefined, false],
      [`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), 62, -1, [], undefined, false],
      [`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), 62, 0, [], undefined, false],
      [`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), 62, 1, [], undefined, false],
      [`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:7, itemsize:8, name:`long`, fmt:`q`, count:1, _scalar:undefined }), 62, 0.0, [], undefined, false]
    ],
    create_lazybuffer,
    'out(tiny.engine.lazy.create_lazybuffer(*data))',
  ),
)
Deno.test(
  'LazyBuffer.init',
  compare(
    [
      [`PYTHON`, new ShapeTracker([new View({ shape:[1], strides:[0], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), 5, undefined, [], undefined, undefined],
      [`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:7, itemsize:8, name:`long`, fmt:`q`, count:1, _scalar:undefined }), 62, 0, [], undefined, new Metadata(`zeros`, ``, false)],
      [`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), 62, 0, [], undefined, new Metadata(`randint`, ``, false)],
      [`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), 62, 0.0, [], undefined, new Metadata(`zeros`, ``, false)],
      [`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), 62, 1, [], undefined, new Metadata(`__getitem__`, ``, false)],
      [`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), 62, 0, [], undefined, new Metadata(`__getitem__`, ``, false)]
      
    ],
    tryCatch((device: string, st: ShapeTracker,dtype: DType, op?: Ops, arg?: any, srcs: LazyBuffer[] = [], base?: LazyBuffer,  metadata?: Metadata)=>new LazyBuffer(device,st,dtype,op,arg,srcs,base,metadata)),
    'out(tiny.engine.lazy.LazyBuffer(*data))',
  ),
)
Deno.test(
  'LazyBuffer.metaop',
  compare(
    [
      [62, [], new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), `CLANG`, 0, [], false],
      [62, [], new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), `CLANG`, 1, [], false],
      [62, [], new DType({ priority:0, itemsize:1, name:`bool`, fmt:`?`, count:1, _scalar:undefined }), `CLANG`, 1, [], false], // Fails, arg:1 vs arg:true
      [62, [], new DType({ priority:7, itemsize:8, name:`long`, fmt:`q`, count:1, _scalar:undefined }), `CLANG`, 1, [], false],
      [62, [], new DType({ priority:7, itemsize:8, name:`long`, fmt:`q`, count:1, _scalar:undefined }), `CLANG`, 2, [], false],
      [62, [], new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), `CLANG`, -1, [], false]
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
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View({ shape:[1], strides:[0], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), 62, 1.0, [], undefined, new Metadata(`uniform`, ``, false)), new Metadata(`ones`, ``, false)), true],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View({ shape:[32], strides:[0], offset:0, mask:undefined, contiguous:false })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), 62, 0.0, [], undefined, new Metadata(`zeros`, ``, false)), new Metadata(`zeros`, ``, false)), true],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View({ shape:[10], strides:[0], offset:0, mask:undefined, contiguous:false })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), 62, 0.0, [], undefined, new Metadata(`zeros`, ``, false)), new Metadata(`zeros`, ``, false)), true],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View({ shape:[64], strides:[0], offset:0, mask:undefined, contiguous:false })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), 62, 0.0, [], undefined, new Metadata(`zeros`, ``, false)), new Metadata(`zeros`, ``, false)), true],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View({ shape:[10, 576], strides:[0, 0], offset:0, mask:undefined, contiguous:false })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), 62, 0.0, [], undefined, new Metadata(`zeros`, ``, false)), new Metadata(`zeros`, ``, false)), true],
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View({ shape:[32, 1, 5, 5], strides:[0, 0, 0, 0], offset:0, mask:undefined, contiguous:false })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), undefined, undefined, [], new LazyBuffer(`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), 62, 0.0, [], undefined, new Metadata(`zeros`, ``, false)), new Metadata(`zeros`, ``, false)), true]
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
      [new LazyBuffer(`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), Ops.CONST, 1.0, [], undefined, new Metadata(`uniform`, ``, false)), new DType({ priority:6, itemsize:4, name:`unsigned int`, fmt:`I`, count:1, _scalar:undefined }), true, true],
    [new LazyBuffer(`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), Ops.CONST, 512, [], undefined, new Metadata(`sparse_categorical_crossentropy`, ``, false)), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), false, true],
    [new LazyBuffer(`CLANG`, new ShapeTracker([new View({ shape:[512, 64, 6, 6], strides:[0, 0, 0, 0], offset:0, mask:undefined, contiguous:false })]), new DType({ priority:0, itemsize:1, name:`bool`, fmt:`?`, count:1, _scalar:undefined }), Ops.CONST, true, [], undefined, new Metadata(`max_pool2d`, ``, false)), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), false, true],
    [new LazyBuffer(`CLANG`, new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), Ops.CONST, 512.0, [], undefined, new Metadata(`sparse_categorical_crossentropy`, ``, false)), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), false, true],
    [new LazyBuffer(`CLANG`, new ShapeTracker([new View({ shape:[512, 32, 20, 20], strides:[0, 0, 0, 0], offset:0, mask:undefined, contiguous:false })]), new DType({ priority:0, itemsize:1, name:`bool`, fmt:`?`, count:1, _scalar:undefined }), Ops.CONST, true, [], undefined, new Metadata(`max_pool2d`, ``, false)), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), false, true],
    [new LazyBuffer(`CLANG`, new ShapeTracker([new View({ shape:[512], strides:[0], offset:0, mask:undefined, contiguous:false })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), Ops.CONST, 1.0, [], undefined, new Metadata(`sparse_categorical_crossentropy`, ``, false)), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), false, true]
    ],
    (x:LazyBuffer,dtype:DType,bitcast:any,allow_buffer_view:boolean)=>x.cast(dtype,bitcast,allow_buffer_view),
    'out(data[0].cast(*data[1:]))',
  ),
)
Deno.test(
  'LazyBuffer.copy_to_device',
  compare(
    [
      [new LazyBuffer(`PYTHON`, new ShapeTracker([new View({ shape:[1], strides:[0], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), Ops.EMPTY, undefined, [], undefined, undefined), `CLANG`, false, false],
      [new LazyBuffer(`PYTHON`, new ShapeTracker([new View({ shape:[2], strides:[1], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:6, itemsize:4, name:`unsigned int`, fmt:`I`, count:1, _scalar:undefined }), Ops.EMPTY, undefined, [], undefined, new Metadata(`uniform`, ``, false)), `CLANG`, false, false],
      [new LazyBuffer(`PYTHON`, new ShapeTracker([new View({ shape:[1], strides:[0], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:6, itemsize:4, name:`unsigned int`, fmt:`I`, count:1, _scalar:undefined }), Ops.EMPTY, undefined, [], undefined, new Metadata(`uniform`, ``, false)), `CLANG`, false, false],
      [new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/684fe191a76d9107981692e6f0d3842f.gunzip`, new ShapeTracker([new View({ shape:[60000], strides:[1], offset:8, mask:undefined, contiguous:false })]), new DType({ priority:2, itemsize:1, name:`unsigned char`, fmt:`B`, count:1, _scalar:undefined }), undefined, undefined, [], new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/684fe191a76d9107981692e6f0d3842f.gunzip`, new ShapeTracker([new View({ shape:[60008], strides:[1], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:2, itemsize:1, name:`unsigned char`, fmt:`B`, count:1, _scalar:undefined }), Ops.EMPTY, undefined, [], undefined, new Metadata(`from_url`, ``, false)), new Metadata(`__getitem__`, ``, false)), `CLANG`, false, false],
      [new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/7b1e157ab43d54d0946f76d1bb6df574.gunzip`, new ShapeTracker([new View({ shape:[10000], strides:[1], offset:8, mask:undefined, contiguous:false })]), new DType({ priority:2, itemsize:1, name:`unsigned char`, fmt:`B`, count:1, _scalar:undefined }), undefined, undefined, [], new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/7b1e157ab43d54d0946f76d1bb6df574.gunzip`, new ShapeTracker([new View({ shape:[10008], strides:[1], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:2, itemsize:1, name:`unsigned char`, fmt:`B`, count:1, _scalar:undefined }), Ops.EMPTY, undefined, [], undefined, new Metadata(`from_url`, ``, false)), new Metadata(`__getitem__`, ``, false)), `CLANG`, false, false],
      [new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/40a9ec2164c7887e8a5f1225de927907.gunzip`, new ShapeTracker([new View({ shape:[10000, 1, 28, 28], strides:[784, 0, 28, 1], offset:16, mask:undefined, contiguous:false })]), new DType({ priority:2, itemsize:1, name:`unsigned char`, fmt:`B`, count:1, _scalar:undefined }), undefined, undefined, [], new LazyBuffer(`DISK:/Users/karel/Library/Caches/tinygrad/downloads/40a9ec2164c7887e8a5f1225de927907.gunzip`, new ShapeTracker([new View({ shape:[7840016], strides:[1], offset:0, mask:undefined, contiguous:true })]), new DType({ priority:2, itemsize:1, name:`unsigned char`, fmt:`B`, count:1, _scalar:undefined }), Ops.EMPTY, undefined, [], undefined, new Metadata(`from_url`, ``, false)), new Metadata(`reshape`, ``, false)), `CLANG`, false, false]
    ],
    (x:LazyBuffer,device:string,force:boolean,clone:boolean) => x.copy_to_device(device,force,clone),
    'out(data[0].copy_to_device(*data[1:]))',
  ),
)
Deno.test(
  'LazyBuffer.alu',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)
Deno.test(
  'LazyBuffer._reduce_op',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)
Deno.test(
  'LazyBuffer.r',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)
Deno.test(
  'LazyBuffer._view',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)
