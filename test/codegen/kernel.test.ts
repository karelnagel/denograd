// deno-fmt-ignore-file
import { _assert_valid_uop, Kernel, Opt, OptOps, verify_ast } from '../../src/codegen/kernel.ts'
import { KernelInfo, UOp } from '../../src/ops.ts'
import { compare, tryCatch } from '../helpers.ts'
import { Renderer } from '../../src/renderer/index.ts'
import { DType, PtrDType } from '../../src/dtype.ts'
import { ShapeTracker } from '../../src/shape/shapetracker.ts'
import { View } from '../../src/shape/view.ts'
import { ClangRenderer } from '../../src/renderer/cstyle.ts'

Deno.test(
  'Opt.real_axis',
  compare(
    [
      [new Opt(OptOps.UNROLL, 0, 0), new Kernel(new UOp({ op:1, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:34, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:0 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[512, 1], strides:[1, 0], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:22, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:33, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:1 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[512, 10], strides:[10, 1], offset:0, mask:undefined, contiguous:true })]) })], arg:undefined })], arg:[40, [1]] })], arg:undefined })], arg:undefined }), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 4), new Kernel(new UOp({ op:1, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:34, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:0 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[32], strides:[1], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:52, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:19, dtype:new DType({ priority:0, itemsize:1, name:`bool`, fmt:`?`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[32], strides:[0], offset:0, mask:undefined, contiguous:false })]) })], arg:undefined }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 })], arg:undefined })], arg:undefined })], arg:undefined }), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 4), new Kernel(new UOp({ op:1, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:34, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:0 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[64], strides:[1], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:52, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:19, dtype:new DType({ priority:0, itemsize:1, name:`bool`, fmt:`?`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[64], strides:[0], offset:0, mask:undefined, contiguous:false })]) })], arg:undefined }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 })], arg:undefined })], arg:undefined })], arg:undefined }), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 3), new Kernel(new UOp({ op:1, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:34, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:0 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[10, 576], strides:[576, 1], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:52, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:19, dtype:new DType({ priority:0, itemsize:1, name:`bool`, fmt:`?`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[10, 576], strides:[0, 0], offset:0, mask:undefined, contiguous:false })]) })], arg:undefined }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 })], arg:undefined })], arg:undefined })], arg:undefined }), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 4), new Kernel(new UOp({ op:1, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:34, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:0 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[32, 1, 5, 5], strides:[25, 0, 5, 1], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:52, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:19, dtype:new DType({ priority:0, itemsize:1, name:`bool`, fmt:`?`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[32, 1, 5, 5], strides:[0, 0, 0, 0], offset:0, mask:undefined, contiguous:false })]) })], arg:undefined }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 })], arg:undefined })], arg:undefined })], arg:undefined }), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 3), new Kernel(new UOp({ op:1, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:34, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:0 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[64, 32, 3, 3], strides:[288, 9, 3, 1], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:52, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:19, dtype:new DType({ priority:0, itemsize:1, name:`bool`, fmt:`?`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[64, 32, 3, 3], strides:[0, 0, 0, 0], offset:0, mask:undefined, contiguous:false })]) })], arg:undefined }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 })], arg:undefined })], arg:undefined })], arg:undefined }), new ClangRenderer())]
    ],
    (opt: Opt, k: Kernel) => opt.real_axis(k),
    'out(data[0].real_axis(data[1]))',
  ),
)

Deno.test(
  'Kernel.init',
  compare(
    [
      [new UOp({ op:1, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:34, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:0 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[1], strides:[0], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:52, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:19, dtype:new DType({ priority:0, itemsize:1, name:`bool`, fmt:`?`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[1], strides:[0], offset:0, mask:undefined, contiguous:true })]) })], arg:undefined }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:1.0 }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 })], arg:undefined })], arg:undefined })], arg:undefined }), new ClangRenderer()],
      [new UOp({ op:1, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:34, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:0 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[512, 1], strides:[1, 0], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:22, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:33, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:1 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[512, 10], strides:[10, 1], offset:0, mask:undefined, contiguous:true })]) })], arg:undefined })], arg:[40, [1]] })], arg:undefined })], arg:undefined }), new ClangRenderer()],
      [new UOp({ op:1, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:34, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:0 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[32], strides:[1], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:52, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:19, dtype:new DType({ priority:0, itemsize:1, name:`bool`, fmt:`?`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[32], strides:[0], offset:0, mask:undefined, contiguous:false })]) })], arg:undefined }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 })], arg:undefined })], arg:undefined })], arg:undefined }), new ClangRenderer()],
      [new UOp({ op:1, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:34, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:0 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[10], strides:[1], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:52, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:19, dtype:new DType({ priority:0, itemsize:1, name:`bool`, fmt:`?`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[10], strides:[0], offset:0, mask:undefined, contiguous:false })]) })], arg:undefined }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 })], arg:undefined })], arg:undefined })], arg:undefined }), new ClangRenderer()],
      [new UOp({ op:1, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:34, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:0 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[64], strides:[1], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:52, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:19, dtype:new DType({ priority:0, itemsize:1, name:`bool`, fmt:`?`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[64], strides:[0], offset:0, mask:undefined, contiguous:false })]) })], arg:undefined }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 })], arg:undefined })], arg:undefined })], arg:undefined }), new ClangRenderer()],
      [new UOp({ op:1, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:34, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:0 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[10, 576], strides:[576, 1], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:52, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:19, dtype:new DType({ priority:0, itemsize:1, name:`bool`, fmt:`?`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[10, 576], strides:[0, 0], offset:0, mask:undefined, contiguous:false })]) })], arg:undefined }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 })], arg:undefined })], arg:undefined })], arg:undefined }), new ClangRenderer()]
    ],
    (ast: UOp, opts: Renderer) => {
      const k = new Kernel(ast, opts)
      return {
        ast: k.ast,
        opts: k.opts,
        vars: k.vars,
        bufs: k.bufs,
        applied_opts: k.applied_opts,
        group_for_reduces: k.group_for_reduces,
        upcasted: k.upcasted,
        local_dims: k.local_dims,
        tensor_core: k.tensor_core,
        tensor_core_opts: k.tensor_core_opts,
        use_tensor_cores: k.use_tensor_cores,
        bufs_for_tensor_core: k.bufs_for_tensor_core,
        dont_use_locals: k.dont_use_locals,
        sts: k.sts,
        reduceops:k.reduceops,
        full_buf_index: k.full_buf_index,
      }
    },
  `k = tiny.codegen.kernel.Kernel(*data)
out({
    "ast":k.ast,
    "opts":k.opts,
    "vars":k.vars,
    "bufs":k.bufs,
    "applied_opts":k.applied_opts,
    "group_for_reduces":k.group_for_reduces,
    "upcasted":k.upcasted,
    "local_dims":k.local_dims,
    "tensor_core":k.tensor_core,
    "tensor_core_opts":k.tensor_core_opts,
    "use_tensor_cores":k.use_tensor_cores,
    "bufs_for_tensor_core":k.bufs_for_tensor_core,
    "dont_use_locals":k.dont_use_locals,
    "sts":k.sts,
    "reduceops":k.reduceops,
    "full_buf_index":k.full_buf_index,
})
`,
  ),
)

Deno.test(
  'Kernel.membufs',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.float4_axis',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.upcasted_axis',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.first_reduce',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.colors',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.colored_shape',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.reshape_and_permute',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.shift_to',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.simplify_ones',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.simplify_merge_adjacent',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.apply_opt',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.required_optimizations',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.hand_coded_optimizations',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.name',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)

Deno.test(
  'Kernel.get_optimized_ast',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)

Deno.test(
  'Kernel.linearize',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)

Deno.test(
  'Kernel.to_program',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)

Deno.test(
  '_assert_valid_uop',
  compare(
    [
      [new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]) }), new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]), new Map<UOp,ShapeTracker>([])],
      [new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[1], strides:[0], offset:0, mask:undefined, contiguous:true })]) }), new ShapeTracker([new View({ shape:[1], strides:[0], offset:0, mask:undefined, contiguous:true })]), new Map<UOp,ShapeTracker>([])],
      [new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[5], strides:[1], offset:0, mask:undefined, contiguous:true })]) }), new ShapeTracker([new View({ shape:[5], strides:[1], offset:0, mask:undefined, contiguous:true })]), new Map<UOp,ShapeTracker>([])],
      [new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[32], strides:[1], offset:0, mask:undefined, contiguous:true })]) }), new ShapeTracker([new View({ shape:[32], strides:[1], offset:0, mask:undefined, contiguous:true })]), new Map<UOp,ShapeTracker>([])],
      [new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[10], strides:[1], offset:0, mask:undefined, contiguous:true })]) }), new ShapeTracker([new View({ shape:[10], strides:[1], offset:0, mask:undefined, contiguous:true })]), new Map<UOp,ShapeTracker>([])],
      [new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[16], strides:[1], offset:0, mask:undefined, contiguous:true })]) }), new ShapeTracker([new View({ shape:[16], strides:[1], offset:0, mask:undefined, contiguous:true })]), new Map<UOp,ShapeTracker>([])]
    ],
    _assert_valid_uop,
    'out(tiny.codegen.kernel._assert_valid_uop(*data))',
  ),
)

Deno.test(
  'verify_ast',
  compare(
    [
      [new UOp({ op:1, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:34, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:0 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[1], strides:[0], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:52, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:19, dtype:new DType({ priority:0, itemsize:1, name:`bool`, fmt:`?`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[1], strides:[0], offset:0, mask:undefined, contiguous:true })]) })], arg:undefined }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:1.0 }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 })], arg:undefined })], arg:undefined })], arg:undefined })],
      [new UOp({ op:1, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:34, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:0 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[512, 1], strides:[1, 0], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:22, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:33, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:1 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[512, 10], strides:[10, 1], offset:0, mask:undefined, contiguous:true })]) })], arg:undefined })], arg:[40, [1]] })], arg:undefined })], arg:undefined })],
      [new UOp({ op:1, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:34, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:0 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[10], strides:[1], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:52, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:19, dtype:new DType({ priority:0, itemsize:1, name:`bool`, fmt:`?`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[10], strides:[0], offset:0, mask:undefined, contiguous:false })]) })], arg:undefined }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 })], arg:undefined })], arg:undefined })], arg:undefined })],
      [new UOp({ op:1, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:34, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:0 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[32], strides:[1], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:52, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:19, dtype:new DType({ priority:0, itemsize:1, name:`bool`, fmt:`?`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[32], strides:[0], offset:0, mask:undefined, contiguous:false })]) })], arg:undefined }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 })], arg:undefined })], arg:undefined })], arg:undefined })],
      [new UOp({ op:1, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:34, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:0 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[64], strides:[1], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:52, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:19, dtype:new DType({ priority:0, itemsize:1, name:`bool`, fmt:`?`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[64], strides:[0], offset:0, mask:undefined, contiguous:false })]) })], arg:undefined }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 })], arg:undefined })], arg:undefined })], arg:undefined })],
      [new UOp({ op:1, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:34, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[new UOp({ op:14, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:0 }), new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:52, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:19, dtype:new DType({ priority:0, itemsize:1, name:`bool`, fmt:`?`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:-1, itemsize:0, name:`void`, fmt:undefined, count:1, _scalar:undefined }), src:[], arg:new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]) })], arg:undefined }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:1.0 }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.0 })], arg:undefined })], arg:undefined })], arg:new KernelInfo(0, 0, false) })]
    ],
    verify_ast,
    'out(tiny.codegen.kernel.verify_ast(*data))',
  ),
)
