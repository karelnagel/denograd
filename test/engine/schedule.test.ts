// // deno-fmt-ignore-file
// import { DType, PtrDType } from '../../src/dtype.ts'
// import { create_schedule_with_vars, init_big_graph, realize_view, ScheduleContext, to_uop } from '../../src/engine/schedule.ts'
// import { Metadata } from '../../src/helpers.ts'
// import { UOp } from '../../src/ops.ts'
// import { ShapeTracker } from '../../src/shape/shapetracker.ts'
// import { View } from '../../src/shape/view.ts'
// import { compare, tryCatch } from '../helpers.ts'

// Deno.test(
//   'to_uop',
//   compare(
//     [
//     ],
//     tryCatch(to_uop),
//     'out(tiny.engine.schedule.to_uop(*data))',
//   ),
// )

// Deno.test(
//   'apply_swizzle',
//   compare(
//     [],
//     () => {},
//     'out(XXX)',
//   ),
// )

// Deno.test(
//   'push_swizzle_down_through_elementwise',
//   compare(
//     [],
//     () => {},
//     'out(XXX)',
//   ),
// )

// Deno.test(
//   '_append_st_vars',
//   compare(
//     [],
//     () => {},
//     'out(XXX)',
//   ),
// )

// Deno.test(
//   'full_ast_rewrite',
//   compare(
//     [],
//     () => {},
//     'out(XXX)',
//   ),
// )

// Deno.test(
//   'uval',
//   compare(
//     [],
//     () => {},
//     'out(XXX)',
//   ),
// )

// Deno.test(
//   'recursive_group',
//   compare(
//     [],
//     () => {},
//     'out(XXX)',
//   ),
// )

// Deno.test(
//   'get_isolated_children',
//   compare(
//     [],
//     () => {},
//     'out(XXX)',
//   ),
// )

// Deno.test(
//   'group_realizes',
//   compare(
//     [],
//     () => {},
//     'out(XXX)',
//   ),
// )

// Deno.test(
//   '_as_const',
//   compare(
//     [],
//     () => {},
//     'out(XXX)',
//   ),
// )

// Deno.test(
//   'realize_view',
//   compare(
//     [
//       [new Map(),new UOp({ op:13, dtype:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), src:[new UOp({ op:15, dtype:new PtrDType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined, _base:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:[83, [`CLANG`, 1, new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined })]] }), new UOp({ op:62, dtype:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), src:[], arg:0 })], arg:new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:13, dtype:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), src:[new UOp({ op:15, dtype:new PtrDType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined, _base:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:[83, [`CLANG`, 1, new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined })]] }), new UOp({ op:62, dtype:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), src:[], arg:0 })], arg:new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]) })], arg:new ShapeTracker([new View({ shape:[512], strides:[0], offset:0, mask:undefined, contiguous:false })]) }), new UOp({ op:62, dtype:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), src:[], arg:0 }), new UOp({ op:15, dtype:new PtrDType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined, _base:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:[83, [`CLANG`, 1, new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined })]] })],
//       [new Map(),new UOp({ op:13, dtype:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), src:[new UOp({ op:15, dtype:new PtrDType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined, _base:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:[88, [`CLANG`, 1, new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined })]] }), new UOp({ op:62, dtype:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), src:[], arg:-1 })], arg:new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:13, dtype:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), src:[new UOp({ op:15, dtype:new PtrDType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined, _base:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:[88, [`CLANG`, 1, new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined })]] }), new UOp({ op:62, dtype:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), src:[], arg:-1 })], arg:new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]) })], arg:new ShapeTracker([new View({ shape:[60000], strides:[0], offset:0, mask:undefined, contiguous:false })]) }), new UOp({ op:62, dtype:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), src:[], arg:-1 }), new UOp({ op:15, dtype:new PtrDType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined, _base:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:[88, [`CLANG`, 1, new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined })]] })],
//       [new Map(),new UOp({ op:13, dtype:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), src:[new UOp({ op:15, dtype:new PtrDType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined, _base:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:[84, [`CLANG`, 1, new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined })]] }), new UOp({ op:62, dtype:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), src:[], arg:60000 })], arg:new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:13, dtype:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), src:[new UOp({ op:15, dtype:new PtrDType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined, _base:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:[84, [`CLANG`, 1, new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined })]] }), new UOp({ op:62, dtype:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), src:[], arg:60000 })], arg:new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]) })], arg:new ShapeTracker([new View({ shape:[512], strides:[0], offset:0, mask:undefined, contiguous:false })]) }), new UOp({ op:62, dtype:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), src:[], arg:60000 }), new UOp({ op:15, dtype:new PtrDType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined, _base:new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:[84, [`CLANG`, 1, new DType({ priority:5, itemsize:4, name:`int`, fmt:`i`, count:1, _scalar:undefined })]] })],
//       [new Map(),new UOp({ op:13, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:15, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:[3, [`CLANG`, 1, new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined })]] }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.9 })], arg:new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:13, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:15, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:[3, [`CLANG`, 1, new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined })]] }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.9 })], arg:new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]) })], arg:new ShapeTracker([new View({ shape:[1], strides:[0], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.9 }), new UOp({ op:15, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:[3, [`CLANG`, 1, new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined })]] })],
//       [new Map(),new UOp({ op:13, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:15, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:[2, [`CLANG`, 1, new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined })]] }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:1.0 })], arg:new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:13, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:15, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:[2, [`CLANG`, 1, new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined })]] }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:1.0 })], arg:new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]) })], arg:new ShapeTracker([new View({ shape:[1], strides:[0], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:1.0 }), new UOp({ op:15, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:[2, [`CLANG`, 1, new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined })]] })],
//       [new Map(),new UOp({ op:13, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:15, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:[99, [`CLANG`, 1, new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined })]] }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.4 })], arg:new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]) }), new UOp({ op:13, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:13, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[new UOp({ op:15, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:[99, [`CLANG`, 1, new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined })]] }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.4 })], arg:new ShapeTracker([new View({ shape:[], strides:[], offset:0, mask:undefined, contiguous:true })]) })], arg:new ShapeTracker([new View({ shape:[32], strides:[0], offset:0, mask:undefined, contiguous:false })]) }), new UOp({ op:62, dtype:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), src:[], arg:0.4 }), new UOp({ op:15, dtype:new PtrDType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined, _base:new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined }), local:false, v:1 }), src:[], arg:[99, [`CLANG`, 1, new DType({ priority:11, itemsize:4, name:`float`, fmt:`f`, count:1, _scalar:undefined })]] })]

//     ],
//     realize_view,
//     'out(XXX)',
//   ),
// )

// Deno.test(
//   'init_big_graph',
//   compare(
//     [],
//     init_big_graph,
//     'out(XXX)',
//   ),
// )

// Deno.test(
//   'create_schedule_with_vars',
//   compare(
//     [
//     ],
//     create_schedule_with_vars,
//     'out(XXX)',
//   ),
// )
