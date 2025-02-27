// import { type DType, dtypes } from '../../src/dtype.ts'
// import { Ops, UOp } from '../../src/ops.ts'
// import { ClangRenderer } from '../../src/renderer/cstyle.ts'
// import { CStyleLanguage, uops_to_dtypes } from '../../src/renderer/cstyle.ts'
// import { compare } from '../helpers.ts'

// test(
//   'uops_to_dtypes',
//   compare(
//     [
//       [[UOp.int(444), UOp.int(555), UOp.float(4.4)]],
//       [[UOp.int(1), UOp.bool(true), UOp.float(1.0)]],
//       [[UOp.int(0), UOp.int(0), UOp.bool(false)]],
//       [[UOp.float(1.1), UOp.float(2.2), UOp.float(3.3)]],
//     ],
//     uops_to_dtypes,
//     'out(cstyle.uops_to_dtypes(*data))',
//   ),
// )

// test(
//   'CStyleLanguage',
//   compare(
//     [
//       [['kernel_prefix', 'buffer_prefix', 'buffer_suffix', 'smem_align', 'smem_prefix', 'smem_prefix_for_cast', 'arg_int_prefix', 'barrier', 'float4', 'infinity', 'nan']],
//     ],
//     (keys: string[]) => {
//       const c = new CStyleLanguage()
//       return keys.map((key) => c[key as keyof typeof c])
//     },
//     `out([getattr(cstyle.CStyleLanguage(), key) for key in data[0]])`,
//   ),
// )

// const codeForOpInput: [op: Ops, args: (string | DType)[]][] = [
//   [Ops.SQRT, ['4', dtypes.int]],
//   [Ops.RECIP, ['4', dtypes.int]],
//   [Ops.NEG, ['4', dtypes.int]],
//   [Ops.EXP2, ['4', dtypes.int]],
//   [Ops.LOG2, ['4', dtypes.int]],
//   [Ops.SIN, ['4', dtypes.int]],
//   [Ops.AND, ['4', '5', dtypes.int]],
//   [Ops.XOR, ['4', '5', dtypes.int]],
//   [Ops.OR, ['4', '5', dtypes.int]],
//   [Ops.ADD, ['4', '5', dtypes.int]],
//   [Ops.SUB, ['4', '5', dtypes.int]],
//   [Ops.MUL, ['4', '5', dtypes.int]],
//   [Ops.MOD, ['4', '5', dtypes.int]],
//   [Ops.IDIV, ['4', '5', dtypes.int]],
//   [Ops.CMPNE, ['4', '5', dtypes.int]],
//   [Ops.SHR, ['4', '5', dtypes.int]],
//   [Ops.SHL, ['4', '5', dtypes.int]],
//   [Ops.CMPLT, ['4', '5', dtypes.int]],
//   [Ops.WHERE, ['4', '5', '6', dtypes.int]],
// ]
// test(
//   'CStyleLanguage.code_for_op',
//   compare(
//     codeForOpInput,
//     (op: Ops, args: (string | DType)[]) => new CStyleLanguage().code_for_op[op]?.(...args),
//     'out(cstyle.CStyleLanguage().code_for_op.get(data[0])(*data[1]))',
//   ),
// )

// test(
//   'CStyleLanguage.render_kernel',
//   compare(
//     [
//       [{ functionName: 'kernel1', kernel: ['int x = 0;', 'x += 1;'], bufs: [['buf1', [dtypes.int, true]]], uops: [UOp.int(3)], prefix: undefined }],
//       [{ functionName: 'kernel2', kernel: ['float4 val = read_imagef(img, smp, (int2)(0,0));'], bufs: [['img', [dtypes.imagef(2, 2), false]]], uops: [], prefix: ['#define IMG_SIZE 256'] }],
//       [{ functionName: 'kernel3', kernel: ['*ptr = 42;'], bufs: [['ptr', [dtypes.int.ptr(), true]]], uops: [], prefix: [] }],
//       [{ functionName: 'kernel4', kernel: ['float val = *in;', '*out = val * 2;'], bufs: [['in', [dtypes.float.ptr(), false]], ['out', [dtypes.float.ptr(), true]]], uops: [], prefix: [] }],
//       [{ functionName: 'kernel5', kernel: ['#pragma unroll', 'for(int i=0; i<4; i++) {}'], bufs: [], uops: [], prefix: ['#define UNROLL 4'] }],
//       [{
//         functionName: 'kernel6',
//         kernel: ['const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE;', 'float4 val = read_imagef(img, smp, (int2)(0,0));'],
//         bufs: [['img', [dtypes.imagef(2, 2), false]], ['out', [dtypes.float.ptr(), true]]],
//         uops: [],
//         prefix: [],
//       }],
//       [{ functionName: 'kernel7', kernel: ['int x = 0;'], bufs: [['buf1', [dtypes.int, true]], ['buf2', [dtypes.int, false]]], uops: [], prefix: ['typedef int int32_t;'] }],
//     ],
//     new CStyleLanguage().render_kernel,
//     'out(cstyle.CStyleLanguage().render_kernel(data[0].get("functionName"),list(data[0].get("kernel")),list(data[0].get("bufs")),list(data[0].get("uops")),data[0].get("prefix")))',
//   ),
// )
// test(
//   'CStyleLanguage.render_cast',
//   compare(
//     [
//       [dtypes.int, '42'],
//       [dtypes.float, '3.14'],
//       [dtypes.bool, '1'],
//       [dtypes.int.vec(4), 'x'],
//       [dtypes.float.ptr(), 'ptr'],
//     ],
//     new CStyleLanguage().render_cast,
//     'out(cstyle.CStyleLanguage().render_cast(*data))',
//   ),
// )

// test(
//   'CStyleLanguage.render_dtype',
//   compare(
//     [
//       [dtypes.int],
//       [dtypes.float, false],
//       [dtypes.bool],
//       [dtypes.imagef(2, 2), true],
//       [dtypes.int.ptr()],
//       [dtypes.float.ptr(), false],
//       [dtypes.int.vec(4)],
//       [dtypes.float.vec(2), false],
//     ],
//     new CStyleLanguage().render_dtype,
//     'out(cstyle.CStyleLanguage().render_dtype(*data))',
//   ),
// )

// // KAREL: todo
// // test(
// //   'CStyleLanguage.render',
// //   compare(
// //     [
// //       ['kernel1', [
// //         new UOp(Ops.SPECIAL, undefined, undefined, ['#pragma unroll']),
// //         new UOp(Ops.DEFINE_GLOBAL, dtypes.int, undefined, 0),
// //         new UOp(Ops.STORE, undefined, [new UOp(Ops.DEFINE_GLOBAL, undefined, undefined, 0), new UOp(Ops.CONST, dtypes.int, undefined, 42)]),
// //       ]],
// //       ['kernel2', [
// //         new UOp(Ops.DEFINE_GLOBAL, dtypes.float, undefined, 0),
// //         new UOp(Ops.DEFINE_VAR, dtypes.float, undefined, ['x']),
// //         new UOp(Ops.ASSIGN, undefined, [new UOp(Ops.CONST, dtypes.float, undefined, 3.14)]),
// //       ]],
// //       ['kernel3', [
// //         new UOp(Ops.DEFINE_LOCAL, dtypes.int, undefined, 'temp'),
// //         new UOp(Ops.RANGE, undefined, [new UOp(Ops.CONST, dtypes.int, undefined, 10)]),
// //         new UOp(Ops.ENDRANGE),
// //       ]],
// //       ['kernel4', [
// //         new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), undefined, 0),
// //         new UOp(Ops.LOAD, undefined, [new UOp(Ops.DEFINE_GLOBAL, undefined, undefined, 0)]),
// //         new UOp(Ops.IF, undefined, [new UOp(Ops.CONST, dtypes.bool, undefined, true)]),
// //         new UOp(Ops.ENDIF),
// //       ]],
// //       ['kernel5', [
// //         new UOp(Ops.DEFINE_ACC, dtypes.float, undefined, 'acc',src:[UOp.int(3)]),
// //         new UOp(Ops.SPECIAL, undefined, undefined, ['#pragma unroll']),
// //         new UOp(Ops.GEP, undefined, [new UOp(Ops.DEFINE_GLOBAL, undefined, undefined, 0)], [0]),
// //       ]],
// //     ],
// //     new CStyleLanguage().render,
// //     'out(cstyle.CStyleLanguage().render(*data))',
// //   ),
// // )
// test(
//   'ClangRenderer.code_for_op',
//   compare(
//     [
//       ...codeForOpInput,
//       [Ops.SQRT, ['x', dtypes.float64]],
//       [Ops.SQRT, ['x', dtypes.float32]],
//     ],
//     (op: Ops, args: (string | DType)[]) => new ClangRenderer().code_for_op[op]?.(...args),
//     `
// fn = cstyle.ClangRenderer().code_for_op.get(data[0])
// out(fn(*data[1]) if fn is not None else None)
// `,
//   ),
// )

// // KAREL: figure out env tests
// // test(
// //   'ClangRenderer.tensor_cores',
// //   compare(
// //     [[]],
// //     () => {
// //       process.env.AMX = '2'
// //       new ClangRenderer().tensor_cores
// //     },
// //     'out("")',
// //   ),
// // )

// test(
//   'ClangRenderer.render_vector_prefix',
//   compare(
//     [
//       [dtypes.int],
//       [dtypes.float],
//       [dtypes.bool],
//       [dtypes.imagef(2, 2, 2)],
//       [dtypes.imageh(2, 2, 2)],
//       [dtypes.imageh(2, 2, 2).ptr()],
//       [dtypes.int.vec(3).ptr()],
//     ],
//     new ClangRenderer().render_vector_prefix,
//     'out(cstyle.ClangRenderer().render_vector_prefix(*data))',
//   ),
// )
// // KAREL: todo
// // test(
// //   'ClangRenderer.render',
// //   compare(
// //     [
// //       ['kernel1', [
// //         new UOp(Ops.DEFINE_GLOBAL, dtypes.int, undefined, 0),
// //         new UOp(Ops.STORE, undefined, [new UOp(Ops.DEFINE_GLOBAL, undefined, undefined, 0), new UOp(Ops.CONST, dtypes.int, undefined, 42)]),
// //       ]],
// //       ['kernel2', [
// //         new UOp(Ops.DEFINE_GLOBAL, dtypes.float, undefined, 0),
// //         new UOp(Ops.DEFINE_VAR, dtypes.float, undefined, ['x']),
// //         new UOp(Ops.ASSIGN, undefined, [new UOp(Ops.CONST, dtypes.float, undefined, 3.14)]),
// //       ]],
// //       ['kernel3', [
// //         new UOp(Ops.DEFINE_LOCAL, dtypes.int, undefined, 'temp'),
// //         new UOp(Ops.RANGE, undefined, [new UOp(Ops.CONST, dtypes.int, undefined, 10)]),
// //         new UOp(Ops.ENDRANGE),
// //       ]],
// //       ['kernel4', [
// //         new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), undefined, 0),
// //         new UOp(Ops.LOAD, undefined, [new UOp(Ops.DEFINE_GLOBAL, undefined, undefined, 0)]),
// //         new UOp(Ops.IF, undefined, [new UOp(Ops.CONST, dtypes.bool, undefined, true)]),
// //         new UOp(Ops.ENDIF),
// //       ]],
// //       ['kernel5', [
// //         new UOp(Ops.DEFINE_ACC, dtypes.float, undefined, 'acc'),
// //         new UOp(Ops.SPECIAL, undefined, undefined, ['#pragma unroll']),
// //         new UOp(Ops.GEP, undefined, [new UOp(Ops.DEFINE_GLOBAL, undefined, undefined, 0)], [0]),
// //       ]],
// //     ],
// //     new ClangRenderer().render,
// //     'out(cstyle.ClangRenderer().render(*data))',
// //   ),
// // )
// test(
//   'ClangRenderer.render_kernel',
//   compare(
//     [
//       [{ functionName: 'kernel1', kernel: ['int x = 0;', 'x += 1;'], bufs: [['buf1', [dtypes.int, true]]], uops: [UOp.int(3)], prefix: undefined }],
//       [{ functionName: 'kernel2', kernel: ['float4 val = read_imagef(img, smp, (int2)(0,0));'], bufs: [['img', [dtypes.imagef(2, 2), false]]], uops: [], prefix: ['#define IMG_SIZE 256'] }],
//       [{ functionName: 'kernel3', kernel: ['*ptr = 42;'], bufs: [['ptr', [dtypes.int.ptr(), true]]], uops: [], prefix: [] }],
//       [{ functionName: 'kernel4', kernel: ['float val = *in;', '*out = val * 2;'], bufs: [['in', [dtypes.float.ptr(), false]], ['out', [dtypes.float.ptr(), true]]], uops: [], prefix: [] }],
//       [{ functionName: 'kernel5', kernel: ['#pragma unroll', 'for(int i=0; i<4; i++) {}'], bufs: [], uops: [], prefix: ['#define UNROLL 4'] }],
//       [{
//         functionName: 'kernel6',
//         kernel: ['const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE;', 'float4 val = read_imagef(img, smp, (int2)(0,0));'],
//         bufs: [['img', [dtypes.imagef(2, 2), false]], ['out', [dtypes.float.ptr(), true]]],
//         uops: [],
//         prefix: [],
//       }],
//     ],
//     new ClangRenderer().render_kernel,
//     'out(cstyle.ClangRenderer().render_kernel(data[0].get("functionName"),list(data[0].get("kernel")),list(data[0].get("bufs")),list(data[0].get("uops")),data[0].get("prefix")))',
//   ),
// )
