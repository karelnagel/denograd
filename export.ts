import { CompiledRunner } from './denograd/engine/realize.ts'
import { withEnv, withEnvAsync } from './denograd/env/index.ts'
import { DType, range, Tensor, TinyJit } from './denograd/mod.ts'
import { get_state_dict, Llama3 } from './denograd/mod.ts'
import { Ops, UOp } from './denograd/ops.ts'
import { ProgramSpec } from './denograd/renderer/index.ts'

const compile_net = (run: TinyJit<any, any>, special_names: Record<number, string>): [Record<string, string>, [string, string[], number[]][], Record<string, [number, DType, number]>, Record<string, Tensor>] => {
  let functions = {}, bufs = {}, bufs_to_save = {}, statements = [], bufnum = 0
  for (const ji of run.jit_cache) {
    let fxn: ProgramSpec = (ji.prg as CompiledRunner).p
    functions[fxn.name] = fxn.src // NOTE: this assumes all with the same name are the same
    let cargs = []
    for (const [i, arg] of ji.bufs.entries()) {
      if (!arg) continue
      console.log({ arg })
      const key = arg!.key
      if (!(key in bufs)) {
        if (key in special_names) {
          bufs[key] = [special_names[key], arg!.size * arg!.dtype.itemsize, arg!.dtype, key]
        } else {
          bufs[key] = [`buf_${bufnum}`, arg!.size * arg!.dtype.itemsize, arg!.dtype, key]
          bufnum += 1
          if (i > 0) bufs_to_save[bufs[key][0]] = arg // if first usage of a buffer is not an output, and it's not a special name
        }
      }
      cargs.push(bufs[key][0])
    }
    cargs.push(...fxn.vars.filter((v) => v.op === Ops.DEFINE_VAR)) // symbolic vars; is it necessary or sufficient to check for DEFINE_VAR?
    statements.push([fxn.function_name, cargs, fxn.global_size, fxn.local_size])
  }
  return [functions, statements, Object.fromEntries(Object.values(bufs).map(([name, size, dtype, key]) => [name, [size, dtype, key]])), bufs_to_save]
}

const jit_model = async (model: any, args: any[]): Promise<[TinyJit<any, any>, Record<number, string>]> => {
  if (!model.forward && !model.call) throw new Error('Model needs a forward function')
  const run = new TinyJit(async (...x: any[]) => {
    let out = model.forward ? await model.forward(...x) : await model.call(...x)
    if (!Array.isArray(out) && !(out instanceof Tensor)) throw new Error('model output must be a Tensor, tuple, or a list of Tensors for export')
    out = out instanceof Tensor ? [out] : out
    for (const o of out) await o.realize()
    return out
  })

  // twice to run the JIT
  let the_output
  for (const _ of range(2)) the_output = await run.call(...args)
  const special_names = {}

  // hack to put the inputs back
  for (const [[j, i], idx] of run.input_replace.entries()) {
    const realized_input = args[idx].lazydata.base.realized
    run.jit_cache[j].bufs[i] = realized_input
    console.log(realized_input, realized_input?.key)
    special_names[realized_input] = `input${idx}`
  }

  // TODO: fetch this from the jit in self.input_replace and self.ret (hint: use get_parameters on self.ret)
  for (const [i, output] of the_output.entries()) {
    special_names[output.lazydata.base.realized] = `output${i}`
  }
  return [run, special_names]
}

const export_model = async (model: any, inputs: any[], model_name = 'model') => {
  const [run, special_names] = await withEnvAsync({ JIT: 2 }, async () => await jit_model(model, inputs))
  const [functions, statements, bufs, bufs_to_save] = compile_net(run, special_names)
  Object.values(functions).forEach(console.log)
  // const state = get_state_dict(model.model)
}

Tensor.no_grad = true
const max_context = 1024
const tok = 128000
const TEMPERATURE = 0.95, TOP_K = 0, TOP_P = 0, ALPHA_F = 0, ALPHA_P = 0.0
const start_pos = UOp.variable('start_pos', 0, max_context).bind(0)
const model_input = () => [new Tensor([[tok]]), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P]
const model = await Llama3.load({ size: '1B', quantize: 'float16' })

const res = await export_model(model.model, model_input())
console.log(res)
