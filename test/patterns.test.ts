import { expect } from 'expect/expect'
import { _substitute, merge_views, Ops, type PatternMatcher, renderer, spec, symbolic_flat, view_left } from '../denograd/ops.ts'
import { asdict, python, removeKeys } from './helpers.ts'
import { base_rewrite, extra_pm } from '../denograd/renderer/cstyle.ts'
import { entries, zip } from '../denograd/helpers.ts'
import { symbolic_simple } from '../denograd/ops.ts'
import { symbolic } from '../denograd/ops.ts'
import { make_basic_blocks, pm_block_merge } from '../denograd/codegen/linearize.ts'
import { pm_lowerer } from '../denograd/codegen/lowerer.ts'
import { devectorize, expander, float4_folding, get_late_rewrite_patterns, load_store_indexing, migrate_indexing, pm_render, sym } from '../denograd/codegen/uopgraph.ts'
import { break_sched, do_realize, multioutput, ops_folding, to_si, view_right } from '../denograd/engine/schedule.ts'

const ALL_PATTERN_MATCHERS: Record<string, PatternMatcher<any, any>> = {
  'tiny.ops.spec': spec,
  'tiny.ops.symbolic_simple': symbolic_simple,
  'tiny.ops.symbolic': symbolic,
  'tiny.ops.symbolic_flat': symbolic_flat,
  'tiny.ops._substitute': _substitute,
  'tiny.ops.renderer': renderer,
  'tiny.ops.merge_views': merge_views,
  'tiny.ops.view_left': view_left,
  'tiny.renderer.cstyle.base_rewrite': base_rewrite,
  'tiny.renderer.cstyle.extra_pm': extra_pm,
  'tiny.codegen.linearize.make_basic_blocks': make_basic_blocks,
  'tiny.codegen.linearize.pm_block_merge': pm_block_merge,
  'tiny.codegen.lowerer.pm_lowerer': pm_lowerer,
  'tiny.codegen.uopgraph.float4_folding': float4_folding,
  'tiny.codegen.uopgraph.get_late_rewrite_patterns((Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.AND, Ops.SHL, Ops.NEG, Ops.MULACC))': get_late_rewrite_patterns([Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.AND, Ops.SHL, Ops.NEG, Ops.MULACC]),
  'tiny.codegen.uopgraph.sym': sym,
  'tiny.codegen.uopgraph.expander': expander,
  'tiny.codegen.uopgraph.devectorize': devectorize,
  'tiny.codegen.uopgraph.load_store_indexing': load_store_indexing,
  'tiny.codegen.uopgraph.migrate_indexing': migrate_indexing,
  'tiny.codegen.uopgraph.pm_render': pm_render,
  'tiny.engine.schedule.view_right': view_right,
  'tiny.engine.schedule.to_si': to_si,
  'tiny.engine.schedule.multioutput': multioutput,
  'tiny.engine.schedule.ops_folding': ops_folding,
  'tiny.engine.schedule.do_realize': do_realize,
  'tiny.engine.schedule.break_sched': break_sched,
}

for (const [name, matcher] of entries(ALL_PATTERN_MATCHERS)) {
  for (let i = 0; i < matcher.patterns.length; i++) {
    Deno.test(`patterns_${name}_${i}`, async () => {
      const ts = matcher.patterns[i][0]
      const py = await python(`out(${name}.patterns[${i}][0])`)
      expect(ts.toString()).toEqual(py.toString())
    })
  }
}
