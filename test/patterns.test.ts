import { expect } from '@std/expect'
import { _substitute, merge_views, Ops, type PatternMatcher, renderer, spec, symbolic_flat, view_left } from '../denograd/ops.ts'
import { python, test } from './helpers.ts'
import { base_rewrite, ClangRenderer, extra_pm } from '../denograd/renderer/cstyle.ts'
import { entries } from '../denograd/helpers.ts'
import { symbolic_simple } from '../denograd/ops.ts'
import { symbolic } from '../denograd/ops.ts'
import { make_basic_blocks, pm_block_merge } from '../denograd/codegen/linearize.ts'
import { pm_lowerer } from '../denograd/codegen/lowerer.ts'
import { devectorize, expander, float4_folding, get_late_rewrite_patterns, load_store_indexing, migrate_indexing, mulacc_unrolled, pm_render, sym } from '../denograd/ops.ts'
import { break_sched, create_ctx, do_realize, multioutput, remove_movement_ops, sym as schedule_sym, tensor_uop_spec, to_si, view_right } from '../denograd/engine/schedule.ts'
import { pm_gradient } from '../denograd/gradient.ts'
import { si_lowerer } from '../denograd/engine/realize.ts'
import { wgsl_matcher, WGSLRenderer } from '../denograd/renderer/wgsl.ts'

const ALL_PATTERN_MATCHERS: Record<string, PatternMatcher<any, any>> = {
  'tiny.gradient.pm_gradient': pm_gradient,
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
  'tiny.codegen.rewriter.float4_folding': float4_folding,
  'tiny.codegen.rewriter.get_late_rewrite_patterns((Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.AND, Ops.SHL, Ops.NEG, Ops.MULACC))': get_late_rewrite_patterns([Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.AND, Ops.SHL, Ops.NEG, Ops.MULACC]),
  'tiny.codegen.rewriter.sym': sym,
  'tiny.codegen.rewriter.expander': expander,
  'tiny.codegen.rewriter.devectorize': devectorize,
  'tiny.codegen.rewriter.load_store_indexing': load_store_indexing,
  'tiny.codegen.rewriter.migrate_indexing': migrate_indexing,
  'tiny.codegen.rewriter.pm_render': pm_render,
  'tiny.engine.schedule.view_right': view_right,
  'tiny.engine.schedule.to_si': to_si,
  'tiny.engine.schedule.multioutput': multioutput,
  'tiny.engine.schedule.sym': schedule_sym,
  'tiny.engine.schedule.do_realize': do_realize,
  'tiny.engine.schedule.break_sched': break_sched,
  'tiny.engine.realize.si_lowerer': si_lowerer,
  'tiny.engine.schedule.tensor_uop_spec': tensor_uop_spec,
  'tiny.engine.schedule.create_ctx': create_ctx,
  'tiny.engine.schedule.remove_movement_ops': remove_movement_ops,
  'tiny.renderer.wgsl.wgsl_matcher': wgsl_matcher,
  'tiny.renderer.cstyle.ClangRenderer().extra_matcher': new ClangRenderer().extra_matcher,
  'tiny.renderer.wgsl.WGSLRenderer().string_rewrite': new WGSLRenderer().string_rewrite,
  'tiny.codegen.rewriter.mulacc_unrolled': mulacc_unrolled,
}

for (const [name, matcher] of entries(ALL_PATTERN_MATCHERS)) {
  let py: any
  for (let i = 0; i < matcher.patterns.length; i++) {
    test(`patterns_${name}_${i}`, async () => {
      const ts = matcher.patterns[i][0]
      if (!py) py = await python(`out(${name}.patterns)`)
      expect(ts.toString()).toEqual(py[i][0].toString())
    })
  }
}
