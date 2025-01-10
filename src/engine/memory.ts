// // **************** memory planning ****************

import { Buffer, Device } from '../device.ts'
import { DEBUG, dedup, get_key, NO_MEMORY_PLANNER } from '../helpers.ts'
import { Ops } from '../ops.ts'
import { ScheduleItem } from './schedule.ts'

const _internal_memory_planner = (buffers: Buffer[][], noopt_buffers?: Buffer[], debug_prefix = ''): Map<Buffer, Buffer> => {
  if (NO_MEMORY_PLANNER) return new Map()
  const [first_appearance, last_appearance] = [new Map<Buffer, number>(), new Map<Buffer, number>()]
  for (const [i, u] of buffers.entries()) {
    for (const buf of u) {
      if (buf.is_allocated() || buf.lb_refcount > 0 || (noopt_buffers !== undefined && noopt_buffers.includes(buf.base))) continue
      if (!first_appearance.has(buf.base)) first_appearance.set(buf.base, i)
      last_appearance.set(buf.base, i)
    }
  }
  //   // Sort buffers by size in descending order, prioritizing largest buffers for allocation first.
  //   // Track free segments, each containing (start, stop, && buffer that could be reused on this segment).
  type Seg = [number, number, Buffer]
  const free_segs = new Map<string, Seg[]>() // Map<buffer key, [start, end, buffer to reuse on the seg>]
  const find_replace_buffer = (buf: Buffer, st: number, en: number) => {
    const key = get_key(buf.device, buf.dtype, buf.options, ...(!('offset' in Device.get(buf.device).allocator!) ? [buf.nbytes] : []))

    const default_buf: Seg = [0, buffers.length - 1, buf] // will return the buffer itthis if the replace one !== found.
    const next = free_segs.get(key)?.entries()?.filter(([i, [sst, sen]]) => sst <= st && en <= sen).next().value
    const [seg_st, seg_en, seg_buf] = next ? free_segs.get(key)!.splice(next[0], 1)[0] : default_buf

    free_segs.set(key, [...free_segs.get(key) || [], ...(st - 1 >= seg_st ? [[seg_st, st - 1, seg_buf] as Seg] : [])])
    free_segs.set(key, [...free_segs.get(key) || [], ...(seg_en >= en + 1 ? [[en + 1, seg_en, seg_buf] as Seg] : [])])

    return seg_buf.nbytes === buf.nbytes ? seg_buf : new Buffer(buf.device, buf.size, buf.dtype, undefined, undefined, undefined, undefined, seg_buf)
  }
  const buffer_requests = [...first_appearance.keys()].map((buf) => [first_appearance.get(buf), last_appearance.get(buf), buf] as Seg).toSorted((a, b) => b[2].nbytes - a[2].nbytes)
  const assigned = new Map(buffer_requests.map(([st, en, buf]) => [buf, find_replace_buffer(buf, st, en)]))

  for (const [i, u] of buffers.entries()) {
    for (const buf of u) {
      if (buf.is_allocated() || buf.lb_refcount > 0 || (noopt_buffers !== undefined && noopt_buffers.includes(buf.base))) continue
      if (buf._base !== undefined) assigned.set(buf, new Buffer(buf.device, buf.size, buf.dtype, undefined, undefined, undefined, undefined, (assigned.get(buf.base) || buf.base).base, buf.offset))
      else assigned.set(buf, assigned.get(buf) || buf)
    }
  }
  const ak = dedup([...assigned.keys().filter((x) => x._base === undefined)])
  const av = dedup([...assigned.values().filter((x) => x._base === undefined)])
  if (DEBUG >= 1 && ak.length !== av.length) console.log(debug_prefix + `memory reduced from ${ak.reduce((sum, x) => sum + x.nbytes, 0) / 1e6} MB -> ${av.reduce((sum, x) => sum + x.nbytes, 0) / 1e6} MB, ${ak.length} -> ${av.length} bufs`)
  return assigned
}

export const memory_planner = (schedule: ScheduleItem[]): ScheduleItem[] => {
  //   // Exclude buffers involved in load ops (e.g transfers) to preserve parallelism in graphs.
  const assigned = _internal_memory_planner(
    schedule.map((si) => si.bufs),
    schedule.filter((si) => si.ast.op !== Ops.SINK).flatMap((si) => si.bufs),
  )
  return schedule.map((si) => new ScheduleItem(si.ast, si.bufs.map((x) => assigned.get(x) || x), si.metadata, si.assign_preloads))
}
