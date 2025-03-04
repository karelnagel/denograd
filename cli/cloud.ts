import type { BufferSpec, Program } from '../denograd/device.ts'
import { ArrayMap, bytes_to_string, DefaultMap, Device, env, MemoryView, string_to_bytes } from '../denograd/mod.ts'
import { BatchRequest, BufferAlloc, BufferFree, CopyIn, CopyOut, ProgramAlloc, ProgramExec, ProgramFree } from '../denograd/runtime/ops_cloud.ts'

class CloudSession {
  programs = new ArrayMap<[string, string], Program>()
  // TODO: the buffer should track this internally
  buffers = new Map<number, [any, number, BufferSpec?]>()
}

const sessions = new DefaultMap<string, CloudSession>(undefined, () => new CloudSession())
let device: string = Device.DEFAULT.startsWith('CLOUD') ? env.get('CLOUDDEV', 'METAL')! : Device.DEFAULT
await Device.get(device).init()

Deno.serve(
  {
    port: 6667,
    onListen: ({ port, hostname }) => console.log(`start cloud server on http://${hostname}:${port} with device ${device}`),
  },
  async (req: Request, info) => {
    console.log(req.method, req.url)
    if (req.method === 'POST' && req.url.includes('/batch')) {
      const session = sessions.get(req.headers.get('Cookie')!.split('session=')[1])
      const r = BatchRequest.deserialize(new Uint8Array(await req.arrayBuffer()))
      let ret: Uint8Array | undefined = undefined
      for (const c of r._q) {
        if (env.DEBUG > 1) console.log(c)
        if (c instanceof BufferAlloc) {
          if (session.buffers.has(c.buffer_num)) throw new Error(`buffer ${c.buffer_num} already allocated`)
          session.buffers.set(c.buffer_num, [Device.get(device).allocator!.alloc(c.size, c.options), c.size, c.options])
        } else if (c instanceof BufferFree) {
          const [buf, sz, buffer_options] = session.buffers.get(c.buffer_num)!
          Device.get(device).allocator!.free(buf, sz, buffer_options)
          session.buffers.delete(c.buffer_num)
        } else if (c instanceof CopyIn) {
          Device.get(device).allocator!._copyin(session.buffers.get(c.buffer_num)![0], new MemoryView(r._h[c.datahash]))
        } else if (c instanceof CopyOut) {
          const [buf, sz, _] = session.buffers.get(c.buffer_num)!
          ret = new Uint8Array(sz)
          await Device.get(device).allocator!._copyout(new MemoryView(ret), buf)
        } else if (c instanceof ProgramAlloc) {
          const lib = await Device.get(device).compiler.compile_cached(bytes_to_string(r._h[c.datahash]))
          session.programs.set([c.name, c.datahash], await (Device.get(device).runtime!).init(c.name, lib))
        } else if (c instanceof ProgramFree) {
          session.programs.delete([c.name, c.datahash])
        } else if (c instanceof ProgramExec) {
          const bufs = c.bufs.map((x) => session.buffers.get(x)![0])
          const r = await session.programs.get([c.name, c.datahash])!.call(bufs, { global_size: c.global_size, local_size: c.local_size, vals: c.vals }, !!c.wait)
          if (r !== undefined) ret = string_to_bytes(r.toString())
        } else return new Response(`Unknown instance ${c}`, { status: 400 })
      }
      return new Response(ret)
    }
    if (req.method === 'GET' && req.url.includes('/renderer')) {
      console.log(`connection established with ${info.remoteAddr.hostname}`)
      return new Response(JSON.stringify(['', Device.get(device!).renderer!.constructor.name, '']))
    }
    return new Response('Not found', { status: 404 })
  },
)
