import type { BufferSpec, Program } from "../jsgrad/device.ts";
import {
  ArrayMap,
  bytes_to_string,
  DefaultMap,
  Device,
  env,
  MemoryView,
  string_to_bytes,
} from "../jsgrad/mod.ts";
import {
  BatchRequest,
  BufferAlloc,
  BufferFree,
  CopyIn,
  CopyOut,
  ProgramAlloc,
  ProgramExec,
  ProgramFree,
} from "../jsgrad/runtime/ops_cloud.ts";
import { bin, install, Tunnel } from "npm:cloudflared";
import { parseArgs, z } from "./parse.ts";

const args = parseArgs({
  port: z.number().default(8080).describe("Port"),
  hostname: z.string().default("0.0.0.0").describe("Hostname"),
  tunnel: z.boolean().optional().describe(
    "Starts a publicly accessible Cloudflare tunnel",
  ),
});

class CloudSession {
  programs = new ArrayMap<[string, string], Program>();
  // TODO: the buffer should track this internally
  buffers = new Map<number, [any, number, BufferSpec?]>();
}

const sessions = new DefaultMap<string, CloudSession>(
  undefined,
  () => new CloudSession(),
);
let device: string = Device.DEFAULT.startsWith("CLOUD")
  ? env.get("CLOUDDEV", "METAL")!
  : Device.DEFAULT;
await Device.get(device).init();

const headers = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
};
Deno.serve(
  {
    port: args.port,
    hostname: args.hostname,
    onListen: async ({ port, hostname }) => {
      const host = `http://${hostname}:${port}`;
      console.log(`start cloud server on ${host} with device ${device}`);

      if (args.tunnel) {
        console.log(`starting cloudflare tunnel...`);
        if (!await env.stat(bin).catch(() => false)) {
          console.log(`installing cloudflare binary to ${bin}`);
          await install(bin);
        }
        const tunnel = Tunnel.quick(host);
        const url = await new Promise((r) => tunnel.once("url", r));
        console.log(`cloudflare url: ${url}`);
      }
    },
  },
  async (req: Request, info) => {
    console.log(req.method, req.url);
    if (req.method === "POST" && req.url.includes("/batch")) {
      const session = sessions.get(
        new URL(req.url).searchParams.get("session")!,
      );
      const r = BatchRequest.deserialize(
        new Uint8Array(await req.arrayBuffer()),
      );
      let ret: Uint8Array | undefined = undefined;
      for (const c of r._q) {
        if (env.DEBUG > 1) console.log(c);
        if (c instanceof BufferAlloc) {
          if (session.buffers.has(c.buffer_num)) {
            throw new Error(`buffer ${c.buffer_num} already allocated`);
          }
          session.buffers.set(c.buffer_num, [
            Device.get(device).allocator!.alloc(c.size, c.options),
            c.size,
            c.options,
          ]);
        } else if (c instanceof BufferFree) {
          const [buf, sz, buffer_options] = session.buffers.get(c.buffer_num)!;
          Device.get(device).allocator!.free(buf, sz, buffer_options);
          session.buffers.delete(c.buffer_num);
        } else if (c instanceof CopyIn) {
          Device.get(device).allocator!._copyin(
            session.buffers.get(c.buffer_num)![0],
            new MemoryView(r._h[c.datahash]),
          );
        } else if (c instanceof CopyOut) {
          const [buf, sz, _] = session.buffers.get(c.buffer_num)!;
          ret = new Uint8Array(sz);
          await Device.get(device).allocator!._copyout(
            new MemoryView(ret),
            buf,
          );
        } else if (c instanceof ProgramAlloc) {
          const lib = await Device.get(device).compiler.compile_cached(
            bytes_to_string(r._h[c.datahash]),
          );
          session.programs.set(
            [c.name, c.datahash],
            await (Device.get(device).runtime!).init(c.name, lib),
          );
        } else if (c instanceof ProgramFree) {
          session.programs.delete([c.name, c.datahash]);
        } else if (c instanceof ProgramExec) {
          const bufs = c.bufs.map((x) => session.buffers.get(x)![0]);
          const r = await session.programs.get([c.name, c.datahash])!.call(
            bufs,
            {
              global_size: c.global_size,
              local_size: c.local_size,
              vals: c.vals,
            },
            !!c.wait,
          );
          if (r !== undefined) ret = string_to_bytes(r.toString());
        } else return new Response(`Unknown instance ${c}`, { status: 400 });
      }
      return new Response(ret, { headers });
    }
    if (req.method === "GET" && req.url.includes("/renderer")) {
      console.log(`connection established with ${info.remoteAddr.hostname}`);
      return new Response(
        JSON.stringify([
          "",
          Device.get(device!).renderer!.constructor.name,
          "",
        ]),
        { headers },
      );
    }
    return new Response("Not found", { status: 400 });
  },
);
