import { DenoEnv } from '../deno.ts'
import { setEnv } from '../index.ts'
setEnv(new DenoEnv())
export * from '../../base.ts'
