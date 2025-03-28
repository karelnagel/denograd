import { BunEnv } from '../bun.ts'
import { setEnv } from '../index.ts'
setEnv(new BunEnv())
export * from '../../base.ts'
