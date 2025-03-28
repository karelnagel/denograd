import { WebEnv } from './env/web.ts'
import { setEnv } from './env/index.ts'
setEnv(new WebEnv())
export * from './base.ts'
