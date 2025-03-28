import { setEnv } from './env/index.ts'
import { WebEnv } from './env/web.ts'

setEnv(new WebEnv())

export * from './base.ts'
