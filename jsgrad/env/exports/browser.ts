import { BrowserEnv } from '../browser.ts'
import { setEnv } from '../index.ts'
setEnv(new BrowserEnv())
export * from '../../base.ts'
