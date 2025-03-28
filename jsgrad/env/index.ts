import { vars } from '../helpers.ts'
import type { BrowserEnv } from './browser.ts'

let _env: BrowserEnv | undefined

export const env = new Proxy<BrowserEnv>({} as BrowserEnv, {
  get(target, prop) {
    if (_env === undefined) throw new Error('EnvironmentError: setEnv must be called before accessing env')
    return _env[prop as keyof BrowserEnv]
  },
})

export const setEnv = (e: BrowserEnv) => {
  _env = e
  if (vars.DEBUG === 1) console.log(`Using env ${e.NAME}`)
}
