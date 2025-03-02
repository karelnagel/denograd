import { WebEnv } from './web.ts'

export let env = new WebEnv()
export const setRuntime = (r: WebEnv) => {
  env = r
}
if (env.get('DEBUG')) console.log(`Using env ${env.NAME}`)
