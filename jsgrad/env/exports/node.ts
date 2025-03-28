import { setEnv } from '../index.ts'
import { NodeEnv } from '../node.ts'
setEnv(new NodeEnv())
export * from '../../base.ts'
