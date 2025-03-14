import * as c from './mod.ts'
import * as dawn from './dawn.ts'

const desc = new dawn.InstanceDescriptor(new c.Pointer(null), new dawn.InstanceFeatures(new c.Pointer(null), new c.U32(1), new c.U64(10n)))
const instance = dawn.createInstance(desc.ptr())

const feats = new dawn.InstanceFeatures(new c.Pointer(null), new c.U32(9), new c.U64(0n))
const ptr = feats.ptr()
const features = dawn.getInstanceFeatures(ptr)
const f2 = dawn.getInstanceFeatures(ptr)
console.log(ptr.load(feats), features, f2)
