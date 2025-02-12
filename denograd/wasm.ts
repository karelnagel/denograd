import wabtInit from 'npm:wabt'

const wabt = await wabtInit()

async function compileWATtoWASM(watSource: string) {
  const parsedModule = wabt.parseWat('inline.wat', watSource)
  parsedModule.validate()
  const { buffer } = parsedModule.toBinary({
    log: false,
    write_debug_names: true,
  })
  parsedModule.destroy()
  return buffer
}

const code = `
(module
  (import "env" "memory" (memory 1))

  (func (export "add_arrays")
    (param $ptrA i32)
    (param $ptrB i32)
    (param $ptrRes i32)
    (param $length i32)
    (local $i i32)

    block $break
      loop $loop
        local.get $i
        local.get $length
        i32.eq
        br_if $break

        local.get $ptrRes
        local.get $i
        i32.const 2
        i32.shl
        i32.add

        local.get $ptrA
        local.get $i
        i32.const 2
        i32.shl
        i32.add
        i32.load

        local.get $ptrB
        local.get $i
        i32.const 2
        i32.shl
        i32.add
        i32.load

        i32.add
        i32.store

        local.get $i
        i32.const 1
        i32.add
        local.set $i

        br $loop
      end
    end
  )
)
`

const wasmBytes = await compileWATtoWASM(code)
// Create your own memory in JS
const memory = new WebAssembly.Memory({ initial: 1 })

// Instantiate the WASM module, passing in the memory
const wasmModule = new WebAssembly.Module(wasmBytes)
const wasmInstance = new WebAssembly.Instance(wasmModule, {
  env: { memory },
})

// Now the module will use this memory
const { add_arrays } = wasmInstance.exports as any

// Create a view so you can read/write the memory
const memI32 = new Uint8Array(memory.buffer)

// Put your data at some known offsets
const a = [10, 0, 0, 0, 20, 0, 0, 0, 30, 0, 0, 0, 40, 0, 0, 0]
const b = [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]

const A_OFFSET_BYTES = 0
const B_OFFSET_BYTES = a.length
const RES_OFFSET_BYTES = a.length + b.length

// Write input arrays into memory
memI32.set(a, A_OFFSET_BYTES)
memI32.set(b, B_OFFSET_BYTES)

// Call the function
add_arrays(A_OFFSET_BYTES, B_OFFSET_BYTES, RES_OFFSET_BYTES, a.length)

// Read the result
const result = memI32.slice(RES_OFFSET_BYTES, RES_OFFSET_BYTES + a.length)

console.log(new Int32Array(result.buffer)) // [11, 22, 33, 44]
