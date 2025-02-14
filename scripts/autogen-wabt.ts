/**
 * This is for getting wabt.js, we get it from here https://raw.githubusercontent.com/AssemblyScript/wabt.js/refs/heads/main/index.js
 * We replace `require()` and `__dirname` to make it work in web + export it as a module
 *
 * TODO: can this file be smaller, rn it,s 1.19MB
 */

const URL = 'https://raw.githubusercontent.com/AssemblyScript/wabt.js/refs/heads/main/index.js'
const PATH = './denograd/runtime/autogen/wabt.js'

const res = await fetch(URL)
let text = await res.text()

text = text.replace('var ', 'export const ')

text = text.replaceAll('__dirname', 'undefined')
text = text.replaceAll('require("fs")', 'undefined')
text = text.replaceAll('require("path")', 'undefined')

await Deno.writeTextFile(PATH, text)
