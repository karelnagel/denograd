import esbuild from 'npm:esbuild'
import ts from 'npm:typescript'

const MOD = './jsgrad/mod.ts', WEB = './jsgrad/web.ts', BASE = './jsgrad/base.ts'
await Deno.remove('./dist', { recursive: true })
await esbuild.build({
  entryPoints: [MOD],
  format: 'esm',
  outdir: 'dist',
  bundle: true,
  platform: 'node',
  logLevel: 'error',
  minify: true,
  sourcemap: true,
  target: ['esnext'],
  external: [
    'bun:ffi',
    'bun:sqlite',
    'ffi-rs',
  ],
})

await esbuild.build({
  entryPoints: [WEB, BASE],
  format: 'esm',
  outdir: 'dist',
  bundle: true,
  platform: 'browser',
  logLevel: 'error',
  minify: true,
  sourcemap: true,
  target: ['chrome100'],
  external: [],
})

// tsc
const program = ts.createProgram({
  rootNames: [MOD, WEB, BASE],
  options: {
    declaration: true,
    emitDeclarationOnly: true,
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ESNext,
    allowImportingTsExtensions: true,
    isolatedModules: true,
    moduleResolution: ts.ModuleResolutionKind.NodeNext,
    outDir: 'dist/types',
    skipLibCheck: true,
  },
})
const emitResult = program.emit()
if (emitResult.emitSkipped || emitResult.diagnostics.length > 0) throw new Error(`Type declaration generation failed: ${JSON.stringify(emitResult.diagnostics.map((x) => x.messageText))}`)

// package.json
const version = JSON.parse(await Deno.readTextFile('deno.json')).version
const packageJson = {
  name: '@jsgrad/jsgrad',
  version,
  type: 'module',
  publishConfig: {
    access: 'public',
  },
  main: './mod.js',
  types: './types/jsgrad/mod.d.ts',
  exports: {
    '.': {
      node: { import: './mod.js', types: './types/jsgrad/mod.d.ts' },
      browser: { import: './web.js', types: './types/jsgrad/web.d.ts' },
      default: { import: './mod.js', types: './types/jsgrad/mod.d.ts' },
    },
    './web': { import: './web.js', types: './types/jsgrad/web.d.ts' },
    './base': { import: './base.js', types: './types/jsgrad/base.d.ts' },
  },
}
await Deno.writeTextFile('dist/package.json', JSON.stringify(packageJson, null, 2))
