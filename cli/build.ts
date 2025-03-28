import esbuild from 'npm:esbuild'
import ts from 'npm:typescript'

const MOD = './jsgrad/mod.ts', BASE = './jsgrad/base.ts', WEB = './jsgrad/web.ts'
await Deno.remove('./dist', { recursive: true }).catch(() => {})

// Build
for (const entry of [MOD, BASE, WEB]) {
  const isBrowser = entry.endsWith('web.ts')
  await esbuild.build({
    entryPoints: [entry],
    format: 'esm',
    outdir: 'dist',
    bundle: true,
    platform: isBrowser ? 'browser' : 'node',
    logLevel: 'error',
    minify: true,
    splitting: !isBrowser,
    sourcemap: true,
    target: [isBrowser ? 'chrome100' : 'esnext'],
    external: ['bun:ffi', 'bun:sqlite', 'ffi-rs'],
  })
}

// tsc
const program = ts.createProgram({
  rootNames: [MOD, BASE, WEB],
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
const jsFile = (entry: string) => entry.replace('./jsgrad/', './').replace('.ts', '.js')
const typesFile = (entry: string) => entry.replace('./', './types/').replace('.ts', '.d.ts')
const packageJson = {
  name: '@jsgrad/jsgrad',
  version,
  type: 'module',
  publishConfig: { access: 'public' },
  main: jsFile(MOD),
  types: typesFile(MOD),
  exports: {
    './web': { import: jsFile(WEB), types: typesFile(WEB) },
    './base': { import: jsFile(BASE), types: typesFile(BASE) },
    '.': {
      default: { import: jsFile(MOD), types: typesFile(MOD) },
      node: { import: jsFile(MOD), types: typesFile(MOD) },
      browser: { import: jsFile(WEB), types: typesFile(WEB) },
    },
  },
}
await Deno.writeTextFile('dist/package.json', JSON.stringify(packageJson, null, 2))
