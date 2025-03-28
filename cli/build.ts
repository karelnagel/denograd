import esbuild from 'npm:esbuild'
import ts from 'npm:typescript'

type Entry = { name: string; entry: string }
const MOD = { name: 'mod', entry: './jsgrad/mod.ts' }, BASE = { name: 'base', entry: './jsgrad/base.ts' }
const ENVS = [...Deno.readDirSync(`./jsgrad/env/exports`)].map((x) => ({ name: x.name.replace('.ts', ''), entry: `./jsgrad/env/exports/${x.name}` }))
const ALL: Entry[] = [...ENVS, MOD, BASE]
await Deno.remove('./dist', { recursive: true }).catch(() => {})

// Build
for (const { entry, name } of ALL) {
  await esbuild.build({
    entryPoints: [entry],
    format: 'esm',
    outdir: 'dist',
    bundle: true,
    platform: name === 'browser' ? 'browser' : 'node',
    logLevel: 'error',
    minify: true,
    sourcemap: true,
    target: [name === 'browser' ? 'chrome100' : 'esnext'],
    external: ['bun:ffi', 'bun:sqlite', 'ffi-rs'],
  })
}

// tsc
const program = ts.createProgram({
  rootNames: ALL.map((x) => x.entry),
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
const jsFile = ({ name }: Entry) => `./${name}.js`
const typesFile = ({ entry }: Entry) => entry.replace('./', './types/').replace('.ts', '.d.ts')
const packageJson = {
  name: '@jsgrad/jsgrad',
  version,
  type: 'module',
  publishConfig: { access: 'public' },
  main: jsFile(MOD),
  types: typesFile(MOD),
  exports: {
    ...Object.fromEntries(ALL.map((x) => [`./${x.name}`, { import: jsFile(x), types: typesFile(x) }])),
    '.': Object.fromEntries([...ENVS, { name: 'default', entry: MOD.entry }].map((x) => [x.name, { import: jsFile(x), types: typesFile(x) }])),
  },
}
await Deno.writeTextFile('dist/package.json', JSON.stringify(packageJson, null, 2))
