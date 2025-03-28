import esbuild from 'npm:esbuild'
import ts from 'npm:typescript'

const NODE = './jsgrad/node.ts', BASE = './jsgrad/base.ts', WEB = './jsgrad/web.ts'
await Deno.remove('./dist', { recursive: true }).catch(() => {})

// Build
for (const entry of [NODE, BASE, WEB]) {
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
  rootNames: [NODE, BASE, WEB],
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

// if deno.json version is updated then push new version otherwise beta
let version = JSON.parse(await Deno.readTextFile('deno.json')).version
const npmVersion = (await fetch('https://registry.npmjs.org/@jsgrad/jsgrad').then((x) => x.json()))['dist-tags'].latest
const beta = version === npmVersion
if (beta) version = `${version}-beta-${new Date().getTime()}`

// package.json
const jsFile = (entry: string) => entry.replace('./jsgrad/', './').replace('.ts', '.js')
const typesFile = (entry: string) => entry.replace('./', './types/').replace('.ts', '.d.ts')
const packageJson = {
  name: '@jsgrad/jsgrad',
  version,
  type: 'module',
  publishConfig: { access: 'public', tag: beta ? 'beta' : 'latest' },
  main: jsFile(NODE),
  types: typesFile(NODE),
  exports: {
    './node': { import: jsFile(NODE), types: typesFile(NODE) },
    './web': { import: jsFile(WEB), types: typesFile(WEB) },
    './base': { import: jsFile(BASE), types: typesFile(BASE) },
    '.': {
      default: { import: jsFile(NODE), types: typesFile(NODE) },
      node: { import: jsFile(NODE), types: typesFile(NODE) },
      browser: { import: jsFile(WEB), types: typesFile(WEB) },
    },
  },
}
await Deno.writeTextFile('dist/package.json', JSON.stringify(packageJson, null, 2))
await Deno.copyFile('README.md', 'dist/README.md')
