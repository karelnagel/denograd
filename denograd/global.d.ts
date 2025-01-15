interface ImportMetaEnv {
  SSR: boolean
  [key: string]: string | boolean | undefined
}

interface ImportMeta {
  env: ImportMetaEnv
}
