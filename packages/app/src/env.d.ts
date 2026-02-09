interface ImportMetaEnv {
  readonly VITE_CODE_HARMONY_SERVER_HOST: string
  readonly VITE_CODE_HARMONY_SERVER_PORT: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
