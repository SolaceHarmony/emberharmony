interface ImportMetaEnv {
  readonly VITE_EMBERHARMONY_SERVER_HOST: string
  readonly VITE_EMBERHARMONY_SERVER_PORT: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
