export const waitUntil = async (promise: Promise<unknown>) => {
  await promise
}

const env = (...names: string[]) => names.map((name) => process.env[name]).find((value) => value !== undefined)

const required = (...names: string[]) => {
  const value = env(...names)
  if (value !== undefined) return value
  throw new Error(`${names.join(" or ")} is required`)
}

const secret = (name: string) => ({
  get value() {
    return required(name)
  },
})

export const Resource = {
  App: {
    get stage() {
      return env("APP_STAGE", "STAGE", "NODE_ENV") ?? "dev"
    },
  },
  Database: {
    get database() {
      return required("DATABASE_NAME", "DATABASE_DATABASE")
    },
    get host() {
      return required("DATABASE_HOST")
    },
    get username() {
      return required("DATABASE_USERNAME", "DATABASE_USER")
    },
    get password() {
      return required("DATABASE_PASSWORD")
    },
    get port() {
      return Number(env("DATABASE_PORT") || "3306")
    },
  },
  AWS_SES_ACCESS_KEY_ID: secret("AWS_SES_ACCESS_KEY_ID"),
  AWS_SES_SECRET_ACCESS_KEY: secret("AWS_SES_SECRET_ACCESS_KEY"),
  EMAILOCTOPUS_API_KEY: secret("EMAILOCTOPUS_API_KEY"),
  GITHUB_CLIENT_ID_CONSOLE: secret("GITHUB_CLIENT_ID_CONSOLE"),
  GITHUB_CLIENT_SECRET_CONSOLE: secret("GITHUB_CLIENT_SECRET_CONSOLE"),
  GOOGLE_CLIENT_ID: secret("GOOGLE_CLIENT_ID"),
  HONEYCOMB_API_KEY: secret("HONEYCOMB_API_KEY"),
  STRIPE_SECRET_KEY: secret("STRIPE_SECRET_KEY"),
  STRIPE_WEBHOOK_SECRET: secret("STRIPE_WEBHOOK_SECRET"),
  ZEN_BLACK_LIMITS: secret("ZEN_BLACK_LIMITS"),
  ZEN_BLACK_PRICE: {
    get plan20() {
      return required("ZEN_BLACK_PRICE_PLAN20")
    },
    get plan100() {
      return required("ZEN_BLACK_PRICE_PLAN100")
    },
    get plan200() {
      return required("ZEN_BLACK_PRICE_PLAN200")
    },
  },
  ZEN_MODELS1: secret("ZEN_MODELS1"),
  ZEN_MODELS2: secret("ZEN_MODELS2"),
  ZEN_MODELS3: secret("ZEN_MODELS3"),
  ZEN_MODELS4: secret("ZEN_MODELS4"),
  ZEN_MODELS5: secret("ZEN_MODELS5"),
  ZEN_MODELS6: secret("ZEN_MODELS6"),
  ZEN_MODELS7: secret("ZEN_MODELS7"),
  ZEN_MODELS8: secret("ZEN_MODELS8"),
  ZEN_MODELS9: secret("ZEN_MODELS9"),
  ZEN_MODELS10: secret("ZEN_MODELS10"),
  ZEN_SESSION_SECRET: secret("ZEN_SESSION_SECRET"),
  GatewayKv: {
    get: async (_key: string) => undefined as string | undefined,
    put: async (_key: string, _value: string, _options?: { expirationTtl?: number }) => {},
  },
  ZenDataNew: {
    put: async (_key: string, _value: string) => {},
  },
} as const
