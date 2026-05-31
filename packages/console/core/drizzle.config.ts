import { defineConfig } from "drizzle-kit"

const env = (...names: string[]) => names.map((name) => process.env[name]).find((value) => value)

const required = (...names: string[]) => {
  const value = env(...names)
  if (value) return value
  throw new Error(`${names.join(" or ")} is required`)
}

export default defineConfig({
  out: "./migrations/",
  strict: true,
  schema: ["./src/**/*.sql.ts"],
  verbose: true,
  dialect: "mysql",
  dbCredentials: {
    database: required("DATABASE_NAME", "DATABASE_DATABASE"),
    host: required("DATABASE_HOST"),
    user: required("DATABASE_USERNAME", "DATABASE_USER"),
    password: required("DATABASE_PASSWORD"),
    port: Number(env("DATABASE_PORT") ?? "3306"),
    ssl: {
      rejectUnauthorized: false,
    },
  },
})
