import path from "path"
import fs from "fs/promises"
import { Global } from "../global"
import z from "zod"

export namespace Log {
  export const Level = z.enum(["DEBUG", "INFO", "WARN", "ERROR"]).meta({ ref: "LogLevel", description: "Log level" })
  export type Level = z.infer<typeof Level>

  const levelPriority: Record<Level, number> = {
    DEBUG: 0,
    INFO: 1,
    WARN: 2,
    ERROR: 3,
  }

  let level: Level = "INFO"

  function shouldLog(input: Level): boolean {
    return levelPriority[input] >= levelPriority[level]
  }

  export type Logger = {
    debug(message?: any, extra?: Record<string, any>): void
    info(message?: any, extra?: Record<string, any>): void
    error(message?: any, extra?: Record<string, any>): void
    warn(message?: any, extra?: Record<string, any>): void
    tag(key: string, value: string): Logger
    clone(): Logger
    time(
      message: string,
      extra?: Record<string, any>,
    ): {
      stop(): void
      [Symbol.dispose](): void
    }
  }

  const loggers = new Map<string, Logger>()

  export const Default = create({ service: "default" })

  export interface Options {
    print: boolean
    dev?: boolean
    level?: Level
  }

  let logpath = ""
  export function file() {
    return logpath
  }
  let write = (msg: any) => {
    process.stderr.write(msg)
    return msg.length
  }

  export async function init(options: Options) {
    if (options.level) level = options.level
    cleanup(Global.Path.log)
    if (options.print) return
    logpath = path.join(
      Global.Path.log,
      options.dev ? "dev.log" : new Date().toISOString().split(".")[0].replace(/:/g, "") + ".log",
    )
    const logfile = Bun.file(logpath)
    await fs.truncate(logpath).catch(() => {})
    const writer = logfile.writer()
    write = async (msg: any) => {
      const num = writer.write(msg)
      writer.flush()
      return num
    }
  }

  async function cleanup(dir: string) {
    const glob = new Bun.Glob("????-??-??T??????.log")
    const files = await Array.fromAsync(
      glob.scan({
        cwd: dir,
        absolute: true,
      }),
    )
    if (files.length <= 5) return

    const filesToDelete = files.slice(0, -10)
    await Promise.all(filesToDelete.map((file) => fs.unlink(file).catch(() => {})))
  }

  function formatError(error: Error, depth = 0): string {
    const result = error.message
    return error.cause instanceof Error && depth < 10
      ? result + " Caused by: " + formatError(error.cause, depth + 1)
      : result
  }

  // Cap any single rendered field so one fat value (e.g. an AI-SDK error that
  // embeds the entire request body — system prompt + message history) can't
  // bloat a log line to tens of KB. The message arg is never capped here, only
  // the structured key=value fields.
  const MAX_VALUE_LENGTH = 2048
  function truncateValue(rendered: string): string {
    if (rendered.length <= MAX_VALUE_LENGTH) return rendered
    return rendered.slice(0, MAX_VALUE_LENGTH) + `…[+${rendered.length - MAX_VALUE_LENGTH} chars]`
  }
  function safeStringify(value: any): string {
    try {
      return JSON.stringify(value) ?? String(value)
    } catch (e) {
      // circular refs / BigInt make JSON.stringify throw. Retry with a
      // circular-safe replacer so we still render partial structure; if that
      // also fails, emit a NAMED marker rather than the useless "[object Object]"
      // (the old code would have thrown loudly here — don't silently swallow).
      try {
        const seen = new WeakSet()
        return JSON.stringify(value, (_k, v) => {
          if (typeof v === "bigint") return `${v.toString()}n`
          if (typeof v === "object" && v !== null) {
            if (seen.has(v)) return "[Circular]"
            seen.add(v)
          }
          return v
        })
      } catch {
        return `[unserializable: ${e instanceof Error ? e.message : String(e)}]`
      }
    }
  }

  let last = Date.now()
  export function create(tags?: Record<string, any>) {
    tags = tags || {}

    const service = tags["service"]
    if (service && typeof service === "string") {
      const cached = loggers.get(service)
      if (cached) {
        return cached
      }
    }

    function build(message: any, extra?: Record<string, any>) {
      const prefix = Object.entries({
        ...tags,
        ...extra,
      })
        .filter(([_, value]) => value !== undefined && value !== null)
        .map(([key, value]) => {
          const prefix = `${key}=`
          const rendered =
            value instanceof Error
              ? formatError(value)
              : typeof value === "object"
                ? safeStringify(value)
                : String(value)
          return prefix + truncateValue(rendered)
        })
        .join(" ")
      const next = new Date()
      const diff = next.getTime() - last
      last = next.getTime()
      return [next.toISOString().split(".")[0], "+" + diff + "ms", prefix, message].filter(Boolean).join(" ") + "\n"
    }
    const result: Logger = {
      debug(message?: any, extra?: Record<string, any>) {
        if (shouldLog("DEBUG")) {
          write("DEBUG " + build(message, extra))
        }
      },
      info(message?: any, extra?: Record<string, any>) {
        if (shouldLog("INFO")) {
          write("INFO  " + build(message, extra))
        }
      },
      error(message?: any, extra?: Record<string, any>) {
        if (shouldLog("ERROR")) {
          write("ERROR " + build(message, extra))
        }
      },
      warn(message?: any, extra?: Record<string, any>) {
        if (shouldLog("WARN")) {
          write("WARN  " + build(message, extra))
        }
      },
      tag(key: string, value: string) {
        if (tags) tags[key] = value
        return result
      },
      clone() {
        return Log.create({ ...tags })
      },
      time(message: string, extra?: Record<string, any>) {
        const now = Date.now()
        result.info(message, { status: "started", ...extra })
        function stop() {
          result.info(message, {
            status: "completed",
            duration: Date.now() - now,
            ...extra,
          })
        }
        return {
          stop,
          [Symbol.dispose]() {
            stop()
          },
        }
      },
    }

    if (service && typeof service === "string") {
      loggers.set(service, result)
    }

    return result
  }
}
