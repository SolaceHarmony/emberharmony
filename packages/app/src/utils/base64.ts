import { base64Decode } from "@thesolaceproject/code-harmony-util/encode"

export function decode64(value: string | undefined) {
  if (value === undefined) return
  try {
    return base64Decode(value)
  } catch {
    return
  }
}
