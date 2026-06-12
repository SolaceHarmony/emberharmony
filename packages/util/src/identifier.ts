import { randomBytes } from "crypto"

export namespace Identifier {
  const LENGTH = 26

  // State for monotonic ID generation
  let lastTimestamp = 0
  let counter = 0

  export function ascending() {
    return create(false)
  }

  export function descending() {
    return create(true)
  }

  function randomBase62(length: number): string {
    const chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    // Rejection sampling: 256 is not a multiple of 62, so `byte % 62` biases
    // toward the first 256 % 62 = 8 characters. Discard bytes in the biased
    // tail (>= 248, the largest multiple of 62 that fits) so every character
    // is equally likely.
    const limit = 256 - (256 % 62) // 248
    let result = ""
    while (result.length < length) {
      for (const byte of randomBytes(length - result.length)) {
        if (byte < limit) result += chars[byte % 62]
      }
    }
    return result
  }

  export function create(descending: boolean, timestamp?: number): string {
    const currentTimestamp = timestamp ?? Date.now()

    if (currentTimestamp !== lastTimestamp) {
      lastTimestamp = currentTimestamp
      counter = 0
    }
    counter++

    let now = BigInt(currentTimestamp) * BigInt(0x1000) + BigInt(counter)

    now = descending ? ~now : now

    const timeBytes = Buffer.alloc(6)
    for (let i = 0; i < 6; i++) {
      timeBytes[i] = Number((now >> BigInt(40 - 8 * i)) & BigInt(0xff))
    }

    return timeBytes.toString("hex") + randomBase62(LENGTH - 12)
  }
}
