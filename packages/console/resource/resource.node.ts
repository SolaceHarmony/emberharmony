import { Resource as ResourceBase } from "sst"

export const waitUntil = async (promise: Promise<unknown>) => {
  await promise
}

export const Resource = ResourceBase as Record<string, any>
