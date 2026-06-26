import { test, expect } from "../fixtures"
import { promptSelector } from "../utils"

// Regression for the session-clone burst: a submit on the new-session view must
// create EXACTLY ONE session even when several submits fire before navigation
// updates params.id. We hold the POST /session response open so the race window
// (params.id still undefined) is deterministic rather than sub-frame.
//
// Skipped on CI: the new-session submit early-returns without a configured model
// provider, so no session would be created (same constraint as prompt.spec.ts).
test.skip(!!process.env.CI, "Disabled in CI: session creation requires a configured model provider")

function isCreateUrl(url: string): boolean {
  try {
    return new URL(url).pathname.replace(/\/+$/, "").endsWith("/session")
  } catch {
    return false
  }
}

test("rapid submit on a new session creates exactly one session", async ({ page, sdk, gotoSession }) => {
  test.setTimeout(60_000)
  await gotoSession()

  let createCount = 0
  await page.route(
    (url) => isCreateUrl(url.toString()),
    async (route) => {
      if (route.request().method() !== "POST") return route.continue()
      createCount++
      // hold the create so params.id stays undefined across the rapid submits
      await new Promise((r) => setTimeout(r, 1500))
      return route.continue()
    },
  )

  const prompt = page.locator(promptSelector)
  await prompt.click()
  await page.keyboard.type("clone-burst regression")
  // fire several submits inside the (now-stretched) create window
  for (let i = 0; i < 6; i++) await page.keyboard.press("Enter")

  // navigation only happens once a session is created — also our "provider present" check
  await expect(page).toHaveURL(/\/session\/[^/?#]+/, { timeout: 30_000 })
  await page.waitForTimeout(1500) // let any stray creates land before asserting

  const sessionID = /\/session\/([^/?#]+)/.exec(page.url())?.[1]
  try {
    expect(createCount).toBe(1)
  } finally {
    if (sessionID) await sdk.session.delete({ sessionID }).catch(() => undefined)
  }
})
