export async function GET() {
  const response = await fetch(
    "https://raw.githubusercontent.com/SolaceHarmony/code-harmony/refs/heads/dev/packages/sdk/openapi.json",
  )
  const json = await response.json()
  return json
}
