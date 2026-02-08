export const domain = (() => {
  if ($app.stage === "production") return "solace.ofharmony.ai"
  if ($app.stage === "dev") return "dev.solace.ofharmony.ai"
  return `${$app.stage}.dev.solace.ofharmony.ai`
})()

export const zoneID = "430ba34c138cfb5360826c4909f99be8"

new cloudflare.RegionalHostname("RegionalHostname", {
  hostname: domain,
  regionKey: "us",
  zoneId: zoneID,
})

export const shortDomain = (() => {
  if ($app.stage === "production") return "solace.ofharmony.ai"
  if ($app.stage === "dev") return "dev.solace.ofharmony.ai"
  return `${$app.stage}.dev.solace.ofharmony.ai`
})()
