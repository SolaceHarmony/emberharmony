const stage = process.env.APP_STAGE || process.env.STAGE || "dev"

export default {
  url: stage === "production" ? "https://solace.ofharmony.ai" : `https://${stage}.solace.ofharmony.ai`,
  console: stage === "production" ? "https://solace.ofharmony.ai/auth" : `https://${stage}.solace.ofharmony.ai/auth`,
  email: "sydney@solace.ofharmony.ai",
  socialCard: "https://solace.ofharmony.ai/social-cards",
  github: "https://github.com/sydneyrenee/emberharmony",
  discord: "https://discord.gg/EdF8f7JR",
  headerLinks: [
    { name: "Home", url: "/" },
    { name: "Docs", url: "/docs/" },
  ],
}
