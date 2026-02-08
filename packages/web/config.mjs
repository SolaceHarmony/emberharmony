const stage = process.env.SST_STAGE || "dev"

export default {
  url: stage === "production" ? "https://solace.ofharmony.ai" : `https://${stage}.solace.ofharmony.ai`,
  console: stage === "production" ? "https://solace.ofharmony.ai/auth" : `https://${stage}.solace.ofharmony.ai/auth`,
  email: "contact@anoma.ly",
  socialCard: "https://social-cards.sst.dev",
  github: "https://github.com/sydneyrenee/code-harmony",
  discord: "https://discord.gg/EdF8f7JR",
  headerLinks: [
    { name: "Home", url: "/" },
    { name: "Docs", url: "/docs/" },
  ],
}
