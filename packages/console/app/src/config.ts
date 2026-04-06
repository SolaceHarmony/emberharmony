/**
 * Application-wide constants and configuration
 */
export const config = {
  // Base URL
  baseUrl: "https://solace.ofharmony.ai",

  // GitHub
  github: {
    repoUrl: "https://github.com/SolaceHarmony/emberharmony",
    starsFormatted: {
      compact: "80K",
      full: "80,000",
    },
  },

  // Social links
  social: {
    website: "https://solace.ofharmony.ai",
    discord: "https://discord.gg/EdF8f7JR",
  },

  // Static stats (used on landing page)
  stats: {
    contributors: "600",
    commits: "7,500",
    monthlyUsers: "1.5M",
  },
} as const
