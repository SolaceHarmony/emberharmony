import type { DesktopTheme, ResolvedTheme } from "./types"
import { resolveThemeVariant, themeToCss } from "./resolve"
import { hexToRgb } from "./color"

let activeTheme: DesktopTheme | null = null
const THEME_STYLE_ID = "emberharmony-theme"

// Cache the macOS detection result at module load time
const IS_MACOS = typeof window !== "undefined" && "__TAURI__" in window && navigator.userAgent.includes("Mac")

function ensureLoaderStyleElement(): HTMLStyleElement {
  const existing = document.getElementById(THEME_STYLE_ID) as HTMLStyleElement | null
  if (existing) {
    return existing
  }
  const element = document.createElement("style")
  element.id = THEME_STYLE_ID
  document.head.appendChild(element)
  return element
}

export function applyTheme(theme: DesktopTheme, themeId?: string): void {
  activeTheme = theme
  const lightTokens = resolveThemeVariant(theme.light, false)
  const darkTokens = resolveThemeVariant(theme.dark, true)
  const targetThemeId = themeId ?? theme.id
  const css = buildThemeCss(lightTokens, darkTokens, targetThemeId)
  const themeStyleElement = ensureLoaderStyleElement()
  themeStyleElement.textContent = css
  document.documentElement.setAttribute("data-theme", targetThemeId)
}

function buildThemeCss(light: ResolvedTheme, dark: ResolvedTheme, themeId: string): string {
  const isDefaultTheme = themeId === "eh-1"
  const lightCss = themeToCss(light)
  const darkCss = themeToCss(dark)

  // On macOS, use semi-transparent backgrounds derived from theme colors
  // to allow vibrancy effects while maintaining theme consistency
  const macOSBackgroundOverride = IS_MACOS
    ? `
  /* macOS vibrancy support - semi-transparent backgrounds for better contrast */
  html, body {
    background-color: transparent !important;
  }
  
  :root {
    --background-base: ${toTransparentRgba(light["background-base"], 0.85)};
    --background-strong: ${toTransparentRgba(light["background-strong"] || light["background-base"], 0.95)};
  }
  
  @media (prefers-color-scheme: dark) {
    :root {
      --background-base: ${toTransparentRgba(dark["background-base"], 0.85)};
      --background-strong: ${toTransparentRgba(dark["background-strong"] || dark["background-base"], 0.95)};
    }
  }
`
    : ""

  if (isDefaultTheme) {
    return `
:root {
  color-scheme: light;
  --text-mix-blend-mode: multiply;

  ${lightCss}

  @media (prefers-color-scheme: dark) {
    color-scheme: dark;
    --text-mix-blend-mode: plus-lighter;

    ${darkCss}
  }
}

${macOSBackgroundOverride}
`
  }

  return `
html[data-theme="${themeId}"] {
  color-scheme: light;
  --text-mix-blend-mode: multiply;

  ${lightCss}

  @media (prefers-color-scheme: dark) {
    color-scheme: dark;
    --text-mix-blend-mode: plus-lighter;

    ${darkCss}
  }
}

${macOSBackgroundOverride}
`
}

function toTransparentRgba(color: string, alpha: number): string {
  // Handle CSS variable references
  if (color.startsWith("var(")) {
    return color
  }

  // Handle rgba() format
  if (color.startsWith("rgba(")) {
    const match = color.match(/rgba\((\d+),\s*(\d+),\s*(\d+)/)
    if (match) {
      return `rgba(${match[1]}, ${match[2]}, ${match[3]}, ${alpha})`
    }
    return color
  }

  // Handle hex color format
  if (color.startsWith("#")) {
    const rgb = hexToRgb(color as `#${string}`)
    return `rgba(${Math.round(rgb.r * 255)}, ${Math.round(rgb.g * 255)}, ${Math.round(rgb.b * 255)}, ${alpha})`
  }

  return color
}

export async function loadThemeFromUrl(url: string): Promise<DesktopTheme> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to load theme from ${url}: ${response.statusText}`)
  }
  return response.json()
}

export function getActiveTheme(): DesktopTheme | null {
  const activeId = document.documentElement.getAttribute("data-theme")
  if (!activeId) {
    return null
  }
  if (activeTheme?.id === activeId) {
    return activeTheme
  }
  return null
}

export function removeTheme(): void {
  activeTheme = null
  const existingElement = document.getElementById(THEME_STYLE_ID)
  if (existingElement) {
    existingElement.remove()
  }
  document.documentElement.removeAttribute("data-theme")
}

export function setColorScheme(scheme: "light" | "dark" | "auto"): void {
  if (scheme === "auto") {
    document.documentElement.style.removeProperty("color-scheme")
  } else {
    document.documentElement.style.setProperty("color-scheme", scheme)
  }
}
