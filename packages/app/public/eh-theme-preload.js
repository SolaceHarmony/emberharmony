;(function () {
  var themeId = localStorage.getItem("emberharmony-theme-id")
  if (!themeId) return

  var scheme = localStorage.getItem("emberharmony-color-scheme") || "system"
  var isDark = scheme === "dark" || (scheme === "system" && matchMedia("(prefers-color-scheme: dark)").matches)
  var mode = isDark ? "dark" : "light"

  document.documentElement.dataset.theme = themeId
  document.documentElement.dataset.colorScheme = mode

  if (themeId === "eh-1") return

  var css = localStorage.getItem(isDark ? "emberharmony-theme-css-dark" : "emberharmony-theme-css-light")
  if (css) {
    var style = document.createElement("style")
    style.id = "emberharmony-theme-preload"
    style.textContent =
      ":root{color-scheme:" +
      mode +
      ";--text-mix-blend-mode:" +
      (isDark ? "plus-lighter" : "multiply") +
      ";" +
      css +
      "}"
    document.head.appendChild(style)
  }
})()
