;(function () {
  var themeId = localStorage.getItem("code-harmony-theme-id")
  if (!themeId) return

  var scheme = localStorage.getItem("code-harmony-color-scheme") || "system"
  var isDark = scheme === "dark" || (scheme === "system" && matchMedia("(prefers-color-scheme: dark)").matches)
  var mode = isDark ? "dark" : "light"

  document.documentElement.dataset.theme = themeId
  document.documentElement.dataset.colorScheme = mode

  if (themeId === "oc-1") return

  var css = localStorage.getItem(isDark ? "code-harmony-theme-css-dark" : "code-harmony-theme-css-light")
  if (css) {
    var style = document.createElement("style")
    style.id = "code-harmony-theme-preload"
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
