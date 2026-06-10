export type DownloadPlatform =
  | "darwin-aarch64-dmg"
  | "windows-x64-nsis"
  | `linux-x64-${"deb" | "rpm" | "appimage"}`
