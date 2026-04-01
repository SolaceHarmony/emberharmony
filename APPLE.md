# APPLE: Notarizing macOS Builds (Developer ID)

This document explains how to sign, notarize, and staple macOS builds for distribution outside the Mac App Store. It covers both the Xcode Organizer (GUI) path and the command-line flow using `notarytool`, plus CI-friendly examples and troubleshooting tips.

If you share what you’re shipping (.app, .dmg, .pkg, or .zip) and whether you prefer Xcode or command line, you can tailor these steps directly to your build.

## Prerequisites
- An Apple Developer account with:
  - Developer ID Application certificate (for apps, tools, frameworks)
  - Developer ID Installer certificate (for .pkg installers)
- Xcode 13+ (recommended; `notarytool` replaces deprecated `altool`)
- Your macOS target signed with Hardened Runtime enabled
  - In Xcode: Signing & Capabilities → Hardened Runtime
  - Ensure `com.apple.security.get-task-allow` is false for distribution

## Supported artifacts
- `.app` (macOS app bundle)
- `.dmg` (disk image)
- `.pkg` (installer package)
- `.zip` (zipped app)

You can submit any of the above to Apple’s notarization service. Best practice is to notarize the app bundle and staple it, then package into a DMG (and optionally notarize/staple the DMG as well).

---

## Option A: Xcode Organizer (GUI)
This is the simplest path if you archive with Xcode and distribute an app bundle.

1. Product → Archive your macOS app.
2. In Organizer, select the archive → Distribute App.
3. Choose “Developer ID” as the method.
4. Choose “Upload” (Organizer will sign, upload for notarization, wait for approval, and export).
5. When complete, Xcode will export a notarized build. If you later package into a DMG/PKG, staple again as needed (see Stapling below).

---

## Option B: Command line with `notarytool` (recommended for CI)
The steps below work for `.app`, `.dmg`, `.pkg`, or `.zip` submissions. Adjust paths to match your build output.

### 1) Code sign with Hardened Runtime + timestamp
Sign nested code (frameworks, helpers, plugins) before the app. Prefer explicit signing over `--deep` when possible.

```bash
# Sign nested frameworks/helpers first (repeat for each nested item)
codesign --force --options runtime --timestamp \
  --sign "Developer ID Application: Your Org (TEAMID)" \
  "MyApp.app/Contents/Frameworks/Some.framework/Versions/A"

# Then sign the app bundle
codesign --force --options runtime --timestamp \
  --sign "Developer ID Application: Your Org (TEAMID)" \
  "MyApp.app"

# Verify signature (use --deep for verification only)
codesign --verify --deep --strict --verbose=2 "MyApp.app"
