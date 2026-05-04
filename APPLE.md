# APPLE: Notarizing macOS Builds (Developer ID)

This document explains how to sign, notarize, and staple macOS builds for distribution outside the Mac App Store. It covers both the Xcode Organizer (GUI) path and the command-line flow using `notarytool`, plus CI-friendly examples and troubleshooting tips.

If you share what you’re shipping (.app, .dmg, .pkg, or .zip) and whether you prefer Xcode or command line, you can tailor these steps directly to your build.

## Prerequisites
- An Apple Developer account with:
  - Developer ID Application certificate (for apps, tools, frameworks)
  - Developer ID Installer certificate (for .pkg installers)
- Xcode 14+ (required for notarization uploads; `notarytool` replaces deprecated `altool`)
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
  "MyApp.app/Contents/Frameworks/Some.framework"

# Then sign the app bundle
codesign --force --options runtime --timestamp \
  --sign "Developer ID Application: Your Org (TEAMID)" \
  "MyApp.app"

# Verify signature (use --deep for verification only)
codesign --verify --deep --strict --verbose=2 "MyApp.app"
```

### 2) Package the artifact you will submit

For a `.app`, submit a ZIP that preserves bundle metadata:

```bash
ditto -c -k --keepParent "MyApp.app" "MyApp.zip"
```

For a `.dmg` or `.pkg`, you can submit the DMG/PKG directly.

### 3) Submit for notarization with `notarytool`

There are two valid authentication methods.

#### Option B1: App Store Connect API key (preferred for CI)

You need an **App Store Connect API key**, downloaded once as `AuthKey_<KEYID>.p8`.
Create it in **App Store Connect** → **Users and Access** → **Integrations** → **App Store Connect API**.

> Important: A `.p8` created in the Apple Developer Portal “Keys” page (APNs, Maps, Sign in with Apple, etc.) is **not** the same thing and won’t work for notarization.

```bash
xcrun notarytool submit "MyApp.zip" \
  --key "/path/to/AuthKey_ABC123DEFG.p8" \
  --key-id "ABC123DEFG" \
  --issuer "YOUR-ISSUER-ID" \
  --wait
```

#### Option B2: Apple ID + app-specific password (fallback)

Create an **app-specific password** at https://appleid.apple.com.

```bash
xcrun notarytool store-credentials "notary" \
  --apple-id "you@example.com" \
  --team-id "TEAMID" \
  --password "xxxx-xxxx-xxxx-xxxx"

xcrun notarytool submit "MyApp.zip" \
  --keychain-profile "notary" \
  --wait
```

### 4) Staple the ticket

Stapling makes the notarization ticket available offline:

```bash
xcrun stapler staple "MyApp.app"
# or
xcrun stapler staple "MyApp.dmg"
# or
xcrun stapler staple "MyApp.pkg"
```

---

## CI notes for this repo

Our GitHub Actions workflow signs/notarizes macOS releases in `.github/workflows/publish.yml`.
It expects these **Actions secrets**:

**Signing**
- `APPLE_CERTIFICATE`: base64 of your exported **Developer ID Application** `.p12`
- `APPLE_CERTIFICATE_PASSWORD`: password for that `.p12`
- `APPLE_SIGNING_IDENTITY` (optional): explicit identity string to use (useful if you have multiple certs)

**Notarization (choose one)**
- App Store Connect API key:
  - `APPLE_API_ISSUER`
  - `APPLE_API_KEY`
  - `APPLE_API_KEY_PATH` (the *contents* of `AuthKey_*.p8`)
- Apple ID fallback:
  - `APPLE_ID`
  - `APPLE_PASSWORD` (app-specific password)
  - `APPLE_TEAM_ID`

