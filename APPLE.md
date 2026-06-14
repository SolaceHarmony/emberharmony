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

You can submit any of the above to Apple’s notarization service. Best practice is to notarize the app bundle and staple it, then package into a DMG or PKG, which should also be notarized and stapled to ensure a smooth Gatekeeper experience.

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

## Local builds in this repo

`build-local.ts` handles signing and notarization automatically. It reads credentials from the repo-root `.env` file and verifies them before starting the build.

### What happens by default

1. **Signing**: `APPLE_SIGNING_IDENTITY` from `.env` is verified against your keychain. If the cert isn't found, the build fails with a list of valid identities. If you don't have a Developer ID cert, pass `--quick` for ad-hoc signing.

2. **Voice runtime signing**: All Mach-O binaries in `src-tauri/resources/voice/` are signed with your Developer ID before the app bundle is sealed. This is handled by `scripts/sign-voice-runtime.ts`, which runs automatically as part of the build.

3. **Notarization**: Apple's notary service receives the app, verifies the signing and hardened runtime, and returns a ticket. The build script waits for this to complete (typically 2-8 minutes). The ticket is then stapled to the app bundle.

4. **DMG creation**: After notarization, `hdiutil create` packages the `.app` into a drag-to-install DMG with an `/Applications` symlink.

### Opting out

| Flag            | What it skips                                                                                                                                 |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `--quick`       | Real signing (uses ad-hoc `-`), notarization, DMG creation. Fastest build, but macOS will quarantine the app.                                 |
| `--no-notarize` | Notarization only. The app is still signed with your Developer ID, so macOS won't quarantine it, but users may see a warning on first launch. |
| `--no-dmg`      | DMG creation only. The `.app` bundle is still signed and notarized.                                                                           |

### Required `.env` variables

```bash
# Signing (required for all builds except --quick)
APPLE_SIGNING_IDENTITY=Developer ID Application: Your Name (TEAMID)

# Notarization (required unless --quick or --no-notarize)
APPLE_API_KEY=ABC123DEFG
APPLE_API_ISSUER=your-issuer-id
APPLE_API_KEY_PATH=/path/to/AuthKey_ABC123DEFG.p8
APPLE_TEAM_ID=TEAMID

# Optional: .p12 certificate import (only needed in CI, not local builds)
APPLE_CERTIFICATE=
APPLE_CERTIFICATE_PASSWORD=
```

### Troubleshooting

**"signing identity not in keychain"**: Your `APPLE_SIGNING_IDENTITY` doesn't match any certificate in your keychain. Run `security find-identity -v -p codesigning` to see valid identities, or pass `--quick` for an ad-hoc build.

**"APPLE_API_KEY_PATH does not exist"**: The `.p8` file path in `.env` doesn't exist on this machine. Download it from App Store Connect → Users and Access → Integrations → App Store Connect API.

**Notarization timeout**: Apple's notary service can take 2-8 minutes. If it consistently times out, check your network connection and API key validity. Pass `--no-notarize` to skip it for faster iteration.

**Quarantine warning on --quick builds**: macOS Gatekeeper quarantines ad-hoc signed apps. Right-click the app and choose Open, or run `xattr -cr EmberHarmony\ Dev.app` to remove the quarantine flag.

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
  - `APPLE_API_KEY_PATH` (the _contents_ of `AuthKey_*.p8`)
- Apple ID fallback:
  - `APPLE_ID`
  - `APPLE_PASSWORD` (app-specific password)
  - `APPLE_TEAM_ID`
