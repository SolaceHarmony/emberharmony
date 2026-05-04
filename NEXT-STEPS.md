# EmberHarmony Rebrand — Next Steps

This document tracks the rebranding from the upstream "opencode" (and the intermediate
"code-harmony") to **EmberHarmony**. The npm org remains `@thesolaceproject`.

---

## Completed

### Bug fix
- **Locale detection** (`packages/app/src/context/language.tsx`): Added `"en"` check
  to `detectLocale()`. Without it, English was never explicitly matched in the
  `navigator.languages` loop, so the first supported non-English locale (e.g. French)
  was selected instead. Known upstream bug with multiple open issues.

### Code & config rebranding
- **All source files**: Zero remaining `opencode` or `code-harmony` references in
  source code, config, markdown, workflows, or scripts (verified via `rg` audit).
- **packages/app**: Vite plugin name, theme preload script (renamed
  `oc-theme-preload.js` -> `eh-theme-preload.js`), localStorage keys updated to
  `emberharmony-*`, HTML references updated in both app and desktop `index.html`.
- **packages/ui**: `site.webmanifest` name/short_name -> `EmberHarmony`.
- **packages/emberharmony**: Binary fallback name in `bin/emberharmony`, Dockerfile
  paths and entrypoint.
- **packages/desktop/src-tauri**: All Rust source (`cli.rs`, `lib.rs`, `main.rs`) --
  binary names, config dirs (`.emberharmony/`), env vars (`EMBERHARMONY_*`), sidecar
  names, auth usernames, error messages. Appstream metadata XML -- name, description,
  GitHub URLs.
- **sdks/vscode**: Publish script VSIX filename.
- **packages/web**: All ~30 documentation MDX files, Lander/Head components, share
  pages -- config filenames, CLI commands, install instructions, brand name, GitHub URLs.

### Lock files & workspace
- **Root `bun.lock`**: Regenerated -- `"name": "emberharmony"`, zero old references.
- **`sdks/vscode/bun.lock`**: Clean, no old references.
- **`github/` workspace**: Added to root `workspaces.packages` so
  `@thesolaceproject/emberharmony-sdk: workspace:*` resolves locally. SDK symlink
  verified working.

---

## Remaining Work

### High Priority

#### 1. Logo SVG asset
`packages/ui/src/components/logo.tsx` contains an SVG that renders pixel-art letter
shapes spelling **"opencode"**. This needs a new SVG asset for **"emberharmony"** in
the same pixel-art style. Requires design work -- the paths encode individual letter
geometry.

Related components: `Logo`, `Mark`, `Splash`.

#### 2. `emberharmony web` in built binary -- no web UI
The upstream `opencode web` works because it falls back to their hosted UI at
`app.opencode.ai`. EmberHarmony has no hosted app, so when the built binary runs
`emberharmony web`, it can't find `packages/app` and opens the server's fallback
page ("EmberHarmony server is running") instead of the web UI.

**Resolution options (pick one):**

**Option A -- Bundle the built app into the server (recommended)**
Add a build step that runs `vite build` on `packages/app` and embeds the output.
The server's catch-all route (`packages/emberharmony/src/server/server.ts:530`)
would serve these static files instead of the placeholder HTML. The build script
(`packages/emberharmony/script/build.ts`) would need to include the app dist as
embedded assets in the compiled binary.

**Option B -- Host the app**
Deploy the built app to a CDN/hosting (e.g. `app.solace.ofharmony.ai`) and update
`web.ts` non-local branch to open that URL with the local server address as a
query parameter.

**Relevant files:**
- `packages/emberharmony/src/cli/cmd/web.ts` -- `web` command, local app discovery
- `packages/emberharmony/src/server/server.ts:530` -- fallback HTML page
- `packages/emberharmony/script/build.ts` -- binary build script

#### 3. `oc-1` default theme ID ✅ DONE
Renamed to `"eh-1"` across:
- `packages/ui/src/theme/themes/eh-1.json` (renamed from `oc-1.json`, `id`/`name` updated)
- `packages/ui/src/theme/default-themes.ts` (import + export + `DEFAULT_THEMES` key)
- `packages/ui/src/theme/context.tsx` (all 3 guard/default references)
- `packages/ui/src/theme/loader.ts` (`isDefaultTheme` guard)
- `packages/ui/src/theme/index.ts` (barrel export `eh1Theme`)
- `packages/app/public/eh-theme-preload.js` (preload guard)

#### 4. Cargo.lock regeneration
`packages/desktop/src-tauri/Cargo.lock` still contains `"code-harmony-desktop"`.
The `Cargo.toml` is already updated to `"emberharmony-desktop"`. Regenerate when
Rust toolchain is available:
```sh
cd packages/desktop/src-tauri && cargo generate-lockfile
```

#### 5. Test snapshots
`packages/emberharmony/test/tool/__snapshots__/tool.test.ts.snap` contains old
`code-harmony` path references. Regenerate by running the test suite:
```sh
bun test packages/emberharmony/test/tool/tool.test.ts --update-snapshots
```

#### 6. Build artifacts
`packages/emberharmony/dist-local/` may contain old build artifacts with `code-harmony`
paths. Clean and rebuild:
```sh
rm -rf packages/emberharmony/dist-local
```

### Medium Priority

#### 7. `.git/opencode` and `.git/code-harmony`
Git internal ref files from the upstream repo. Harmless but can be cleaned up:
```sh
rm .git/opencode .git/code-harmony
```

### Low Priority / Cosmetic

#### 8. `oc-theme-preload` script ID prefix
The preload script was renamed to `eh-theme-preload.js` with `id="eh-theme-preload-script"`.
If a different prefix convention is preferred (e.g. `emberharmony-theme-preload`),
update in `packages/app/index.html`, `packages/desktop/index.html`, and the JS file.

---

## npm Publishing Checklist

The npm org is **`@thesolaceproject`**. Packages to verify/publish:

| Package | Expected name |
|---------|--------------|
| Core CLI | `@thesolaceproject/emberharmony` |
| SDK | `@thesolaceproject/emberharmony-sdk` |
| UI | `@thesolaceproject/emberharmony-ui` |
| Util | `@thesolaceproject/emberharmony-util` |

Ensure all `package.json` `name` fields use the `@thesolaceproject/emberharmony*`
convention before publishing.

---

## Verification

After all changes are complete, run a final audit:
```sh
# Source files (should return zero matches)
rg -i 'opencode|code-harmony|CodeHarmony' --glob '!*.lock' --glob '!.git/**' --glob '!node_modules/**' --glob '!dist-local/**' --glob '!__snapshots__/**'

# Lock files (only Cargo.lock should match until regenerated)
rg -i 'opencode|code-harmony' --glob '*.lock' --glob '!node_modules/**'
```
