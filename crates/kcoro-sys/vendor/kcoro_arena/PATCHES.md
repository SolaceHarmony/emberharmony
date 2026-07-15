# kcoro_arena Provenance

This directory is a source snapshot of:

- Repository: `/Volumes/stuff/Projects/kotlinmania/kcoro_arena`
- Upstream revision: `bd530f4c9196d948472067c5bc379e7117c645b2`
- Ticket/wait implementation revision: `bcdc03d1a0731ee3116c850f3f9bd7cb27b55101`
- License: BSD-3-Clause

Vendored production paths are `include/`, `core/src/`, and `port/`, plus the
upstream `LICENSE` and `README.md`. EmberHarmony carries no local source patch
inside this snapshot. Changes belong upstream first; after its gates pass and
the change is committed, resync these paths and update both revisions here.

The upstream C test suite is authoritative for runtime behavior. The
`kcoro-sys` Rust tests verify that this exact snapshot compiles, links, and
preserves the FFI contracts EmberHarmony consumes.
