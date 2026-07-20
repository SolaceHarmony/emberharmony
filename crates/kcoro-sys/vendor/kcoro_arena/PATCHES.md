# kcoro_arena Provenance

This directory is a source snapshot of:

- Repository: `/Volumes/stuff/Projects/kotlinmania/kcoro_arena`
- Upstream revision: `bd530f4c9196d948472067c5bc379e7117c645b2`
- Ticket/wait implementation revision: `bcdc03d1a0731ee3116c850f3f9bd7cb27b55101`
- License: BSD-3-Clause

The original import came from that revision. EmberHarmony now intentionally
carries a reduced production profile: retained services, fixed teams, exact
product-edge identity, private idle doorbells, and the POSIX port. Product
tickets and borrowed numerical views are owned by Flashkern. The generic
channel/scheduler/timer, process-global compatibility runtime, and persistence
surfaces were deleted after the native Flashkern cutover. Do not restore them
during a vendor refresh.

The upstream C test suite is authoritative for runtime behavior. The
`kcoro-sys` Rust tests verify that this exact snapshot compiles, links, and
preserves the FFI contracts EmberHarmony consumes.
