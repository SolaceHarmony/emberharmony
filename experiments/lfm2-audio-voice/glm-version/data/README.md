# `data/` — Rust-side architecture docs for the data pipeline

Rust-first companions for the `liquid-audio/src/data/` training-data
pipeline. The whole tree is **off the inference path** — it exists to build
Arrow datasets of pre-packed training rows.

| File | Rust source | Python source | Role |
|---|---|---|---|
| [`types.md`](types.md) | `src/data/types.rs` | `data/types.py` | `ChatMessage` + the six-tensor bundles (`LFM2AudioTrainingSample`/`LFM2AudioRow`/`LFM2AudioModelInput`) |
| [`dataloader.md`](dataloader.md) | `src/data/dataloader.rs` | `data/dataloader.py` | `LFM2DataLoader` + `lfm2_collator` (Arrow → padded rows → batched `LFM2AudioModelInput`) |
| [`mapper.md`](mapper.md) | `src/data/mapper.rs` | `data/mapper.py` | `LFM2AudioChatMapper` (chat → six-tensor training sample) |
| [`preprocess.md`](preprocess.md) | `src/data/preprocess.rs` + `src/data/arrow_io.rs` | `data/preprocess.py` | `preprocess_dataset` (chat iterable → Arrow `save_to_disk`) |

See [`../README.md`](../README.md) for the recurring Rust divergences
(device-agnostic, `symphonia`/`arrow` backends, U8 for bool, etc.) and the
parity summary. See `wiki/data/` for the Python-first companions.