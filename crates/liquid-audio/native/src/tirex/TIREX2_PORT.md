# TiRex-2 → C++/NEON/kcoro port (concept: spec-10 convergence referee)

Status: shaping. Unwired standalone — nothing in build.rs links this yet.
**Port home is the sibling workspace**: `/Volumes/stuff/Projects/mlxports/
tirex-2-mlx` (vendored flashkern+kcoro, torch rim, `native/tirex/
ENGINE_PLAN.md` P0–P8). THIS file remains the math source of truth both
trees cite — edit here, sync there; do not fork the semantics. Ember's
mount is the spec-10 §9 referee consumer, landing after their P5.

Additional trap found reviewing ENGINE_PLAN v2 (stage T2, "fwd set + rev
set"): the rev direction may reuse z/o gate Wx (projections of x_t are
direction-invariant; read reversed) but f/i are CONV-FED when
conv1d_kernel_size>0, and causal conv does NOT commute with flip —
conv(flip(x)) ≠ flip(conv(x)). The rev scan needs its own conv pass +
f/i projections over the flipped known-covariate rows. Reuse-all would be
silently wrong; recompute-all is correct but wasteful.
Sources read line-by-line 2026-07-10 at tirex-2 commit (shallow clone
`/Volumes/stuff/Projects/tirex-2`), flashrnn 1.0.5 + xlstm 2.0.5 site-packages
in that repo's `.venv`.

## What this is (Sydney's framing, 2026-07-10)

**Not a parity harness — an improvement to the original using our work.**
The torch oracle pins correctness of the base forward (faithful to trained
numerics, as always); the PRODUCT is what the original cannot do:
1. **Streaming state carry** — open release recomputes full history per
   forecast (streaming is pro-gated upstream); ours carries the cell states
   forward: constant work per new patch. No torch oracle exists for this —
   its gate is the SELF-oracle: incremental feed ≡ full-recompute pass over
   the same series, within tier. Same pattern gates hibernate/restore
   (save/restore mid-stream ≡ uninterrupted — spec-10's bit-exact rule).
2. **Schedule freedom** — the latency schedule (whole machine ganged on one
   stream's recurrence) is structurally inexpressible in flashrnn's CUDA or
   a Metal dispatch; sub-team fences make it legal here.
3. **Hibernatable fixed-size state** — the conversation-image easy case,
   feeding spec-10 residency machinery.
4. **The referee application** — coupled agent-trace forecasting driving the
   cognitive barrier (below).

## Why

Spec-10 §9's cognitive barrier needs a convergence referee: agent trace
streams (pairwise embedding cosines, confidence, entropy) as coupled
multivariate series → TiRex-2 forecasts convergence with quantile bands →
barrier integrates when convergence is predicted, not when a timer fires.
TiRex-2 was pretrained on coupled series (cointegration, SCMs) — the exact
question shape. 38.4M active params (univariate) + 44.1M (multivariate),
Apache 2.0. Fixed-size xLSTM state = the referee hibernates with the session
(one small blob, spec-10 easy case).

## Blockers / external actions

- **HF checkpoint is GATED**: `NX-AI/TiRex-2` returns 401 without an
  authenticated account that accepted the gate. Sydney: accept access on HF +
  `huggingface-cli login` in `/Volumes/stuff/Projects/tirex-2/.venv`. TiRex-1
  (`NX-AI/TiRex`) probes 404 unauthenticated too. Until then: no oracle dumps,
  no checkpoint hyperparams (embedding_dim, num_blocks, recipe, heads, patch
  sizes, quantile list all live in the checkpoint's model config).
- Streaming wrapper is pro-only upstream, but ALL cell-level machinery is in
  the open code (`_FlashRNNLayer.step()`, conv `.step()`,
  `variate_mixing_block.py:183` "TODO: return state as well for streaming").
  Our state carry is assembly of published parts.

## Architecture (verified against source, not the paper)

Forward (`src/tirex2/model/tirex2.py::forward`):
1. `Scaler.scale`: loc=nanmean, scale=sqrt(nanmean((x-loc)²)) clamped ≥eps
   (1e-8); optional binary-aware bypass; optional arcsinh (config).
2. `Tokenizer.input_transform`: left-pad with NaN to patch multiple; unfold
   into [num_patches, patch_size] (stride==patch for inference).
3. NaN mask: x_mask = ~isnan (as dtype), x = nan_to_num(x, 0), concat →
   [.., 2·patch]; `ResidualBlock` embed (2-layer MLP + linear residual) → D.
4. 12× `MultivariateBlock` per recipe (templates e.g. "standard"/"recurrent";
   exact recipe comes from checkpoint config):
   - **TimeMixer** = `BiXLSTM` over [B·V, L, D]:
     RMSNorm (fp32 reductions) → fwd cell over ALL variates; rev cell
     (weight-shared, `rev_cell = fwd_cell`) over flip(known covariates) →
     recombine: shared variates concat(fwd,rev) [2D] → Linear(2D→D, no bias);
     targets keep fwd — then +residual; then RMSNorm → FeedForward (xlstm
     large FFN) +residual.
     Cell per template: sLSTM (flashrnn) or mLSTM (xlstm large, conv variant).
   - **VariateMixer** = `AttentionBlock` over [L, B·V, D]: grouped attention
     across variates per timestep, asymmetric target/covariate masking,
     optional QK RMSNorm. (attention_block.py not yet read line-by-line —
     next read.)
5. stack_out_norm (LayerNorm or RMSNorm per config) → `ResidualBlock` head →
   unflatten [Q, patch] → transpose → detokenize → re_scale
   (optional sinh clamp ±20 first).
6. Rim-level: `PostProcessor` (tta_diff differencing) + optional sign-flip TTA
   (second pass on negated inputs, quantile axis complement-mapped, averaged).
   Port later at the rim, not in the kernel.

## sLSTM cell — exact math (flashrnn/vanilla/slstm.py, verbatim semantics)

Per head h, head_dim d. States (y, c, n, m) each [d]. Per step:
```
Ry[g]  = R[h]ᵀ-blockᵍ · y_prev            # R stored [H, P=d, G=4, D=d]; Ry[g][j] = Σ_p y[p]·R[h,p,g,j]
raw[g] = Wx[g] + Ry[g] + b[h,g]           # gate slots g = 0..3
i_raw, f_raw, z_raw, o_raw = raw[0..3]    # POINTWISE slot semantics
logfplusm = logsigmoid(f_raw) + m
m_new = (first step: all n==0) ? i_raw : max(i_raw, logfplusm)
o = sigmoid(o_raw); i = exp(i_raw − m_new); f = exp(logfplusm − m_new)
c_new = f·c + i·tanh(z_raw)
n_new = max(f·n + i, 1.0)
y_new = o · c_new / n_new
```

### ⚠ GATE-ORDER PLUMBING (do NOT "fix")
- The TiRex layer stacks `Wx = (fgate_proj, igate_proj, zgate, ogate)`
  (flashrnn_slstm.py step/forward), i.e. layer-"f" feeds pointwise slot 0
  (= the exp/max "i" path) and layer-"i" feeds slot 1 (= logsigmoid "f").
- Bias/recurrent tensors are initialized in (i,f,z,o) slot order with the
  powerlaw forget init at slot 1 (reset_bias), consistent with POINTWISE
  slots, not with the layer's Wx stack labels.
- flashrnn dispatch performs NO gate permutation (axis reshapes only,
  `_internal_input_permutation` is shape-level). Verified flashrnn.py
  ~940-1010 + config fields.
- The trained checkpoint is self-consistent with this exact wiring. Port the
  DATA PATHS bit-for-bit: stack layer projections (f,i,z,o) into Wx slots
  0..3, load R/b as stored, apply pointwise slot semantics above.
- First-step branch: torch uses a WHOLE-TENSOR `all(n==0)` predicate. In the
  kernel: per-stream step counter (state starts zeroed; n≥1 after step 1).
  Never batch fresh+running streams into one torch-parity comparison.

### ⚠ THREE sLSTM VARIANTS EXIST — only flashrnn's is TiRex's truth
1. **flashrnn vanilla** (above; TiRex-2 trained on THIS): per-element gates
   from headwise projections + R·y recurrent matvec; NO gate clamps;
   n floored at 1 (`max(fn+i, 1)`); whole-tensor first-step branch.
2. **xlstm-package canonical** (Sydney's kernel notes reference):
   `min(exp(...), 1.0)` clamps on i/f gates; divide by n (notes) or n+eps
   (her Metal docstring); no first-step branch.
3. **Her Metal step cell** (xLSTM-metal stepwise): scalar per-head gates
   (7B-style), NO R·y inside the kernel (projections outside), /(n+eps).
Do not mix them. Parity target is #1 exclusively. Her kernels contribute the
decomposition, the Metal harness pattern (`mx.fast.metal_kernel`, pointwise
step, pre-simdgroup_matrix — a future Metal twin should fuse R·y with
simdgroup_matrix), and the two-branch logsigmoid idiom.

Gate projections: `LinearHeadwiseExpand` ×4 — block-diagonal per-head linear
d→d, bias=False. f/i projections read the conv path (CausalConv1d k=?, then
SiLU) IF conv1d_kernel_size>0 (checkpoint config decides); z/o read raw x.
Post: MultiHeadLayerNorm per head, eps 1e-6, weight yes bias no,
**force_float32_reductions=True** (accumulate in fp32).

## mLSTM cell — exact math (Sydney's MLX port, the house reference)

Authority: `/Volumes/stuff/Projects/mlxports/xLSTM-metal/xlstm_metal/mlx_jit/
blocks/mlstm/mlstm_chunkwise/mlstm_recurrent_kernel_cell.py` — her xLSTM-7b
port, cross-backend parity-tested vs torch_native. She has NOT ported TiRex;
this is the SHAPE to work from. Cross-validate against tirex-2's
`xlstm/xlstm_large` backend via the oracle once the HF gate opens.

Per head: state C[dqk, dv] matrix, n[dqk] vector, m scalar. Per step
(fp32 compute + fp32 state on CPU):
```
f_log  = −log(1 + exp(−f_t))              # logsigmoid(f_preact)
m_new  = max(f_log + m, i_t)
f_gate = exp(f_log + m − m_new)
i_gate = exp(i_t − m_new)
C      = f_gate·C + i_gate·(k_t ⊗ v_t)
n      = f_gate·n + i_gate·k_t
q_s    = q_t / sqrt(dqk)
h_num[v] = Σ_qk C[qk,v] · q_s[qk]
h_den  = max(|q_s · n|, exp(−m_new)) + eps      # eps 1e-6
h_t    = h_num / h_den
```
The denominator (`max(|q·n|, exp(−m)) + eps`) is the subtle part — copy it
exactly. Zero-init state; no first-step special case (unlike sLSTM).

Layer wrapping (tirex2 mlstm_block.py): optional CausalConv1d+SiLU → q,k from
conv path (Linear, qk_dim = D·qk_dim_factor), v from RAW x (v_dim =
D·v_dim_factor); i/f gate preacts are per-head SCALARS Linear(D→NH, bias=True)
passed through `soft_cap(x,c) = c·tanh(x/c)` (gate_soft_cap) BEFORE the cell;
o gate = sigmoid(Linear(D→v_dim, per-element)) applied to the multihead-normed
h; out_proj v_dim→D. use_rope=False for the bi-mlstm time mixer. Verify at
oracle time: whether the q/sqrt(dqk) scaling sits inside the backend in
xlstm_large exactly as in the MLX cell (her cell scales inside).

Her repo is also the pattern source for: checkpoint config inference
(`mlx_jit/utils/infer_config_from_safetensors.py`), weight loading, and the
per-cell decomposition (projection cell / recurrent kernel cell / output
cell / neuron) — mirror that decomposition in the C++ kernel.

## Numerics tier

Faithful (bf16/fp32 bit-matched vs torch CPU reference where feasible; ulp
band per module elsewhere). Torch CPU = fp32 with fp32 reductions in norms.
-ffp-contract=off (house rule; clang fuses fma otherwise). logsigmoid: use
log1p(exp(−|x|)) − max(x,0)… NO — match torch's logsigmoid formulation:
torch computes -softplus(-x) numerically as min(0,x) − log1p(exp(−|x|)).
Verify against torch elementwise before trusting.

## Kernel plan (rungs, each gated)

1. **Oracle** (blocked on HF gate): dump per-module tensors from torch CPU on
   fixed seeds — scaler out, patch embed out, per-block time/variate mixer
   out, final quantiles. Script goes in tirex-2 clone, `oracle_dump.py`.
2. **Serial C++ correctness**: this directory, plain scalar loops, structs
   below; parity vs oracle per module then end-to-end.
3. **NEON**: pointwise gates vectorize over head_dim lanes (fixed d, e.g. 4×
   f32x4); headwise projections + Ry as small per-head GEMVs (AMX cblas for
   the big MLPs, hand NEON for d×4d recurrent matvec).
4. **kcoro mount**: lanes-as-heads (FlashRNN shape). Per lane: pinned R block
   + gate projections for its heads; per-stream states resident. Fan-out per
   patch via engine grid; fences at block boundaries only (two-barrier
   doctrine holds — no channel per timestep).
5. **State carry + hibernation blob**: explicit (y,c,n,m)×heads per sLSTM
   block, (C,n,m)×heads per mLSTM block, conv rings, scaler (loc,scale),
   patch remainder buffer. Serialize = memcpy of one struct; this IS the
   spec-10 conversation-image easy case.
6. **Referee harness**: feed three synthetic coupled traces (cointegrated →
   converging; independent → not), assert quantile-band behavior matches
   torch reference; then live agent traces offline replay.

## Layout decisions

- Streams are independent → one stream per kcoro coroutine is legal, but the
  batching win (weight stream amortization) wants batch-of-streams per pass:
  batch dim in the state structs from day one (SoA: states[stream][head][d]).
- Weight table mmap'd from a converted flat blob (safetensors → our packer),
  same discipline as the LFM engine: no per-call repacks, pointers stable for
  process lifetime.

## Continuation pointers (if context is lost)

- tirex-2 model code: `/Volumes/stuff/Projects/tirex-2/src/tirex2/model/`
- flashrnn vanilla cell: `.venv/lib/python3.12/site-packages/flashrnn/flashrnn/vanilla/slstm.py`
  + `__init__.py` (forward/step drivers).
- xlstm mLSTM backend: `.venv/lib/python3.12/site-packages/xlstm/xlstm_large/model.py`
- **House reference (the shape to work from): Sydney's xLSTM-Metal port**,
  `/Volumes/stuff/Projects/mlxports/xLSTM-metal` — MLX is the live tree
  (`xlstm_metal/mlx_jit/`), torch_native is the cross-backend parity twin.
  mLSTM math transcribed above from her recurrent kernel cell. No TiRex
  there — cells and 7B only.
- Unread yet: attention_block.py (variate mixer), postprocessor.py,
  layernorm.py, xlstm FeedForward/RMSNorm/MultiHeadLayerNorm/CausalConv1d
  exact defs vs her MLX twins, checkpoint model-config.
- Do not add fallbacks. Hard-error on unsupported config values.
