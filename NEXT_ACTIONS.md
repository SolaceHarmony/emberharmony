# Immediate Actions - High-Value Files

Based on AST analysis, here are the concrete next steps.

## Summary

- **Files Present:** 16/64 (25.0%)
- **Function parity:** 82/571 matched (target 186) — 14.4%
- **Class/type parity:** 0/0 matched — N/A
- **Combined symbol parity:** 82/571 matched (target 186) — 14.4%
- **Average inline-code cosine:** 0.23 (function body across 12 matched files)
- **Average documentation cosine:** 0.00 (doc text across 12 matched files)
- **Cheat-zeroed Files:** 1
- **Critical Issues:** 16 files with <0.60 function similarity

## Priority 1: Fix Incomplete High-Dependency Files

No incomplete high-dependency files detected.

## Priority 2: Port Missing High-Value Files

Critical missing files (>10 dependencies):

No missing high-value files detected.

## Detailed Work Items

Every matched file is listed below with function and type symbol parity.

### 1. models.loaders

- **Target:** `loader`
- **Similarity:** 0.00
- **Dependents:** 3
- **Priority Score:** 3111110.0
- **Functions:** 0/11 matched (target 8)
- **Missing functions:** `hf_get`, `from_hf_repo`, `get_mimi`, `get_moshi`, `get_text_tokenizer`, `_is_safetensors`, `get_moshi_lm`, `get_conditioner`, `get_conditioner_provider`, `get_condition_fuser`, `get_lora_moshi`
- **Types:** 0/0 matched
- **Missing types:** _none_
- **Lint issues:** 2

### 2. model.transformer

- **Target:** `model.transformer`
- **Similarity:** 0.16
- **Dependents:** 1
- **Priority Score:** 1041308.4
- **Functions:** 9/13 matched (target 28)
- **Missing functions:** `__init__`, `forward_cached`, `_norm`, `_validate_cache`
- **Types:** 0/0 matched
- **Missing types:** _none_

### 3. model.lfm2_audio

- **Target:** `model.lfm2_audio [STUB]`
- **Similarity:** 0.00
- **Dependents:** 1
- **Priority Score:** 1040910.0
- **Functions:** 5/9 matched (target 25)
- **Missing functions:** `__init__`, `from_pretrained`, `forward`, `_sample_text_token`
- **Types:** 0/0 matched
- **Missing types:** _none_
- **Lint issues:** 3

### 4. conformer.mha

- **Target:** `conformer.mha`
- **Similarity:** 0.37
- **Dependents:** 1
- **Priority Score:** 1010806.3
- **Functions:** 7/8 matched (target 15)
- **Missing functions:** `__init__`
- **Types:** 0/0 matched
- **Missing types:** _none_

### 5. conformer.modules

- **Target:** `conformer.modules`
- **Similarity:** 0.17
- **Dependents:** 1
- **Priority Score:** 1010508.3
- **Functions:** 4/5 matched (target 11)
- **Missing functions:** `__init__`
- **Types:** 0/0 matched
- **Missing types:** _none_
- **Lint issues:** 2

### 6. model.mlp

- **Target:** `model.mlp`
- **Similarity:** 0.26
- **Dependents:** 1
- **Priority Score:** 1010207.4
- **Functions:** 1/2 matched
- **Missing functions:** `__init__`
- **Types:** 0/0 matched
- **Missing types:** _none_

### 7. liquid_audio.utils

- **Target:** `utils`
- **Similarity:** 0.35
- **Dependents:** 1
- **Priority Score:** 1000406.5
- **Functions:** 4/4 matched (target 9)
- **Missing functions:** _none_
- **Types:** 0/0 matched
- **Missing types:** _none_

### 8. liquid_audio.processor

- **Target:** `processor`
- **Similarity:** 0.25
- **Dependents:** 0
- **Priority Score:** 62107.5
- **Functions:** 15/21 matched (target 26)
- **Missing functions:** `__init__`, `from_pretrained`, `rename_layer`, `__repr__`, `__getitem__`, `add_audio`
- **Types:** 0/0 matched
- **Missing types:** _none_
- **Lint issues:** 3

### 9. conformer.processor

- **Target:** `conformer.processor`
- **Similarity:** 0.25
- **Dependents:** 0
- **Priority Score:** 51107.5
- **Functions:** 6/11 matched (target 17)
- **Missing functions:** `__init__`, `stft`, `log_zero_guard_value_fn`, `get_seq_len`, `normalize_batch`
- **Types:** 0/0 matched
- **Missing types:** _none_
- **Lint issues:** 7

### 10. conformer.encoder

- **Target:** `conformer.encoder`
- **Similarity:** 0.22
- **Dependents:** 0
- **Priority Score:** 21807.8
- **Functions:** 16/18 matched (target 19)
- **Missing functions:** `__init__`, `_calc_context_sizes`
- **Types:** 0/0 matched
- **Missing types:** _none_
- **Lint issues:** 11

### 11. conformer.subsampling

- **Target:** `conformer.subsampling`
- **Similarity:** 0.33
- **Dependents:** 0
- **Priority Score:** 11306.7
- **Functions:** 12/13 matched (target 17)
- **Missing functions:** `__init__`
- **Types:** 0/0 matched
- **Missing types:** _none_
- **Lint issues:** 3

### 12. liquid_audio.detokenizer

- **Target:** `detokenizer`
- **Similarity:** 0.15
- **Dependents:** 0
- **Priority Score:** 10208.5
- **Functions:** 1/2 matched (target 7)
- **Missing functions:** `__init__`
- **Types:** 0/0 matched
- **Missing types:** _none_

### 13. conformer.utils

- **Target:** `conformer.utils`
- **Similarity:** 0.28
- **Dependents:** 0
- **Priority Score:** 207.2
- **Functions:** 2/2 matched
- **Missing functions:** _none_
- **Types:** 0/0 matched
- **Missing types:** _none_

### 14. demo.model

- **Target:** `model.mod [STUB]`
- **Similarity:** 1.00
- **Dependents:** 0
- **Priority Score:** 0.0
- **Functions:** 0/0 matched
- **Missing functions:** _none_
- **Types:** 0/0 matched
- **Missing types:** _none_

### 15. liquid_audio.__init__

- **Target:** `lib [STUB]`
- **Similarity:** 1.00
- **Dependents:** 0
- **Priority Score:** 0.0
- **Functions:** 0/0 matched
- **Missing functions:** _none_
- **Types:** 0/0 matched
- **Missing types:** _none_

### 16. conformer.__init__

- **Target:** `conformer.mod [STUB]`
- **Similarity:** 1.00
- **Dependents:** 0
- **Priority Score:** 0.0
- **Functions:** 0/0 matched
- **Missing functions:** _none_
- **Types:** 0/0 matched
- **Missing types:** _none_

## Success Criteria

For each file to be considered "complete":
- **Similarity ≥ 0.85** (Excellent threshold)
- All public APIs ported
- All tests ported
- Documentation ported
- port-lint header present

