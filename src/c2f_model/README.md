# `src/c2f_model/` — the C2F model

`C2FForCausalLM` is **Qwen3 with three modifications**:

1. **Per-scale absolute embeddings** (`C2FScaleEmbedding`) replace RoPE — token
   position is determined by which *scale block* it belongs to, not its index in
   the sequence.
2. **Block-prefix attention mask** (`create_c2f_block_causal_mask`) replaces the
   causal mask — tokens at scale `k` attend to all coarser scales but are
   independent within the same scale.
3. **Unshifted cross-entropy loss** — labels align 1:1 with input positions
   rather than predicting the next token.

For the *why* behind these choices (ELBO derivation, posterior-collapse
prevention, links to VAR / Block Diffusion), see the **Algorithm** and **Model
Architecture** sections of the top-level `README.md`. This README is the bridge
from algorithm → code.

## How to read this code

Read in this order — each piece motivates the next:

1. **`create_c2f_block_causal_mask`** in `modeling.py` — the heart of the model.
   Builds an additive bias mask (B, 1, T, T) where each scale block can attend
   only to *coarser* scales. Once you understand this mask, the rest follows.
2. **`C2FScaleEmbedding`** — adds a learned vector per scale to every token in
   that scale. Replaces what RoPE does in standard Qwen3.
3. **`C2FModel.forward`** — the main forward. Read past the import-time shims
   for `auto_docstring`, `capture_outputs`, `merge_with_config_defaults` (these
   exist for transformers 4.51↔4.52 compat).
4. **`C2FAttention`** — minimal: just `Qwen3Attention` with RoPE pulled out.
   Skim it; nothing surprising.
5. **`C2FForCausalLM.forward`** — wraps `C2FModel` with the unshifted CE loss.

## The `# C2F:` annotation convention

Every line in `modeling.py` that diverges from upstream
`transformers.models.qwen3.modeling_qwen3` is marked with a `# C2F:` comment.
This makes the upstream diff bidirectional and auditable.

**To upgrade the base model** (when transformers releases a new Qwen3):

```bash
# Diff against upstream Qwen3 to see what changed there
diff <(python -c "from transformers.models.qwen3 import modeling_qwen3; \
  print(open(modeling_qwen3.__file__).read())") src/c2f_model/modeling.py
```

Re-apply each `# C2F:` block onto the new upstream version. When you *remove* a
line from upstream, leave a `# C2F: removed — <original line>` stub so the
removal is also tracked.

## Gotchas

- **Flash Attention 2 is disabled** (`attn_implementation="eager"`). FA2 silently
  ignores additive masks, which would break block-prefix attention. Don't enable
  it — the speed gain is not worth the silent correctness regression.
- **`mask_embedding`** in `C2FScaleEmbedding` exists to break the residual-stream
  shortcut under unshifted loss. Without it, the model can copy embeddings
  through residuals and the loss collapses without learning.
- **Mask cache key** is `(seq_len, device, dtype)`. If you change attention dtype
  on the fly, the cache will miss; keep dtype stable across batches.
- **Two `mask_type` modes coexist**: `"block"` (default, C2F semantics) and
  `"causal"` (AR baseline used in steps 8 and 9). Both code paths must keep
  working — when modifying `C2FModel.forward`, run scripts 08 and 09 in both
  modes as a smoke test.
