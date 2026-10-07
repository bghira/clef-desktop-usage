# clef-desktop

Experiment: locate an object on the macOS desktop with Cloudflare's **clef-flash**
decision model by recursive quadrant narrowing, ending in click-precision coordinates.

## Research summary

### What Clef / Clef-flash is

- Cloudflare's first in-house decision models (announced 2026-10-01), post-trained from
  Qwen3.8-27B (`clef`) and Qwen3.5-9B (`clef-flash`) with a joint schema scoring head.
- Instead of generating text, they take a **state** (string/JSON + up to 4 images) and a
  **schema of typed questions**, and return a **probability for every allowed option** of
  every question in a single forward pass. No output parsing, no hallucinated categories.
- Hosted: `@cf/cloudflare/clef-flash` (9B, ~39ms median latency, $0.09/M input tokens,
  65k context) and `@cf/cloudflare/clef` (27B, ~209ms, $0.24/M). Apache-2.0 weights on
  HuggingFace (`Cloudflare/clef-flash`).
- Clef's differentiator vs. Jev (Typesafe) and other decision models: **vision encoder** —
  Jev is text-only; Clef reads PNG/JPEG/WebP images and video frames.

### API essentials (Workers AI REST)

```
POST https://api.cloudflare.com/client/v4/accounts/$ACCOUNT_ID/ai/run/@cf/cloudflare/clef-flash
{
  "model": "clef-flash",
  "state": "<string or JSON describing the situation>",
  "questions": {
    "<qid>": {"type": "noul",   "instructions": "yes/no question"},            # -> {noul: P(yes)}
    "<qid>": {"type": "choice", "instructions": "...", "criteria": {id: desc}}, # -> {choice, probabilities, confidence}
    "<qid>": {"type": "score",  "instructions": "...", "criteria": [levels]}    # -> {score, probabilities, confidence}
  },
  "images": [{"content_type": "image/jpeg", "base64": "..."}]  # or data URLs
}
```

Limits: max 4 images per request; 4 MiB / 16 MP each; 8 MiB total decoded; 13 MiB body;
up to 64 questions; ids `[A-Za-z0-9_.-]`; base64 only (no remote URLs). Response comes
wrapped in the REST envelope `{success, result: {model, answers, usage}}`.

### How others are using decision models for desktop/agent interaction

Clef is two days old, so there is no public "Clef for desktop" pattern yet — we'd be
early. The established ecosystem patterns, which this experiment follows:

- **Decision models on the agent hot path** (the pitch from the launch blog + community
  writeups): LLMs waste capacity on bounded choices. Put a cheap, calibrated decision
  model in the loop for routing/branching/verification, and escalate to a big model (or a
  human) only on low confidence — the returned probabilities are calibrated (Brier loss +
  RLCD training), so thresholds are meaningful.
- **Cascade architecture**: rules → decision model → LLM → human.
- Cloudflare's own adjacent stack: `@cloudflare/computer` (agent runtime with a virtual
  computer in a Durable Object), Browser Run + Clef for classifying rendered pages
  (2.2s fetch+render+classify vs 4.7s with gpt-oss-120b).
- Classical VLM desktop agents (OSWorld-style) use screenshot → structured questions →
  act loops; Clef fits the **verification/branching** steps of such loops extremely well
  (fast, bounded, typed) but is *not* a grounding model — it won't emit bounding boxes or
  coordinates (compare `moondream3.1-9B` on Workers AI for pointing). Our script gets
  coordinates architecturally instead: by recursive halving.

### Our design: quadrant binary search

1. Capture the screen once (`screencapture -x`, native Retina resolution).
2. Each level: ask Clef two questions in one request — `where` (choice over TL/TR/BL/BR)
   and `visible` (noul). Default `--mode full` sends one image of the current region
   (single forward pass per level, ~1.3s on MPS); `--mode crops` sends the four quadrant
   crops as separate images (higher per-region resolution, but a target near a boundary
   can appear in several crops and confuse the choice).
3. Narrow the region to the winning quadrant (tracked in pixel space), repeat until the
   region is ≤ `--stop-px` logical points wide.
4. Final call: a padded close-up (3x the converged region, so the whole target is in
   frame) + `noul` — **"did we locate the item?"**
5. Emit the region center in logical screen points; `--click` sends a mouse click there
   (via `cliclick` if installed, else JXA/CoreGraphics).

Per-level confidence is used as a bail-out: if the winning quadrant drops below
`--min-conf` (default 0.4; uniform is 0.25) the search stops with the best-known region
instead of chasing a hallucination. The first upload is boosted to 1536px for coarse
views so small targets (menu-bar icons ~24pt) stay visible.

### Performance work (Metal kernel + sync removals)

Profiling the local pipeline exposed where the ~3s per forward goes, and two
surprises: the torch profiler's per-op numbers on MPS are unreliable (async
backpressure inflates "self time"), and attention is *not* the bottleneck
(MPS SDPA on our shapes already runs at ~7 TFLOPS effective — head_dim 72 forces
the materializing MATH path everywhere, but that path is near hardware limits).

What actually helped:

- **`metal_chunk.py`** — a Metal kernel compiled at runtime via
  `torch.mps.compile_shader`. The gated-delta chunk scan contains a 63-step
  sequential forward-substitution loop (`(I-L)^{-1}` per 64×64 chunk matrix,
  unbatchable in torch). The kernel does it in **one dispatch per layer**: one
  threadgroup per (head, chunk) matrix, threads 0..63 each owning a column,
  lockstep rows through threadgroup memory, identity folded into write-back.
  10.0ms → 1.8ms per layer-call, ~210ms saved per forward.
- **Sync removals**: `torch_compilable_check` no-op'd (2 device drains per
  forward) and the joint head's mask read routed through CPU (the expensive
  end-of-forward drain). Verified benign: chunk outputs agree to 2.4e-04,
  decisions unchanged.
- End-to-end: forward 3.08s → ~2.9s; locate runs now ~0.8s per level.

Known remaining costs (measured, not yet fixed): vision tower ~1.2s (SDPA ~0.9s
of it is near-irreducible without a custom flash kernel for head_dim 72),
remaining 3 `grid_thw[.].item()` syncs in `get_rope_index`, chunk loop + prep
~0.3s, text projections/conv ~0.4s.

### UMFA: working on macOS 26 with real async copies

The macOS 26 compatibility issue is fully resolved, and the fix is elegant:

- **Root cause (refined)**: the macOS 26+ MSL frontend has a blocklist on
  `__asm("air.…")` labels specifically — custom asm labels compile fine. It is
  not a blanket asm ban, and no MSL language version (probed 2.0–3.2) or
  `MTLCompileOptions` flag re-enables the `air.` prefix.
- **The fix**: prefix the label with byte 0x01 — LLVM's standard "do not
  mangle" escape, spelled `\001` in a Metal string literal
  (`__asm("\001air.…")`). This bypasses the frontend blocklist while producing
  the identical intrinsic call, and the 26.6 backend still lowers the original
  `air.simdgroup_async_copy_*` ops. Kernels compile through the normal JIT
  path — **no Xcode or Metal toolchain required**.
- Correctness: matches torch SDPA to ~3e-7 (fp16, single- and multi-head).
- **Discovered along the way**: Xcode 27's Metal Toolchain 27A *removed* the
  old ops entirely (renamed to workgroup-scoped builtins
  `__metal_async_wg_copy` / `__metal_wait_wg_events` = `air.async_wg_copy` /
  `air.wait_wg_events`); the 26.6 runtime does not lower those, so the new-tool
  path is a dead end on this OS. The `\001` escape + JIT is the correct
  solution for macOS 26.
- Also found an upstream bug: the first MFA call in a process being
  genuine-fp16 multi-head page-faults the GPU; a small fp32 call first works
  around it.
- **Fair GPU-to-GPU benchmarks** (multi-head, reusable device buffers):

| shape | umfa fp16 (sync) | umfa fp16 (async) | sdpa bf16 (MPS) |
| --- | ---: | ---: | ---: |
| 16×2304×72 (vision) | 10.0 ms | 10.9 ms | **5.4 ms** |
| 16×5952×72 (vision @1536px) | 54.4 ms | **42.0 ms** | 34.8 ms |
| 16×1653×256 (text) | 109 ms | **13.9 ms** | 8.3 ms |

  Async restores MFA's design performance (8× on the strided text loads vs
  sync mode). MPS SDPA remains competitive on our shapes; MFA's further upside
  is in its quantized (int8/int4) modes and much longer sequences.

### UMFA in-process integration (zero-copy, encode path) — working

The "warmup bug" turned out to be three separate bugs, all fixed:

1. **The real bug (MFA's Python FFI)**: the forward kernel **always writes O as
   FP32 in memory** (hardcoded in `AttentionDescriptor+Precisions.swift`), but
   the FFI allocated the output as `zeros_like(q)` — half the size for 16-bit
   inputs. The kernel overflowed the buffer: page faults on first use ("the
   warmup bug"), garbage afterwards. Fixed in `umfa/core.py` (FP32 output +
   cast back) and by contract in our wrappers. There was never a warmup bug.
2. **Per-instance pipeline cache**: `MultiHeadAttention.pipelineCache` was an
   instance property, and the encode path creates a fresh instance per call —
   recompiling the pipeline every call, with the first dispatch after
   compilation producing nothing (the driver quirk the MFA source itself
   warns about). Made the cache `static` (global) with a lock.
3. **didModifyRange hazard**: `mfa_buffer_from_ptr` marks memory CPU-modified;
   for buffers whose contents are produced by GPU commands later in the same
   command buffer, this corrupts the data. Added `mfa_buffer_from_ptr_gpu`
   (no didModifyRange) for the encode path.

Plus new infrastructure:
- `mps_stream.py` — ctypes access to PyTorch's MPS command stream
  (`getCurrentMPSStream`, `commandBuffer`, `endKernelCoalescing`,
  `commitAndContinue`, `commitAndWait`) via exported libtorch symbols
  (single-underscore dlsym names).
- `umfa_encode.py` — `umfa_sdpa_encoded`: encodes the MFA kernel **onto
  PyTorch's MPS command buffer** (zero-copy via `torch.mps._host_alias_storage`
  + `mfa_buffer_from_ptr_gpu`; bf16 inputs native on Apple9, no conversions).

**Integration contract discovered by experiment**: foreign encoders appended
to a continued command buffer only execute reliably if the buffer is committed
and waited *after* the encode (`commit_and_wait`). With `commitAndContinue`
instead, all calls after the first produce silent garbage — on this driver.
So the path costs one host wait per attention call.

**Result**: correct end-to-end (max prob diff 0.0030 vs baseline, same
argmax), but still ~13% slower than MPS SDPA overall — MFA's kernel itself is
slower at these shapes, and the wait serializes the pipeline. The GQA
limitation (text layers unrouted) also remains.

### Clef Flash with UMFA attention (tested — net loss, kept MPS)

Wired UMFA into the live model as a drop-in for the vision tower's SDPA calls
(zero-copy via `torch.mps._host_alias_storage` + MFA's `bytesNoCopy` shared
buffers — MPS storage is host-visible, so the kernel reads the model's actual
pages; verified bit-exact aliasing):

- **Root cause of the earlier NaNs (resolved)**: MFA's forward kernel always
  writes the output O as FP32 in memory (hardcoded in
  `AttentionDescriptor+Precisions.swift`), but the Python FFI allocated the
  output as `zeros_like(q)` — half the size for 16-bit inputs, so the kernel
  overflowed the buffer (the "page fault on first call" was this too). With an
  FP32 output buffer, fp16 AND bf16 paths are correct and deterministic
  (maxdiff 0.001-0.03, matching documented rounding). BF16 runs natively on
  Apple9 with zero input conversion.
- The model's text layers use GQA (16 q / 4 kv heads) which the standard MFA
  bridge doesn't support (`numKVHeads` is hardcoded to `numHeads`), so only the
  vision tower (head_dim 72, no GQA) can be routed.
- Non-contiguous vision inputs (post-transpose views) need explicit
  `.contiguous()` before zero-copy aliasing.

**Result**: forward 4.89s (MPS baseline) → 5.57s with UMFA vision attention
(+14%) using direct BF16 zero-copy. MFA's kernel (~42ms at 5952×72 fp16/bf16)
still trails the SDPA math path (~30ms), and each UMFA call needs a host
synchronization on both sides (zero-copy aliasing reads raw memory), which
serializes the pipeline 27 times per forward. Model decisions unchanged
(max probability diff 0.003, same argmax). The remaining gap is closeable via
MFA's `mfa_attention_encode_mtl` (encodes onto PyTorch's MPS command buffer —
no host syncs), but that needs a small torch extension to reach
`MPSStream::commandBuffer()`, which has no Python binding.

**When to revisit**: if MFA ships a fixed fp16 path, or for much longer
sequences where flash-style kernels pull ahead of the materializing math path.
The zero-copy bridge (`umfa_sdpa.py`) is reusable as-is.

### Schema lessons learned (important)

Empirical findings from debugging the live `apple menu icon` case:

- **Binary existence questions are unreliable in cle-flash.** Asked "is X visible?"
  (noul) or "is X visible?" (yes/no choice), the model answers *no/absent* at 0.8–0.9+
  even when X is plainly visible and identified correctly by other question forms.
  Do not gate on existence questions.
- **Spatial choices are robust.** "Which quadrant contains the target?" tracks reliably
  (0.4–0.95 per level, stable across levels).
- **"What is here?" identity choices work for concrete visual phrases** ("the red
  circle" → 0.97) but fail for abstract UI jargon ("apple menu icon" → 0.04). The final
  check therefore asks *both* the quadrant and identity questions in one pass and takes
  the max as `did_we_locate`.
- Verified end-to-end: `apple menu icon` converges to (28, 18)pt — the actual Apple logo
  position — with `did_we_locate=0.60`; the synthetic circle test confirms at 0.97.

## Usage

Two backends: the hosted Workers AI model, or **fully local** on this Mac using the
int8 ConvRot release in `../cloudflare-clef-flash` (no API keys needed).

### Local (MPS, real weights)

```sh
# Terminal 1: load the local release and serve the native SystemOne schema API
# (one-time bake: packed int8/6/4 weights decoded to bf16 via the release's own
#  unpacker, then moved to MPS; ~25s, then every request is a real joint-head pass)
python3 serve_local.py --port 8790

# Terminal 2: recursive quadrant narrowing against the local server
python3 locate.py "the Safari reload button" --url http://127.0.0.1:8790/v1/systemone --save ./captures
python3 locate.py "the red circle" --image ./test.png --url http://127.0.0.1:8790/v1/systemone --click
```

Measured on M5 Pro (48GB, MPS, transformers 5.12.1 torch path): ~1.3s per level,
~9s for a full 7-call locate run. Deterministic across runs.

Note on what "local" means here: the calibrated CuTe/W8A8 integer-GEMM runtimes in the
release are L40S/SM89-only (`clef_flash.py` hard-checks the GPU), so on Apple Silicon we
use the release's pure-torch packed loader (`clef_flash_packed.load_checkpoint(backend="torch")`)
and bake the packed weights to bf16 once. Those are the shipped weights decoded by the
official unpacker — the accuracy hit vs native BF16 is only the release's own RTN
quantization (their eval: 509/512 decision agreement).

### Hosted (Workers AI, needs credentials)

```sh
export CLOUDFLARE_ACCOUNT_ID=...
export CLOUDFLARE_API_TOKEN=...        # needs Workers AI run permission

python3 locate.py "the Safari reload button"
python3 locate.py "Finder search icon" --model clef --depth 6 --save ./captures
python3 locate.py "the trash can in the Dock" --click          # clicks at the end
python3 locate.py "whatever" --dry-run                          # mock decider, no API calls
python3 locate.py "whatever" --image ./existing.png             # analyze an image instead
```

Permissions needed on macOS:
- **Screen Recording** for the terminal app you run this from (first `screencapture` run
  will prompt; without it you get `could not create image from display`).
- **Accessibility** permission for `--click` (cliclick/JXA event posting).

Flags: `--model clef|clef-flash`, `--depth N`, `--stop-px P`, `--min-conf X`,
`--img-px N` (upload size), `--save DIR` (crops + trace.json), `--image PATH`,
`--click`, `--dry-run`.

## Caveats / next steps

- Deep zooms lose global context; if the model drifts, per-level `visible` catches it.
  A hybrid mode (full-screen image + zoomed region image in one request, 4-slot limit
  permitting) is worth trying.
- Uploading 4 crops per level costs network time on top of the ~40ms model latency;
  a `single` mode (one image, quadrant choice) trades resolution for speed.
- The output is a region, not a pixel-perfect bbox; clicking the center is right for
  buttons/icons, wrong for large targets. An ask-Clef-vs-ask-moondream comparison is the
  natural follow-up.
- Clef's RL fine-tuning platform (announced same day) could specialize a Clef on UI
  widget vocabulary for higher quadrant accuracy at depth.
