# Parameter Golf — SOTA parallel workstreams

**Branch:** `sota-parallel-workstreams`  
**Goal:** Beat the current **main-track** README SOTA (see root `README.md` leaderboard; as of this doc, **~1.0810 bpb** on the SP8192 + recurrence + parallel residuals + legal TTT line).  
**You have not run a baseline yet** — Phase 0 is mandatory.

This file is the **single source of truth** for *what we are doing*, *who owns which questions*, and *how work runs in parallel*. Cursor rules in `.cursor/rules/parameter-golf-sota-workstreams.mdc` point agents here.

---

## 1) Constraints (non-negotiable)

| Constraint | Target |
|------------|--------|
| Training time | ≤ 600s wall on **8×H100** (SXM for final record) |
| Eval time | ≤ 600s (sliding + TTT if used) |
| Artifact | ≤ **16,000,000** bytes (code + compressed weights; `train_gpt.py` is the counted code surface per README) |
| Record margin | Beat prior SOTA by **≥ 0.005 nats** with **p < 0.01** (usually **3 seeds** + logs) |
| Fairness | No val leakage; legal TTT = **score-before-update** and rules in README / Issue #1017 |

---

## 2) Roles (Researcher guides everyone)

Work is **parallel**; order of operations is **gated** (see §5).

| Role | Responsibility | Primary outputs |
|------|----------------|-----------------|
| **Researcher** | Priority of ideas, ablation order, stop/pivot criteria, paper/technique refs | Short **Research notes** (hypothesis, 1–2 lines expected effect, risk); **backlog ranked** P0/P1/P2 |
| **Developer** | Implement changes **only** in a new `records/track_10min_16mb/<RUN_ID>/` tree (copy from a known SOTA `train_gpt.py`), feature flags / env vars | Working `train_gpt.py`, reproducible command block in that folder’s README |
| **Tester** | Correctness + stats: `val_bpb`, seed variance, train/eval seconds, artifact bytes; tokenizer byte-audit if tokenizer changes | Logs per seed, comparison table vs baseline, **go/no-go** for merge to “candidate SOTA” |
| **Engineer** | Throughput, GPU util, eval wall time, packaging (compression, `submission.json`), RunPod/H100 reality | Meets **600s + 600s + 16MB** with margin; documents bottlenecks |

**Handoff rule:** Researcher does **not** land code without a **Dev owner**; Dev does not mark “done” without **Tester** sign-off on metrics; **Engineer** approves **time/size** budgets.

---

## 3) Parallel workstreams (run at the same time after Phase 0)

Each stream has its own **hypothesis** so people are not blocked on one experiment.

| ID | Stream | Focus | Typical owner metaphor |
|----|--------|--------|---------------------------|
| **A** | Training dynamics | EMA decay, warmdown shape/fraction, parallel-residual **start layer** (e.g. 5 vs 7 vs 9), QK-gain / WD / MLR sweeps | Researcher + Dev + Tester |
| **B** | Quantization + artifact | SDClip **k** (or per-block), act-order GPTQ, group size vs **compressed** payload, full-Hessian on worst layers | Researcher + Dev + Engineer |
| **C** | Legal TTT | Gated step budget (hard vs easy chunks), LoRA-rank (if 16MB allows), KL anchor to pre-TTT logits, grad clip / per-layer norm after step | Researcher + Dev + Tester |
| **D** | Data / tokenizer (optional, later) | SP8192 vs compact vocab path (e.g. Scylla-style) — **only** after A/B on fixed tokenizer OR if Researcher elevates; **higher review risk** for `val_bpb` proof | Researcher + Dev + Tester + **extra** byte audit |

Streams **A** and **B** are the default first split (low process risk). **C** if baseline already uses TTT. **D** is a **bold** track, not the default start.

---

## 4) High level: what we are **not** changing vs what we **introduce**

### Baseline we copy from (starting point)

- Use an existing top **record** under `records/track_10min_16mb/` as the **code baseline** (e.g. the README’s current SOTA run folder, once you identify it by date/name).  
- **Do not** treat root `train_gpt.py` as the SOTA product — the competition ships **per-record** `train_gpt.py` in `records/…`.

### What we are **not** doing ( unless Researcher re-prioritizes )

- Rewriting the whole repo or merging unvetted “new architecture from scratch” before a **record-class baseline** reproduces.  
- Using validation data for **hyperparameter or quantization** search (forbidden / bad faith).  
- Adding dependencies that break the **16MB** or **eval** story without Engineer sign-off.

### What we **introduce** (candidates; not all at once)

| Area | Example introductions (each needs its own run folder or clear flags) |
|------|------------------------------------------------------------------------|
| Training | Sweeps: EMA, warmdown length, optional SWA late window; **parallel residual start**; recurrence schedule (fixed vs light curriculum) |
| Optimizer / loss | QK-gain value or **layerwise** schedule; weight decay / Muon already in stack — small deltas only with ablations |
| Quant | Tighter **row/percentile** clip; per-tensor or per-role **bit policy** with **zero** extra float tables in artifact; act-order; group-size by **measured** compressed size |
| TTT (if in stack) | **Chunk difficulty gating**; **KL** to frozen logits; fewer unfrozen params (late layers / LoRA); cap **total TTT wall time** |
| Tokenizer (stream D) | New vocab / retokenize — only with **byte audit** and README proof path |

**Rule:** One main **independent variable** per short run; combine only after each axis shows a win or neutral on **screening seeds**.

---

## 5) Gated phases (order matters)

1. **Phase 0 — Reproduce**  
   - One SOTA `records/…` run: 1 seed minimum to verify pipeline, then 3 seeds for variance.  
   - Deliverable: table (bpb, train s, eval s, bytes) — **this is your baseline row**.

2. **Phase 1 — Screen (parallel)**  
   - Streams A/B (and C if applicable): **1 seed** per config; promote if **~0.002–0.003 bpb** better than baseline (rough screen; Researcher sets threshold).

3. **Phase 2 — Confirm**  
   - Top **≤3** configs × **3 seeds**; check **0.005 nats** / **p** plan with Tester.

4. **Phase 3 — Submit**  
   - Engineer verifies caps; add `submission.json`, logs, README per README submission section.

**Pivot:** If Training plateaus, Researcher moves effort to **Quant** (B) or **TTT** (C). If nats improve but bpb does not, **Data/tokenizer** (D). If TTT destabilizes int6, reduce adaptation or add **KL / LoRA** before more arch.

---

## 6) Artifact locations (convention)

| Item | Where |
|------|--------|
| This plan | `docs/SOTA_WORKSTREAM.md` |
| New attempts | `records/track_10min_16mb/<YYYY-MM-DD>_<short_name>/` |
| Backlog (optional) | `docs/SOTA_BACKLOG.md` (create only if the team wants a living checklist) |

---

## 7) Quick links

- Root `README.md` — rules, submission checklist, leaderboard.  
- `data/README.md` — dataset/tokenizer download.  
- SOTA example write-up (path changes with README): e.g. `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/README.md`.

When in doubt, **Researcher** updates priorities in **§3–§4**; **Tester** holds the bar on **repro + stats**.
