# Parameter Golf — in-depth work log, analysis, and bottlenecks

This document records the full arc of work on this fork: goals, what was built, what was run, PR/competition context, engineering fixes, results, and where things stand. Official rules and submission process remain in the root `README.md` and `docs/SOTA_WORKSTREAM.md`.

---

## 1) Context: goals over time

- **Initial:** Learn the challenge; eventually “stand out” and use the early-career / researcher angle OpenAI advertises in the challenge copy.
- **Mid:** Non-record ideas (e.g. JEPA-style) vs main track; time pressure (e.g. ~6 days); RunPod on 8×H100.
- **Current focus:** **Main-track, record-class** quality on the same stack as the published SOTA: **SP8192**, depth **recurrence**, **parallel residuals**, **QK-gain**, **legal score-first TTT**, **GPTQ int6 + int8 + SDClip + Brotli**, **&lt;16MB** artifact, **≤600s** train on 8×H100.

**Record bar (main `README.md`):** beat prior SOTA by **≥0.005 nats** with statistical significance at **p &lt; 0.01** (typically **3 training runs** + logs). The leaderboard SOTA line we treated as the target is **~1.0810** TTT BPB (3-seed mean) in `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`.

**`SOTA_WORKSTREAM.md` adds:** for the coordinated “main” plan, **eval ≤600s** (sliding + TTT) alongside train ≤600s. That matters for any eval that runs many minutes (e.g. TTT+SLOT).

---

## 2) What exists in this fork (code and git)

- **Branch:** `sota-sweep-v2` (pushed to the user’s GitHub remote), containing the experiment tree below.
- **Primary deliverable:**  
  `records/track_10min_16mb/2026-04-25_GatedAttn_SweepSOTA/`
  - `train_gpt.py` — large single file: forked from the SOTA-style CUDA script and extended with many **env-flagged** techniques (see §3).
  - `run_experiments.sh` — scripted ablations (baseline through “ultimate combo”).
  - `README.md` — research-style writeup (hypotheses, techniques, experiment table).
  - `submission.json` — metadata stub for post-run filling.
- **Other:** `2026-04-09_.../train_gpt_decoded.py` was produced locally from a **lzma** copy of the official record script for reading/compare (not the canonical training source).
- **Root** `train_gpt.py` is the repo tutorial baseline — **not** the competition submission product; per README, SOTA lives under `records/...`.

---

## 3) Techniques implemented (and intent)

All are behind env vars / `Hyperparameters` unless noted. Grouped for clarity.

**Architecture / attention (train)**  
- **Gated attention output:** learnable per-dim scale on attention output before projection (`attn_gate`); listed in `CONTROL_TENSOR_NAME_PATTERNS` for optimizers.  
- **QK-gain, depth recurrence, parallel residuals, XSA, skip gates, etc.:** same family as public SOTA records.

**Training dynamics**  
- **Cosine warmdown** vs linear (`WARMDOWN_TYPE=cosine`).  
- **EMA decay annealing** (`EMA_ANNEAL`): linear ramp of EMA decay over training.  
- **SWA:** collect weight snapshots in warmdown; final blend **50% EMA + 50% SWA** when enabled.  
- **Stochastic recurrence:** DropPath-style scaling only on **recurrence** positions (encoder/decoder indices that repeat); **full** depth at eval. Intended as regularization + novel angle.

**Optimizer**  
- **Polar Express NS:** alternative to Newton–Schulz 5 in Muon backend (hybrid iteration).

**muP (partial)**  
- Wider init `1/sqrt(fan_in)` and logit scale `1 / width_mult`; **no** extra LR on Muon (conflicts with Muon’s normalization).

**Pre-quantization TTT**  
- Short SGD on **held-out training** data after EMA, **before** `serialize` / GPTQ — **not** on validation (val would be illegal for this use).

**Quantization-aware training (late QAT)**  
- STE fake-quant on `CastedLinear` weights in late training, **`clip_range` matched to `matrix_bits`** (e.g. int6 → 31), not a mismatched int5. Class-level `CastedLinear.qat_enabled` toggled by training progress.

**SLOT (eval)**  
- Split **forward** into `forward_hidden` + `compute_logits`; optimize a **per-window** small vector (delta on hidden state before head), **not** a shared delta across a batch of windows (that pattern matches closed PRs for causal / score issues).

**Eval**  
- Legal **TTT** (score chunk / windows first, then train on seen data), optional **SLOT** integrated into TTT path or standalone sliding eval.  
- **Removed** undefined ETLB eval path.  
- **ETLB** env vars may remain in `Hyperparameters` but no active path if removed.

**Time accounting**  
- When `pre_quant_ttt` is on, **extra wall time** is added to the **train** side reserve so the 600s cap still leaves room for GPTQ + that phase (rough formula in code).

---

## 4) Engineering bugs fixed during review (high level)

- **ETLB:** call path removed; function was missing → crash.  
- **Pre-quant TTT on val:** reverted to **train-only** data.  
- **muP init:** was over-aggressive; aligned to `1/sqrt(fan_in)`.  
- **SLOT compliance:** per-window delta (see §3) to avoid cross-window leakage in the same sense as #1350-style objections.  
- **Late QAT:** was int5-style clamp; aligned to **GPTQ `matrix_bits`**.  
- **Stochastic recurrence:** decoder `skip_idx` / loop indexing corrected once.  
- **`CastedLinear.qat_enabled`:** reset after training before downstream eval.  
- **Sliding / SLOT eval:** return with model in **`eval()`**, not `train()`.  
- **Pre-quant TTT + time budget:** dynamic reserve for pre-quant phase so end-to-end train pipeline stays consistent with 600s design.

---

## 5) RunPod and operations (what went wrong, what works)

- **Template:** Use the **official Parameter Golf** RunPod template (README link). Generic images caused **PyTorch / flash-attn / compile** pain (`RecompileLimitExceeded`, wrong APIs).  
- **Data:** `openai/parameter-golf` HF dataset is **gated** — need `huggingface-cli login` **or** use **`python3 data/cached_challenge_fineweb.py --variant sp8192`** with e.g. `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf` as in `data/README.md`.  
- **`DATA_DIR`:** must point to the parent of `data/datasets/fineweb10B_sp8192/` and tokenizers.  
- **`brotli`:** not always preinstalled — **`pip install brotli`** or serialization fails after training.  
- **Wrong script:** `torchrun` must be run from the **record folder** that contains your submission `train_gpt.py`; running from repo root hits the **wrong** `train_gpt.py` (e.g. sp1024 tokenizer paths).  
- **Logs / disk:** `mkdir -p logs` before `tee`; use **network volume** if you need persistence after pod stop (container disk is ephemeral).  
- **Charging:** GPU billed while running; storage cheaper than long 8×H100 sessions.

---

## 6) Experiments actually run (user’s pod) — numbers

**Training** stops at **~588s** wall + reserve (not 20k steps): `max_wallclock_seconds=600` minus GPTQ (and pre-quant) reserve. Typical **~4.5k–4.6k** steps in one log.

| Label | What varied | TTT BPB (main headline) | Notes |
|--------|-------------|---------------------------|--------|
| **A0 / baseline** | “Minimal” new flags, TTT on | **~1.0804** | Pre-quant ~1.087; sliding ~1.0817; **beats single-seed read of 1.0810 SOTA** but not a 3-seed mean |
| **A10 SWA** | SWA+EMA 50/50 | **~1.0805** | ~neutral vs A0 |
| **A11 Late QAT** | STE int6 late | **~1.0803** | Slight help vs A0 on TTT line |
| **A14 Stoch recurrence** | drop prob 0.2 | **~1.0823** | **Worse** — too aggressive |
| **TTT+SLOT+“novel” stack** (long TTT) | many flags incl. SLOT | **~1.22** TTT+SLOT | **Catastrophic**; also **~36 min** eval — violates intended **&lt;600s eval** story; do not use as submission candidate |

**Not completed in the chained run:** “A15 ultimate” (command chain / environment issue).

**Gaps vs a real record PR**  
- No **3-seed, same script as official `2026-04-09` record** in isolation.  
- Mega-script mixes many flags; hard to attribute effects cleanly.

---

## 7) PR and leaderboard observations (from repo + merged records)

- **Magnitude of wins:** Strong PRs move on the order of **~0.002–0.01 BPB** (often quoted with **nats/token** and **3-seed means**). Promising **~1.07** in one jump is not aligned with the public board.

- **Single-axis stories:** e.g. QK-gain 5.0, legal TTT on an existing stack, depth recurrence + parallel res — each record tends to be **one main idea** (plus tuned hypers), not ten simultaneous novelties.

- **Issue #1017 (Track B):** **Score before update** on each chunk; no illegal lookahead; **single pass** scoring per token. Any eval scheme must be checkable against these.

- **SLOT / shared adaptation:** PRs like **#1350** were **closed** when adaptation leaked across causal windows. **Per-window** SLOT is the minimal fix for that *class* of objection; the **1.0810 SOTA README** still lists **“No SLOT”** in compliance — so SLOT is **high review risk** and was **not** part of the winning line.

- **Pre-quant / calibration:** using **val** for adaptation before quant is a red flag; **train-held-out** or self-gen calibration appears in other records with explicit stories.

- **Record checklist:** `README.md` — `train_gpt.py` + `submission.json` + `README` + **train logs**; **&lt;16MB**; **10 min** train on 8×H100; statistical evidence for the nats margin.

---

## 8) What worked vs what did not (hypotheses)

**Worked (or neutral-positive)**  
- Reproducing **~1.08x** TTT BPB in the same **ballpark** as **1.0810** with a working pipeline.  
- **Legal TTT** as the main eval-time gain (same story as SOTA).  
- **Late QAT** in principle (small quant-gap improvement in one ablation) — **if** the STE path is really active (see older record footnote: `torch.compile` can constant-fold `qat_enabled` in some setups — worth verifying in any new PR).  
- **Tight time reserve** for pre-quant TTT + training so wall clock stays coherent.

**Did not help or hurt**  
- **SWA 50/50 with EMA:** ~neutral.  
- **Stochastic recurrence @0.2:** **hurt** — likely over-regularized / too few effective recurrence passes.  
- **Stacking many changes** (ultimate combo): interaction risk; not validated.

**Failed badly**  
- **TTT + SLOT** in the long “novel” run: **absurd TTT+SLOT BPB and loss** — indicates **bug**, wrong interaction, and/or optimizer explosion; not a small tuning issue.  
- **SLOT+TTT** also blew **eval time**; conflicts with **≤600s eval** in `SOTA_WORKSTREAM.md`.

---

## 9) Research / design threads (discussed, not all implemented)

- **GICA / global error injection:** Rejected: **non-causal** / **second-look** risk under #1017 if error is pooled from full-chunk NLL and fed back.  
- **NGC-style “NPA” (Gumbel-Top-k over int4/6/8 per block):** Theoretically interesting for the **quant gap**, but **large** code surface (`serialize` / `dequantize` / byte budget). **Deferred** until a simpler adaptation win is proven.  
- **LoRA-TTT (frozen int6/quantized base, train only small LoRA during legal TTT, state across chunks like today):** Sensible **A/B** vs full-weight TTT; **not implemented** in the tree as of this log. Stated to avoid quant drift; needs fair comparison (same steps, same score-first order).  
- **Dynamic TTT steps by chunk difficulty:** Optional later; only after LoRA/TTT path is stable.

---

## 10) Current bottlenecks and problems (actionable)

1. **No official 3-seed table** on the **canonical 2026-04-09** `train_gpt.py` — cannot claim “beat 1.0810” with full rigor until that exists.  
2. **Experimental script** is a **flag zoo**; record PRs favor **minimal diff** on top of a **known-good** record folder.  
3. **SLOT+TTT** path: **numeric failure** + **time**; **SOTA** does not depend on SLOT — **parking** SLOT until a minimal, correct, **fast** implementation exists.  
4. **0.005 nats + p&lt;0.01** require **power** from multiple seeds; single-seed **~1.0804** is promising but not sufficient.  
5. **LoRA-TTT** and **NPA** are **roadmap** items, not shipped.  
6. **Ops / human error:** easy to run **wrong `train_gpt.py`** or forget **`brotli`**, **`DATA_DIR`**, or **`cd` to record path**.

---

## 11) Recommended order of work (aligned with above)

1. **Reproduce** `2026-04-09_.../train_gpt.py` (or copy into a **new** dated `records/...` folder) with **seeds 42, 314, 999** — match published mean within tolerance; log train/eval seconds and bytes.  
2. **One change at a time** for ablations (QK, warmdown, etc.) on that baseline if still chasing margin.  
3. **Implement LoRA-TTT** only in `eval_val_ttt` (flags), **1-seed** vs full TTT, then **3-seed** if not worse.  
4. **Only then** consider quant allocation (NPA) or dynamic TTT steps, with **byte** and **eval** budgets explicit.

---

## 12) Target design: Quantization-Recovery In-Place TTT Transformer (QRI-IPTT)

**Intent:** The best balance of **novelty, plausibility, and reviewability** for a main-track submission is to keep the **strongest public non–CaseOps transformer surface** (depth recurrence, parallel residuals, QK-gain, standard tokenizer accounting), and unify three high-yield challenge directions so they **hit the same residual errors** instead of acting as separate patches.

**Surface (unchanged where possible).** Standard SP8192 stack and legal eval rules (#1017); no disputed normalization or custom tokenizer math.

**Fast-weight path (replaces default LoRA-TTT as the main TTT path).** **In-place TTT (IPTT):** small low-rank delta on **selected MLP output projections** (activation-side delta on the path into / out of `mlp.proj`), PiSSA-style init, trained only at test time under the same legal TTT protocol as public records. **Env:** `TIT_MODE=iptt`, `TTT_LORA=0` (no parallel full LoRA on attn in this mode).

**Routing / gating (tiny).** **SmearGate** (light blend with the previous position after embed/norm) plus existing **attention-output gating** (`ATTN_OUT_GATE=1`). Optional future: a very light differential-attention–inspired gate; the current code uses Smear + attn-out gate as the “routing” story.

**Quantization repair (one explicit mechanism).** **Rank-4 asymmetric LQER** when artifact headroom allows: merged pre-GPTQ **fake-quant error** compressed by rank-`LQER_RANK` SVD and **fused into weights** (no extra shipped tensors). If headroom is tight, the design allows falling back to a merged **FlatQuant/SpinQuant-style** affine/rotation calibration before GPTQ; the shipped record uses the **merged LQER** path (`PRE_GPTQ_LQER=1`, `LQER_RANK=4`).

**Training objective.** The **main loss remains exact next-token cross-entropy**; any auxiliary must stay **tiny and purely regularizing** (this fork does not add a large auxiliary head loss on the headline CE path).

**Why this variant (challenge evidence, qualitative).** Public evidence has already **isolated** high-yield directions: legal **4-epoch TTT** ~**0.0081 BPB**; **SmearGate + attention-output gating** ~**0.0096 BPB** over a merged line; **alpha-scaled warm-start-A LoRA-TTT** adds a smaller but **monotonic** gain; a **1.06157**-class stack shows **quantization repair** can still recover **meaningful** score after those gains. QRI-IPTT is the first **clean** design that aims all three at **shared** residual error rather than separate patches.

**Expected landing (inferential, not a guarantee).** With tokenizer unchanged, a **plausible** band for a strong run is roughly **1.056–1.066** TTT BPB. **Below ~1.058** needs IPTT to **outperform** prior LoRA-TTT in the **same** eval harness **and** quant repair to recover **most** of the remaining low-bit damage — **plausible, not guaranteed**. The key claim for reviewers is: **strongest path that stays review-clean** without CaseOps, disputed norm, or custom byte accounting for the base model.

**Code / run pointer:** `records/track_10min_16mb/2026-04-25_GatedAttn_SweepSOTA/train_gpt.py` implements IPTT, SmearGate, attn-out gate, and merged LQER; `run_experiments.sh` **A16** is the one-line stack for this design (see that folder’s `README`).

---

## 13) File index (this workstream)

| Path | Role |
|------|------|
| `records/.../2026-04-25_GatedAttn_SweepSOTA/train_gpt.py` | Monolithic SOTA+experiments code |
| `records/.../2026-04-25_GatedAttn_SweepSOTA/run_experiments.sh` | Batch ablations on RunPod |
| `docs/SOTA_WORKSTREAM.md` | Internal process: roles, streams, eval/train caps |
| Root `README.md` | Challenge rules, leaderboard, submission checklist |
| `records/.../2026-04-09_.../README.md` | Prior merged line documentation (~1.0810 class) |
| `records/.../2026-04-25_.../README.md` | QRI-IPTT / GatedAttn sweep record README |

---

*This is a working document for the fork; it is not an official OpenAI or upstream submission file.*
