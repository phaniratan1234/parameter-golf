# Record: Quantization-Recovery In-Place TTT Transformer (QRI-IPTT)

**Target line:** A review-clean submission that unifies **legal TTT**, **gating / Smear**, and **quantization repair** on the same residual errors—without CaseOps, disputed normalization, or custom tokenizer byte accounting.

**Primary design (this folder implements):**

| Mechanism | Role | Key env |
|-----------|------|---------|
| **In-place TTT (IPTT)** | Replaces default **LoRA-TTT** as the main fast-weight path: low-rank delta on **selected MLP output** paths (PiSSA init), legal TTT only | `TIT_MODE=iptt` `TTT_LORA=0` |
| **SmearGate** | Tiny “routing”: light blend with previous position (post-embed / norm) | `SMEAR_GATE=1` |
| **Attention-output gating** | Learnable per-dim scale on attention output (public PR pattern) | `ATTN_OUT_GATE=1` |
| **Pre-GPTQ LQER (rank-4)** | One explicit **quantization-repair** step: fake-quant error → rank-`k` SVD → **merged into** linear weights (0 extra artifact params) | `PRE_GPTQ_LQER=1` `LQER_RANK=4` |
| **Training loss** | Exact next-token **cross-entropy**; no large auxiliary on the headline path | (default) |

**Expected landing (inferential; tokenizer unchanged):** a plausible band for a competitive run is roughly **1.056–1.066** TTT BPB. Getting **below ~1.058** needs IPTT to beat prior **LoRA-TTT** in the same harness **and** LQER to recover most low-bit damage—**plausible, not guaranteed**.

**Evidence narrative (from public challenge history, qualitative):** ~**0.0081 BPB** from legal 4-epoch TTT; ~**0.0096 BPB** from Smear + attn-out gating on a merged line; monotonic but smaller gain from **alpha warm-start-A LoRA-TTT**; **~1.06157**-class stacks show **quant repair** can still help after other gains. This record aims to align those mechanisms on **one** design instead of separate patches.

**How to run (core only — no Polar / pre-quant TTT / late QAT / SWA / EMA):** **`run_experiments.sh`** experiment **`A16_iptt_smear_lqer`**, or the minimal **`env` + `torchrun`** below. Set **`DATA_DIR`** to the repo root (so `data/datasets/...` and `data/tokenizers/...` resolve). **`run_experiments.sh`**’s **`COMMON_ENV`** still turns on legal TTT (`TTT_ENABLED=1`, chunk 2048, adaptive) — that is the required eval path, not optional extras.

```bash
REPO="/workspace/parameter-golf"
export DATA_DIR="$REPO"
cd "$REPO/records/track_10min_16mb/2026-04-25_GatedAttn_SweepSOTA"
mkdir -p logs
env TTT_ENABLED=1 TTT_LORA=0 TTT_CHUNK_TOKENS=2048 TTT_ADAPTIVE=1 TTT_LR=0.005 TTT_EPOCHS=2 \
  TIT_MODE=iptt QK_GAIN_INIT=5.25 ATTN_OUT_GATE=1 SMEAR_GATE=1 PRE_GPTQ_LQER=1 LQER_RANK=4 LAYER_SCALE_INIT=1e-4 \
  POLAR_EXPRESS=0 PRE_QUANT_TTT=0 LATE_QAT_ENABLED=0 SWA_ENABLED=0 EMA_ANNEAL=0 MUP_ENABLED=0 STOCHASTIC_RECURRENCE=0 \
  SEED=42 \
  torchrun --standalone --nproc_per_node=8 ./train_gpt.py \
  2>&1 | tee logs/A16_qri_iptt_core_seed42.log
```

---

## Stochastic Recurrence (earlier ablation thread)

**Note:** The following section described an **additional** ablation (stochastic depth on recurrence). It is **not** part of the default QRI-IPTT submission story unless explicitly enabled by env.

**Target (historical text):** Beat current SOTA of 1.0810 BPB (3-seed mean)

## Novel Contribution: Stochastic Recurrence

Our primary novel contribution is **Stochastic Recurrence** — applying DropPath (stochastic depth) specifically to the recurrence iterations of depth-recurrent transformers.

**Key insight:** In weight-shared recurrence, randomly zeroing the contribution of loop iterations during training while using all iterations at evaluation provides three simultaneous benefits:
1. **Regularization** — the model learns to produce useful representations at any recurrence depth, acting as an implicit ensemble of sub-networks with different depths
2. **Gradient strengthening** — shorter effective backward paths strengthen gradients to earlier layers (Huang et al., 2016)
3. **Free depth at eval** — the model is trained with an average effective depth lower than the eval depth, getting "deeper than it was trained" for free

**Why this is novel:** Stochastic depth (Huang et al., 2016) has been transformative in vision but has **never been applied to weight-shared recurrence in language models**. Our approach is distinct from standard layer dropout because shared-weight loops mean dropping an iteration doesn't reduce model capacity — it only reduces the number of refinement passes. This is closer in spirit to adaptive computation time (Graves, 2016) but requires no learned halting mechanism.

**Implementation:** DropPath with scale compensation on recurrence iterations only (non-recurrence layers are never dropped):
```
residual = x
x = block(x, x0)
if is_recurrence_iteration and training:
    keep = 1.0 - drop_prob
    mask = bernoulli(keep)
    x = residual + (x - residual) * mask / keep
```

**References:**
- Huang et al. (2016). "Deep Networks with Stochastic Depth." ECCV.
- Geiping et al. (2025). Depth recurrence for parameter-efficient LMs.
- AdaPonderLM (2026). Token-wise adaptive depth for recurrent LMs.

## Other Techniques

### 1. Gated Attention Output (AttnOutGate)
Learnable sigmoid gate on attention output, before projection. This technique appears in the best-performing open PRs (#1787 at 1.0635 BPB, #1769, #1784) but has never been in a merged record.

```
y = y * sigmoid(attn_gate)  # per-dim learned gate
```

### 2. Cosine Warmdown Schedule
Replace the linear warmdown with a cosine schedule for smoother LR decay. The cosine schedule may better preserve learning capacity during the final training phase.

### 3. SLOT (Sample-specific LM Optimization at Test-time)
Eval-time adaptation inspired by Hu et al. (arXiv:2505.12392). Optimizes a lightweight delta vector (R^512) at the final hidden layer per-batch using AdamW, 5 steps. Stacks on top of TTT for additional gain (-0.004 to -0.005 BPB). Zero artifact cost.

### 4. Polar Express NS (Improved Newton-Schulz Optimizer)
Hybrid Newton-Schulz iteration for Muon optimizer — standard NS steps followed by a final Schulz refinement step for better orthogonalization.

### 5. Pre-Quant TTT (Legal)
Adapt model on held-out training data before GPTQ quantization, reducing quant gap.

### 6. muP (Maximal Update Parameterization)
Width-scaled initialization (1/sqrt(fan_in)) and logit scaling (1/width_mult) for better hyperparameter transfer.

### 7. Hyperparameter Sweep
Systematic sweep of knobs not fully explored in existing records:
- QK-Gain: 5.25 (baseline), 5.5
- Parallel Residual Start: 5, 7 (baseline)
- Depth Recurrence Loops: 2 (baseline), 3
- Warmdown Type: linear (baseline), cosine

## Architecture (Baseline = Current SOTA)

11L x 512d x 8H/4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64), tied embeddings, logit softcap=30.0, depth recurrence (layers 3-5, 2 loops), parallel residuals (layer 7+), skip gates.

## Experiment Plan

### Phase 1: Screen (1 seed each)

| Exp | Change from baseline | Expected effect |
|-----|---------------------|-----------------|
| A0  | Baseline reproduction | Reference point |
| A1  | + Gated Attention | Improved feature selection |
| A2  | + Cosine warmdown | Smoother LR decay |
| A3  | + Polar Express NS (7 steps) | Better optimizer convergence |
| A4  | + Pre-Quant TTT | Reduced quantization gap |
| A5  | + muP parameterization | Better HP transfer |
| A6  | QK-Gain 5.5 | Sharper attention |
| A7  | Parallel residual start 5 | Earlier parallel computation |
| A8  | NUM_LOOPS=3 | Deeper effective model |
| A9  | Full combo (Gate+Cosine+Polar+PreQTTT) | Max stacking |
| A10 | + SLOT (5 steps, lr=0.003) | Eval-time hidden adaptation |
| A11 | Ultimate (A9 + SLOT) | Everything combined |

### Phase 2: Confirm (3 seeds for top configs)

Run top 2-3 configs with seeds 42, 314, 999.

## Results

| Exp | Config | val_bpb | delta_vs_baseline | notes |
|-----|--------|---------|-------------------|-------|
| A0  | baseline | TBD | - | |
| A1  | +gate | TBD | TBD | |
| A2  | +cosine | TBD | TBD | |
| A3  | +polar | TBD | TBD | |
| A4  | +preqttt | TBD | TBD | |
| A5  | +mup | TBD | TBD | |
| A6  | qk=5.5 | TBD | TBD | |
| A7  | par_start=5 | TBD | TBD | |
| A8  | loops=3 | TBD | TBD | |
| A9  | full_combo | TBD | TBD | |
| A10 | +slot | TBD | TBD | |
| A11 | ultimate | TBD | TBD | |

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

# Baseline
SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 ATTN_OUT_GATE=0 \
  torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-04-25_GatedAttn_SweepSOTA/train_gpt.py

# With Gated Attention
SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 ATTN_OUT_GATE=1 \
  torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-04-25_GatedAttn_SweepSOTA/train_gpt.py

# With Gated Attention + Cosine Warmdown + QK 5.5
SEED=42 QK_GAIN_INIT=5.5 TTT_ENABLED=1 ATTN_OUT_GATE=1 WARMDOWN_TYPE=cosine \
  torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-04-25_GatedAttn_SweepSOTA/train_gpt.py
```

## Credits

Building on the SOTA recipe from PR #1493:
- @clarkkev, @dexhunter, @abaybektursun, @Robby955, @msisovic, @X-Abhishek-X
