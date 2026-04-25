# Can Stochastic Depth Unlock Deeper Models Under Parameter Golf Constraints?

**Non-Record Submission (Research Contribution)**
**Author:** Phani Rata Nyalamanchili ([@phaniratanyalamanchili](https://github.com/phaniratanyalamanchili))
**Date:** 2026-04-25

---

## The Hypothesis

Every step in Parameter Golf has a fixed compute cost per layer ("step tax"). With 9 layers on 8xH100, each layer adds ~X ms/step. Under the 600-second wallclock cap, this directly limits how many gradient updates the model can take — the central tension of the competition.

**Stochastic depth** (Huang et al., 2016; also called LayerDrop, Fan et al., 2019) randomly skips layers during training with a linearly increasing probability from layer 0 (never skipped) to the deepest layer (skipped with probability `1 - survival_prob_last`). At inference, all layers run.

This creates two opportunities unique to Parameter Golf:
1. **Faster steps → more gradient updates** in the same wallclock budget
2. **"Free depth" exploration** — deeper, narrower models that would be too slow without stochastic depth become trainable within the time cap

No existing PR (out of 1500+) has explored stochastic depth. This submission tests whether the faster-step advantage outweighs the information loss from skipping layers.

---

## Implementation

### Core Changes to `train_gpt.py`

**1. Linear Survival Schedule in `GPT.forward()`**

Each layer `i` has survival probability:
```
p_survive(i) = 1.0 - (i / (num_layers - 1)) * drop_rate
```

Layer 0 always runs. The deepest layer is skipped most aggressively. This matches the original Deep Networks with Stochastic Depth paper's finding that later layers contribute less during early/middle training.

The skip logic preserves the U-Net encoder/decoder structure: when an encoder layer is skipped, we still record the current activation as the skip connection, so the decoder's reverse-order skip consumption stays aligned.

**2. Wallclock-Aware Drop Rate Annealing**

During warmdown (the final phase where learning rates decay to zero), we linearly anneal the drop rate to 0. This ensures the model trains with all layers active in its final phase, matching the inference-time computation graph. The schedule piggybacks on the existing `lr_mul()` wallclock-aware schedule:

```python
if stochastic_depth and sd_warmdown_aware:
    base_model.set_drop_rate(base_drop * lr_scale)
```

**3. Per-Layer Importance Analysis**

After training, we optionally run a single-layer ablation: drop each layer individually, measure BPB, and log the delta from baseline. This reveals which layers are critical vs. redundant — useful data for future pruning or architecture decisions.

### New Hyperparameters

| Variable | Default | Description |
|----------|---------|-------------|
| `STOCHASTIC_DEPTH` | 0 | Enable/disable |
| `SURVIVAL_PROB_LAST` | 0.7 | Survival probability of the deepest layer |
| `SD_WARMDOWN_AWARE` | 1 | Anneal drop rate to 0 during warmdown |
| `LAYER_IMPORTANCE_ANALYSIS` | 0 | Run per-layer ablation after training |

---

## Experiment Plan

### Axis 1: Stochastic Depth on Baseline (9L/512d)

Test whether stochastic depth improves the standard 9-layer baseline by enabling more training steps.

| Run | Config | Expected Effect |
|-----|--------|----------------|
| A (control) | 9L, SD=0 | Baseline |
| B | 9L, SD=1, surv=0.8 | Mild: ~10% step speedup |
| C | 9L, SD=1, surv=0.7 | Moderate: ~15% step speedup |
| D | 9L, SD=1, surv=0.5 | Aggressive: ~25% step speedup |

### Axis 2: Deep Narrow Models ("Free Depth")

Use stochastic depth to train models that would be too slow without it. Same parameter count, redistributed toward depth.

| Run | Layers | Width | MLP Mult | Approx Params | SD surv |
|-----|--------|-------|----------|---------------|---------|
| E | 15 | 384 | 2 | ~3.6M | 0.7 |
| F | 18 | 352 | 2 | ~3.5M | 0.6 |
| G | 12 | 448 | 2 | ~3.8M | 0.7 |

### Axis 3: Ablations

| Run | Variant | Tests |
|-----|---------|-------|
| H | No warmdown annealing | SD_WARMDOWN_AWARE=0 |
| I | Uniform drop (not linear) | Modified schedule |
| J | Layer importance analysis | LAYER_IMPORTANCE_ANALYSIS=1 |

---

## Results

*Results will be filled in after experiments on RunPod 8xH100.*

| Run | Config | val_bpb | Steps Completed | ms/step | Delta vs Baseline |
|-----|--------|---------|-----------------|---------|-------------------|
| A | 9L baseline | — | — | — | — |
| B | 9L SD surv=0.8 | — | — | — | — |
| C | 9L SD surv=0.7 | — | — | — | — |
| D | 9L SD surv=0.5 | — | — | — | — |
| E | 15L/384d SD surv=0.7 | — | — | — | — |
| F | 18L/352d SD surv=0.6 | — | — | — | — |
| G | 12L/448d SD surv=0.7 | — | — | — | — |

### Per-Layer Importance Analysis

*Will contain per-layer BPB deltas showing which layers are critical vs. redundant.*

---

## Reproducing These Results

```bash
# Baseline (no stochastic depth)
STOCHASTIC_DEPTH=0 NUM_LAYERS=9 MODEL_DIM=512 \
torchrun --nproc_per_node=8 train_gpt.py

# Stochastic depth, moderate drop rate
STOCHASTIC_DEPTH=1 SURVIVAL_PROB_LAST=0.7 SD_WARMDOWN_AWARE=1 \
NUM_LAYERS=9 MODEL_DIM=512 \
torchrun --nproc_per_node=8 train_gpt.py

# Deep narrow model
STOCHASTIC_DEPTH=1 SURVIVAL_PROB_LAST=0.7 SD_WARMDOWN_AWARE=1 \
NUM_LAYERS=15 MODEL_DIM=384 NUM_HEADS=6 NUM_KV_HEADS=3 \
torchrun --nproc_per_node=8 train_gpt.py

# With layer importance analysis
STOCHASTIC_DEPTH=1 SURVIVAL_PROB_LAST=0.7 LAYER_IMPORTANCE_ANALYSIS=1 \
NUM_LAYERS=9 MODEL_DIM=512 \
torchrun --nproc_per_node=8 train_gpt.py
```

---

## Why This Matters

Stochastic depth has been a standard regularization technique since 2016, yet nobody in 1500+ Parameter Golf PRs has tried it. The competition's unique wallclock constraint transforms it from a pure regularization technique into a **throughput optimization**: skipped layers = faster steps = more gradient updates.

Whether this trade-off is net positive is an empirical question this submission aims to answer. If it works, it opens a new axis of model design (depth-width trade-offs under stochastic depth). If it doesn't, documenting why is valuable: it would suggest that under these constraints, every layer's computation is too valuable to skip, even probabilistically.
