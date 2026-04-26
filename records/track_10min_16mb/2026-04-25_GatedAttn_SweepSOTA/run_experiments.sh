#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_gpt.py"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# LaCT-style chunk=2048, LoRA-TTT (Day 2+); TTT_EPOCHS capped to ≤2 inner steps in train_gpt
COMMON_ENV="TTT_ENABLED=1 TTT_LORA=1 TTT_CHUNK_TOKENS=2048 TTT_ADAPTIVE=1 TTT_LR=0.005 TTT_EPOCHS=2"

run_experiment() {
    local name="$1"
    local seed="$2"
    shift 2
    local extra_env="$@"

    echo "=========================================="
    echo "Running: $name (seed=$seed)"
    echo "Extra env: $extra_env"
    echo "=========================================="

    local logfile="$LOG_DIR/${name}_seed${seed}.log"

    env $COMMON_ENV $extra_env SEED=$seed \
        torchrun --standalone --nproc_per_node=8 "$TRAIN_SCRIPT" \
        2>&1 | tee "$logfile"

    echo "Done: $name (seed=$seed) -> $logfile"
    echo ""
}

echo "============================================"
echo "PHASE 1: Screening (1 seed each, ~12 min per run)"
echo "============================================"

# A0: Baseline (LayerScale init 1e-4, stoch off by default) — Day 1 stack
run_experiment "A0_baseline" 42 \
    "QK_GAIN_INIT=5.25 ATTN_OUT_GATE=0 POLAR_EXPRESS=0 PRE_QUANT_TTT=0 MUP_ENABLED=0 LAYER_SCALE_INIT=1e-4 STOCHASTIC_RECURRENCE=0"

# A1: + Gated Attention
run_experiment "A1_gated_attn" 42 \
    "QK_GAIN_INIT=5.25 ATTN_OUT_GATE=1"

# A2: + Cosine Warmdown
run_experiment "A2_cosine_warmdown" 42 \
    "QK_GAIN_INIT=5.25 ATTN_OUT_GATE=0 WARMDOWN_TYPE=cosine"

# A3: + Polar Express NS (7 steps)
run_experiment "A3_polar_express" 42 \
    "QK_GAIN_INIT=5.25 ATTN_OUT_GATE=0 POLAR_EXPRESS=1 POLAR_EXPRESS_STEPS=7"

# A4: + Pre-Quant TTT (adapt before GPTQ)
run_experiment "A4_pre_quant_ttt" 42 \
    "QK_GAIN_INIT=5.25 ATTN_OUT_GATE=0 PRE_QUANT_TTT=1 PRE_QUANT_TTT_LR=0.003"

# A5: + muP (width-scaled parameterization)
run_experiment "A5_mup" 42 \
    "QK_GAIN_INIT=5.25 ATTN_OUT_GATE=0 MUP_ENABLED=1 MUP_WIDTH_BASE=256"

# A6: QK-Gain 5.5
run_experiment "A6_qk55" 42 \
    "QK_GAIN_INIT=5.5 ATTN_OUT_GATE=0"

# A7: Parallel Residual Start 5
run_experiment "A7_par_start5" 42 \
    "QK_GAIN_INIT=5.25 ATTN_OUT_GATE=0 PARALLEL_RESIDUAL_START=5"

# A8: 3 Loops (deeper recurrence)
run_experiment "A8_3loops" 42 \
    "QK_GAIN_INIT=5.25 ATTN_OUT_GATE=0 NUM_LOOPS=3"

# A9: BEST COMBO — all winning techniques combined
run_experiment "A9_full_combo" 42 \
    "QK_GAIN_INIT=5.5 ATTN_OUT_GATE=1 WARMDOWN_TYPE=cosine POLAR_EXPRESS=1 PRE_QUANT_TTT=1"

# A10: + SWA (Stochastic Weight Averaging, proven -0.005 to -0.01 BPB)
run_experiment "A10_swa" 42 \
    "QK_GAIN_INIT=5.25 ATTN_OUT_GATE=0 SWA_ENABLED=1 SWA_START_FRAC=0.8 SWA_EVERY=50"

# A11: + Late QAT (STE int6 fake-quant during warmdown, reduces quant gap)
run_experiment "A11_late_qat" 42 \
    "QK_GAIN_INIT=5.25 ATTN_OUT_GATE=0 LATE_QAT_ENABLED=1 LATE_QAT_START_FRAC=0.9"

# A12: + EMA Decay Annealing (novel — ramp from 0.99 to 0.999)
run_experiment "A12_ema_anneal" 42 \
    "QK_GAIN_INIT=5.25 ATTN_OUT_GATE=0 EMA_ANNEAL=1 EMA_DECAY_START=0.99 EMA_DECAY_END=0.999"

# A13: (removed) SLOT is dropped — use TTT_LORA+adaptive in train_gpt instead

# A14: + Stochastic Recurrence (NOVEL — DropPath on recurrence iterations)
run_experiment "A14_stoch_recur" 42 \
    "QK_GAIN_INIT=5.25 ATTN_OUT_GATE=0 STOCHASTIC_RECURRENCE=1 STOCH_DROP_PROB=0.2"

# A15: ULTIMATE COMBO (no SLOT / no stoch drop on recurrence)
run_experiment "A15_ultimate" 42 \
    "QK_GAIN_INIT=5.5 ATTN_OUT_GATE=1 WARMDOWN_TYPE=cosine POLAR_EXPRESS=1 PRE_QUANT_TTT=1 SWA_ENABLED=1 SWA_EVERY=50 LATE_QAT_ENABLED=1 EMA_ANNEAL=1 LAYER_SCALE_INIT=1e-4 TTT_LORA=1 TTT_ADAPTIVE=1"

# A16: Plan stack — IPTT + SmearGate + pre-GPTQ LQER + strong train (matches A9/A11/A15-style wins; TTT_LORA=0)
# Polar + pre-quant TTT (train) + late QAT; SWA+EMA anneal (A15) — TIT_MODE=iptt for val TTT
run_experiment "A16_iptt_smear_lqer" 42 \
    "TIT_MODE=iptt TTT_LORA=0 QK_GAIN_INIT=5.5 ATTN_OUT_GATE=1 SMEAR_GATE=1 PRE_GPTQ_LQER=1 LQER_RANK=4 WARMDOWN_TYPE=cosine LAYER_SCALE_INIT=1e-4 POLAR_EXPRESS=1 POLAR_EXPRESS_STEPS=7 PRE_QUANT_TTT=1 PRE_QUANT_TTT_LR=0.003 LATE_QAT_ENABLED=1 LATE_QAT_START_FRAC=0.9 SWA_ENABLED=1 SWA_START_FRAC=0.8 SWA_EVERY=50 EMA_ANNEAL=1 EMA_DECAY_START=0.99 EMA_DECAY_END=0.999"

echo "============================================"
echo "PHASE 1 COMPLETE — 16 experiments (A16 = IPTT + Smear + LQER)"
echo "Review logs in $LOG_DIR"
echo "grep for 'val_bpb' to compare results:"
echo "  grep 'val_bpb' $LOG_DIR/*.log"
echo "============================================"
