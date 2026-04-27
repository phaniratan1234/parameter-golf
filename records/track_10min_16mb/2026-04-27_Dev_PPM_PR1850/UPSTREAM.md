# Dev track: PPM mixture stack (openai/parameter-golf PR #1850)

This directory is a **verbatim snapshot** of the submission files from an **unmerged** upstream PR. Use it to reproduce the PPM eval path, then iterate (e.g. merge ideas from `2026-04-25_GatedAttn_SweepSOTA/`).

## Source

| Field | Value |
|--------|--------|
| Upstream PR | https://github.com/openai/parameter-golf/pull/1850 |
| Fork / branch | `someone114514/parameter-golf` → `sp8192-strict-fullval-ppm-0426` |
| Snapshot commit | `37ce906f59f0fe9af3ed9752c97f6c920626cec2` |
| Claimed headline | ~**1.005** BPB (3-seed) with strict full-val byte PPM mixture at eval |

## Contents

- `pr1850_upstream_snapshot/` — `train_gpt.py`, `README.md`, `submission.json`, and `train_seed*.log` as in the PR.

## Refresh snapshot from git (this repo)

```bash
git fetch https://github.com/someone114514/parameter-golf.git \
  sp8192-strict-fullval-ppm-0426:refs/remotes/ppm-pr1850
git archive ppm-pr1850 records/track_10min_16mb/2026-04-26_SP8192_StrictFullValPPM \
  | tar -x -C records/track_10min_16mb/2026-04-27_Dev_PPM_PR1850/
# then move/rename into pr1850_upstream_snapshot/ if you prefer a flat layout
```

## Run (see upstream README in snapshot)

From repo root, after data setup as in Parameter Golf docs, use the env block in PR #1850 / `pr1850_upstream_snapshot/README.md`.

## Next steps for your experiments

1. Copy `pr1850_upstream_snapshot/train_gpt.py` to a **new dated folder** (e.g. `2026-04-27_PPM_plus_YourStack/`) and apply patches there so this snapshot stays a clean reference.
2. Compare against your GatedAttn / quant work with `diff`—PPM logic is eval-time; AWQ/mixed-bit/GPTQ are training/serialize-time; they can be composed only if you port carefully and stay under byte + time rules.

**Note:** PR #1850 is **not** merged; scores are **not** official leaderboard until organizers accept. See also open PR **#1835** (~1.001 BPB, related PPM mixture idea).
