#!/usr/bin/env python3
"""
Offline checks for Parameter Golf compressed artifacts (final_model.int6.ptz).

1) Bit packing: GPTQ stores each quantized weight as one int8 (values in
   [-clip_range, clip_range], e.g. 31 for 6-bit), not 6 physical bits per weight.
   “Packing” 4×int6 into 3 bytes is NOT implemented — using int8 per element is
   common and matches train_gpt.serialize().

2) Brotli: payload is torch.save({w, m}) after byte-shuffle (see train_gpt._byte_shuffle).

3) Entropy diagnostic: for each *.q tensor, report empirical H (bits/symbol) of
   int8 codes; compare to log2(|unique|) and 8.0. If H ≈ 8, extra entropy coding
   barely helps; if H << 8, rANS/tANS may save bytes.

Usage:
  python pg_artifact_audit.py path/to/final_model.int6.ptz
  python pg_artifact_audit.py path/to/final_model.int6.ptz --verbose
"""
from __future__ import annotations

import argparse
import io
import math
import sys
import zlib
from pathlib import Path

import numpy as np

_BSHF_MAGIC = b'BSHF'


def _byte_unshuffle(data: bytes) -> bytes:
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]
    if stride < 2:
        return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload)
    out = np.empty(n, dtype=np.uint8)
    src_off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off:src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()


def decompress_blob(raw: bytes, compressor: str) -> bytes:
    if compressor == 'lzma':
        import lzma
        return lzma.decompress(raw)
    if compressor == 'brotli':
        import brotli
        raw_d = brotli.decompress(raw)
    elif compressor == 'zlib':
        raw_d = zlib.decompress(raw)
    else:
        raise ValueError(f'Unknown compressor {compressor!r}')
    return _byte_unshuffle(raw_d)


def entropy_int8(arr: np.ndarray) -> tuple[float, int, float]:
    """Return (H in bits/symbol, n_unique, log2(n_unique))."""
    arr = arr.astype(np.int64).ravel()
    if arr.size == 0:
        return 0.0, 0, 0.0
    uniq, counts = np.unique(arr, return_counts=True)
    p = counts.astype(np.float64) / counts.sum()
    h = float(-np.sum(p * np.log2(p + 1e-30)))
    return h, int(uniq.size), math.log2(max(uniq.size, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description='PG artifact entropy / format audit')
    ap.add_argument('ptz', type=Path, help='Path to final_model.int6.ptz')
    ap.add_argument(
        '--compressor', default='brotli',
        choices=('brotli', 'lzma', 'zlib'),
        help='Must match COMPRESSOR used in training (default brotli)',
    )
    ap.add_argument('--verbose', action='store_true', help='Print per-tensor lines')
    args = ap.parse_args()
    try:
        import torch
    except ImportError as e:
        print('torch is required for loading the blob', file=sys.stderr)
        raise SystemExit(1) from e
    raw = args.ptz.read_bytes()
    print(f'File: {args.ptz}  bytes_on_disk={len(raw)}')
    try:
        inner = decompress_blob(raw, args.compressor)
    except Exception as e:
        print(
            f'Decompress failed ({e}). Try --compressor lzma if training used COMPRESSOR=lzma.',
            file=sys.stderr,
        )
        raise SystemExit(2) from e
    state = torch.load(io.BytesIO(inner), map_location='cpu', weights_only=False)
    if not isinstance(state, dict) or 'w' not in state:
        print('Unexpected payload: expected dict with key "w"', file=sys.stderr)
        raise SystemExit(3)
    w = state['w']
    q_keys = sorted(k for k in w if k.endswith('.q'))
    if not q_keys:
        print('No *.q tensors found in w', file=sys.stderr)
        raise SystemExit(4)
    hs, us = [], []
    for k in q_keys:
        t = w[k].numpy()
        h, nu, lu = entropy_int8(t)
        hs.append(h)
        us.append(nu)
        if args.verbose:
            print(f'  {k:60s} H={h:.4f} bits/sym  |unique|={nu}  log2(|U|)={lu:.4f}  n={t.size}')
    print(
        f'Summary: {len(q_keys)} int tensors, mean H={float(np.mean(hs)):.4f} bits/sym, '
        f'min H={float(np.min(hs)):.4f}, max H={float(np.max(hs)):.4f}'
    )
    print(
        f'Headroom vs 8 bits/sym: {8.0 - float(np.mean(hs)):.4f} (rough upper bound; '
        'Brotli is not a bitwise entropy coder for int codes.)'
    )


if __name__ == '__main__':
    main()
