# ==============================================================================
# Copyright (C) 2024-2026 Shipei Xing <sxing@fudan.edu.cn>
#
# Licensed under the Apache License 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://github.com/Philipbear/msbuddy/blob/main/LICENSE
# ==============================================================================
"""msbuddy 2.0 unified CPU ONNX + LightGBM inference engine (format_version = 4).

Pipeline:
  Stage 0: Exact-Mass Sparse Peak-Set Transformer (ONNX, 60-dim universal query_meta)
  Stage 1: Compiled Numba precursor + MS2 stitching + Hybrid-Protected Encoder-Guided Pruning
  Stage 2: Compiled Numba subformula assignment + analytical MS1 isotope engine (136-dim features)
  Stage 3: Unified Hybrid Reranker (Cross-Candidate PureSetReranker ONNX + LightGBM LambdaRank)
  Stage 4: Out-of-source calibrated posterior probabilities & polarity-aware adduct="auto" pooling
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import gammaln, logsumexp

from msbuddy._fast_db import (
    ELEM_MASSES_F64,
    MASS_E,
    MS1_FEAT_DIM,
    QUERY_META_60_DIM,
    build_query_meta_60,
    compute_ms1_features_numba,
    compute_ms2_evidence_fast,
    compute_ms2_evidence_fast_v3,
    generate_candidates_fast_numba,
    get_fast_store,
)
from msbuddy.base import Adduct, CandidateFormula, Formula, MS2Explanation

PEAK_WAVELENGTHS = (500.0, 100.0, 20.0, 4.0, 1.0, 0.2, 0.02)
PEAK_FEAT_DIM = 40
CAND_FEAT_DIM = 136
POOL_FEAT_DIM = 10
CAND_V5_DIM = 179
POOL_V5_DIM = 14
CL_DIM = 128
SPEC_CTX_DIM = 34 + 3 + QUERY_META_60_DIM  # 97-d canonical context [ep, pr, query_meta]
DEFAULT_K_MAX = 80
DEFAULT_PROTECT = 24
DEFAULT_LL_W = 0.35
DEFAULT_CL_W = 0.25

_ELEM_MASSES_V3_F32 = np.array([
    12.0, 1.00782503223, 78.9183376, 34.968852682,
    18.99840316273, 126.9044719, 38.9637064864, 14.00307400443,
    22.9897692820, 15.99491461957, 30.97376199842, 31.9720711744,
], dtype=np.float32)

AUTO_ADDUCTS_POS = (
    '[M+H]+', '[M+Na]+', '[M+NH4]+', '[M+K]+',
    '[M-H2O+H]+', '[2M+H]+', '[2M+Na]+',
)
AUTO_ADDUCTS_NEG = (
    '[M-H]-', '[M+Cl]-', '[M+CH2O2-H]-', '[M+C2H4O2-H]-', '[2M-H]-',
)
# Bayesian adduct prevalence & baseline-offset log-priors applied only during
# opt-in adduct="auto" disambiguation (calibrated to format_version=5 log-odds
# scale to suppress rare/isobaric and half-mass dimer false positives while
# leaving known-adduct scoring untouched).
AUTO_ADDUCT_LOG_PRIORS = {
    '[M+H]+': 0.0,
    '[M+Na]+': -2.5,
    '[M+NH4]+': -5.5,
    '[M-H2O+H]+': -6.0,
    '[M+K]+': -6.0,
    '[2M+H]+': -3.5,
    '[2M+Na]+': -4.2,
    '[M-H]-': 0.0,
    '[M+CH2O2-H]-': -5.5,
    '[M+C2H4O2-H]-': -6.5,
    '[M+Cl]-': -3.2,
    '[2M-H]-': -2.5,
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_np_dict(d: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(d.keys()):
        arr = np.ascontiguousarray(d[k], dtype=np.float32)
        h.update(k.encode('utf-8'))
        h.update(str(arr.shape).encode('utf-8'))
        h.update(arr.tobytes())
    return h.hexdigest()


def build_formula_features_v3(counts: np.ndarray, adduct_meta: np.ndarray) -> np.ndarray:
    """Build 32-d formula+adduct descriptors matching FormulaEncoderV3 training bit-for-bit."""
    c = np.maximum(counts.astype(np.float32), 0.0)
    log_c = np.log1p(c) / 5.0
    c_denom = c[..., 0:1] + 1.0
    hetero_ratios = np.clip(c[..., 2:12] / c_denom, 0.0, 5.0)
    halos = c[..., 2] + c[..., 3] + c[..., 4] + c[..., 5]
    dbe = c[..., 0] - 0.5 * (c[..., 1] + halos) + 0.5 * (c[..., 7] + c[..., 10]) + 1.0
    hc = np.clip(c[..., 1] / c_denom[..., 0], 0.0, 4.0)
    hetero_sum = c[..., 2:12].sum(axis=-1)
    hetero_r = hetero_sum / (c[..., 0] + hetero_sum + 1.0)
    mass = (c * _ELEM_MASSES_V3_F32).sum(axis=-1)
    log_m = np.log1p(mass / 500.0)
    mdef = mass - np.rint(mass)
    glob = np.stack([dbe / 20.0, hc / 4.0, hetero_r, log_m, mdef], axis=-1)
    return np.concatenate([log_c, hetero_ratios, glob, adduct_meta.astype(np.float32)], axis=-1).astype(np.float32)


def extract_adduct_meta5(qm: np.ndarray) -> np.ndarray:
    """Extract [charge, m/2, has_Na, has_K, has_N] from 60-d query_meta."""
    charge = qm[..., 15:16]
    m_half = qm[..., 16:17]
    has_na = (qm[..., 3 + 8:3 + 9] > 0.5).astype(np.float32)
    has_k = (qm[..., 3 + 6:3 + 7] > 0.5).astype(np.float32)
    has_n = (qm[..., 3 + 7:3 + 8] > 0.5).astype(np.float32)
    return np.concatenate([charge, m_half, has_na, has_k, has_n], axis=-1).astype(np.float32)


def _np_gelu(x: np.ndarray) -> np.ndarray:
    # Fast float32 tanh-GELU approximation (max diff < 2e-4 vs exact erf, ~4x faster & nogil-friendly)
    return (0.5 * x * (1.0 + np.tanh(np.float32(0.7978845608) * (x + np.float32(0.044715) * x * x * x)))).astype(np.float32)


def _np_layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    return (xc / np.sqrt(var + np.float32(eps)) * w + b).astype(np.float32)


def _prep_fenc_np(p: dict) -> dict:
    """Pre-transpose & make C-contiguous weight matrices once at runtime load for fast BLAS GEMM."""
    out = {}
    for k, v in p.items():
        arr = np.ascontiguousarray(v, dtype=np.float32)
        out[k] = arr
        if k.endswith('.weight') and arr.ndim == 2:
            out[k + '.T'] = np.ascontiguousarray(arr.T, dtype=np.float32)
    return out


def formula_enc_forward_np(x: np.ndarray, p: dict) -> np.ndarray:
    """Pure-NumPy forward pass of FormulaEncoderV3 (~1 ms/spec after M_pre pruning)."""
    w0 = p.get('in_proj.0.weight.T')
    w0 = p['in_proj.0.weight'].T if w0 is None else w0
    h = x @ w0 + p['in_proj.0.bias']
    h = _np_gelu(_np_layernorm(h, p['in_proj.1.weight'], p['in_proj.1.bias']))

    w10 = p.get('res1.0.weight.T')
    w10 = p['res1.0.weight'].T if w10 is None else w10
    w14 = p.get('res1.4.weight.T')
    w14 = p['res1.4.weight'].T if w14 is None else w14
    r1 = h @ w10 + p['res1.0.bias']
    r1 = _np_gelu(_np_layernorm(r1, p['res1.1.weight'], p['res1.1.bias']))
    h = h + (r1 @ w14 + p['res1.4.bias'])

    w20 = p.get('res2.0.weight.T')
    w20 = p['res2.0.weight'].T if w20 is None else w20
    w24 = p.get('res2.4.weight.T')
    w24 = p['res2.4.weight'].T if w24 is None else w24
    r2 = h @ w20 + p['res2.0.bias']
    r2 = _np_gelu(_np_layernorm(r2, p['res2.1.weight'], p['res2.1.bias']))
    h = h + (r2 @ w24 + p['res2.4.bias'])

    h = _np_layernorm(h, p['out_norm.weight'], p['out_norm.bias'])
    wo = p.get('out_proj.weight.T')
    wo = p['out_proj.weight'].T if wo is None else wo
    z = h @ wo + p['out_proj.bias']
    z /= np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), np.float32(1e-12))
    return z.astype(np.float32)


def load_v2_bundle_dir(bundle_dir: Path) -> dict:
    """Load an unpacked msbuddy 2.0 bundle directory into an in-memory format_version=4 dict."""
    bundle_dir = Path(bundle_dir)
    meta_path = bundle_dir / 'bundle_meta.json'
    if not meta_path.is_file():
        raise FileNotFoundError(f'Missing bundle_meta.json in {bundle_dir}')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    s0_bytes = (bundle_dir / 'stage0_encoder.onnx').read_bytes()
    s3_bytes = (bundle_dir / 'stage3_set_reranker.onnx').read_bytes()
    lgb_text = (bundle_dir / 'stage3_lgb.txt').read_text(encoding='utf-8')
    bundle = {
        'format_version': 4,
        'version': meta.get('variant', '2.0.0'),
        'variant': meta.get('variant', 'public'),
        'cand_feat_dim': int(meta['cand_feat_dim']),
        'pool_feat_dim': int(meta['pool_feat_dim']),
        'query_meta_dim': int(meta['query_meta_dim']),
        'feat_center': np.asarray(meta['feat_center'], dtype=np.float32),
        'feat_scale': np.asarray(meta['feat_scale'], dtype=np.float32),
        'fusion': dict(meta['fusion']),
        'calibration': dict(meta['calibration']),
        'stage0_encoder_onnx': s0_bytes,
        'stage0_encoder_sha256': meta['files']['stage0_encoder.onnx']['sha256'],
        'stage3_reranker_onnx': s3_bytes,
        'stage3_reranker_sha256': meta['files']['stage3_set_reranker.onnx']['sha256'],
        'stage3_lgb_txt': lgb_text,
        'stage3_lgb_sha256': meta['files']['stage3_lgb.txt']['sha256'],
    }
    validate_v2_bundle(bundle)
    return bundle


def pack_v2_bundle_joblib(bundle_dir: Path, out_joblib: Path, version: str = '2.0.0') -> dict:
    """Pack an exported msbuddy 2.0 bundle directory into a single portable joblib file."""
    from joblib import dump as j_dump
    bundle = load_v2_bundle_dir(bundle_dir)
    bundle['version'] = version
    out_joblib = Path(out_joblib)
    out_joblib.parent.mkdir(parents=True, exist_ok=True)
    j_dump(bundle, out_joblib, compress=3)
    return {
        'path': str(out_joblib),
        'bytes': out_joblib.stat().st_size,
        'sha256': _sha256_bytes(out_joblib.read_bytes()),
    }


def validate_v2_bundle(bundle: dict) -> None:
    """Verify schema, finite parameters, and SHA-256 checksums for format_version 4 or 5 bundles."""
    if not isinstance(bundle, dict) or bundle.get('format_version') not in (4, 5):
        raise ValueError('msbuddy 2.0 bundle must have format_version == 4 or 5.')
    fv = int(bundle['format_version'])
    exp_cdim = CAND_FEAT_DIM if fv == 4 else CAND_V5_DIM
    exp_pdim = POOL_FEAT_DIM if fv == 4 else POOL_V5_DIM
    if bundle.get('cand_feat_dim') != exp_cdim or bundle.get('pool_feat_dim') != exp_pdim:
        raise ValueError('msbuddy 2.0 bundle feature dimensions do not match engine schema.')
    if bundle.get('query_meta_dim') != QUERY_META_60_DIM:
        raise ValueError('msbuddy 2.0 bundle query_meta_dim must equal 60.')

    for data_key, hash_key in (
        ('stage0_encoder_onnx', 'stage0_encoder_sha256'),
        ('stage3_reranker_onnx', 'stage3_reranker_sha256'),
    ):
        raw = bundle.get(data_key)
        expected = bundle.get(hash_key)
        if not isinstance(raw, bytes) or not isinstance(expected, str) or _sha256_bytes(raw) != expected:
            raise ValueError(f'msbuddy 2.0 checksum verification failed for {data_key}.')

    lgb_txt = bundle.get('stage3_lgb_txt')
    lgb_sha = bundle.get('stage3_lgb_sha256')
    if not isinstance(lgb_txt, str) or not isinstance(lgb_sha, str) or _sha256_bytes(lgb_txt.encode('utf-8')) != lgb_sha:
        raise ValueError('msbuddy 2.0 checksum verification failed for stage3_lgb_txt.')

    if fv == 5:
        if bundle.get('cl_dim') != CL_DIM or bundle.get('spec_ctx_dim') != SPEC_CTX_DIM:
            raise ValueError('Invalid msbuddy 2.0 v5 contrastive / context dimensions.')
        fenc_np = bundle.get('formula_enc_np')
        fenc_sha = bundle.get('formula_enc_sha256')
        if not isinstance(fenc_np, dict) or not isinstance(fenc_sha, str) or _sha256_np_dict(fenc_np) != fenc_sha:
            raise ValueError('msbuddy 2.0 v5 checksum verification failed for formula_enc_np.')
        ls = float(bundle.get('logit_scale', -1.0))
        if not math.isfinite(ls) or ls <= 0.0:
            raise ValueError('Invalid msbuddy 2.0 v5 logit_scale.')

    center = np.asarray(bundle.get('feat_center'), dtype=np.float32)
    scale = np.asarray(bundle.get('feat_scale'), dtype=np.float32)
    if center.shape != (exp_cdim,) or scale.shape != (exp_cdim,):
        raise ValueError('Invalid msbuddy 2.0 feature normalization arrays.')
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError('Non-finite or non-positive msbuddy 2.0 normalization parameters.')

    fusion = bundle.get('fusion', {})
    w_nn = float(fusion.get('w_nn', -1.0))
    lgb_scale = float(fusion.get('lgb_scale', -1.0))
    if not math.isfinite(w_nn) or not 0.0 <= w_nn <= 1.0 or not math.isfinite(lgb_scale) or lgb_scale <= 0.0:
        raise ValueError('Invalid msbuddy 2.0 fusion parameters.')

    cal = bundle.get('calibration', {})
    temp = float(cal.get('temp', -1.0))
    b_null = float(cal.get('b_null', float('nan')))
    if not math.isfinite(temp) or temp <= 0.0 or not math.isfinite(b_null):
        raise ValueError('Invalid msbuddy 2.0 calibration parameters.')


def _get_v2_runtime(gd: dict) -> dict:
    """Lazily create CPU ONNX Runtime sessions and LightGBM booster."""
    cache = gd.setdefault('model_v2_cache', {})
    if 'stage0' not in cache:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                'msbuddy 2.0 requires onnxruntime. Install package dependencies '
                'or pass MsbuddyConfig(use_spectral=False).'
            ) from exc
        from lightgbm import Booster

        spec = gd['model_v2']
        fv = int(spec.get('format_version', 4))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        cache['format_version'] = fv
        cache['cand_feat_dim'] = CAND_FEAT_DIM if fv == 4 else CAND_V5_DIM
        cache['pool_feat_dim'] = POOL_FEAT_DIM if fv == 4 else POOL_V5_DIM
        cache['stage0'] = ort.InferenceSession(
            spec['stage0_encoder_onnx'], sess_options=opts, providers=['CPUExecutionProvider']
        )
        cache['stage3_nn'] = ort.InferenceSession(
            spec['stage3_reranker_onnx'], sess_options=opts, providers=['CPUExecutionProvider']
        )
        cache['stage3_lgb'] = Booster(model_str=spec['stage3_lgb_txt'])
        cache['feat_center'] = np.asarray(spec['feat_center'], dtype=np.float32)
        cache['feat_scale'] = np.asarray(spec['feat_scale'], dtype=np.float32)
        cache['w_nn'] = float(spec['fusion']['w_nn'])
        cache['lgb_scale'] = float(spec['fusion']['lgb_scale'])
        cache['temp'] = float(spec['calibration']['temp'])
        cache['b_null'] = float(spec['calibration']['b_null'])
        if fv == 5:
            cache['formula_enc_np'] = _prep_fenc_np(spec['formula_enc_np'])
            cache['logit_scale'] = float(spec['logit_scale'])
            sl = spec.get('shortlist', {})
            cache['ll_w'] = float(sl.get('ll_w', DEFAULT_LL_W))
            cache['cl_w'] = float(sl.get('cl_w', DEFAULT_CL_W))
        else:
            cache['formula_enc_np'] = None
            cache['logit_scale'] = 1.0
            cache['ll_w'] = DEFAULT_LL_W
            cache['cl_w'] = 0.0
    return cache


def featurize_peaks_v2(
    mzs: np.ndarray,
    ints: np.ndarray,
    precursor_mz: float,
    ms2_tol_ppm: float = 20.0,
    max_peaks: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Exact NumPy peak featurization matching Stage-0 training (40 dims per peak)."""
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)
    keep = np.isfinite(mzs) & np.isfinite(ints) & (mzs > 0.0) & (ints > 0.0) & (mzs < precursor_mz - 0.5)
    mzs, ints = mzs[keep], ints[keep]

    peak_feats = np.zeros((max_peaks, PEAK_FEAT_DIM), dtype=np.float32)
    peak_mzs = np.zeros((max_peaks,), dtype=np.float32)
    peak_ints = np.zeros((max_peaks,), dtype=np.float32)
    peak_mask = np.zeros((max_peaks,), dtype=bool)
    n_valid = 0

    if len(mzs) > 0:
        imax = float(ints.max())
        if imax > 0.0:
            keep2 = ints >= imax * 0.001
            mzs, ints = mzs[keep2], ints[keep2]
            if len(mzs) > max_peaks:
                top_idx = np.argsort(-ints, kind='stable')[:max_peaks]
                mzs, ints = mzs[top_idx], ints[top_idx]
            order = np.argsort(mzs, kind='stable')
            mzs, ints = mzs[order], ints[order]
            n_valid = len(mzs)

            irel = ints / max(float(ints.max()), 1e-12)
            isum = ints / max(float(ints.sum()), 1e-12)
            ranks = np.argsort(np.argsort(-ints, kind='stable')).astype(np.float64) / max(n_valid, 1)
            losses = np.maximum(precursor_mz - mzs, 0.0)

            col = 0
            peak_feats[:n_valid, col] = (mzs / 1000.0).astype(np.float32); col += 1
            peak_feats[:n_valid, col] = (losses / 1000.0).astype(np.float32); col += 1
            peak_feats[:n_valid, col] = (mzs / max(precursor_mz, 1.0)).astype(np.float32); col += 1
            peak_feats[:n_valid, col] = (mzs - np.rint(mzs)).astype(np.float32); col += 1
            peak_feats[:n_valid, col] = (losses - np.rint(losses)).astype(np.float32); col += 1

            two_pi = 2.0 * math.pi
            for wl in PEAK_WAVELENGTHS:
                peak_feats[:n_valid, col] = np.sin((two_pi / wl) * mzs).astype(np.float32); col += 1
            for wl in PEAK_WAVELENGTHS:
                peak_feats[:n_valid, col] = np.cos((two_pi / wl) * mzs).astype(np.float32); col += 1
            for wl in PEAK_WAVELENGTHS:
                peak_feats[:n_valid, col] = np.sin((two_pi / wl) * losses).astype(np.float32); col += 1
            for wl in PEAK_WAVELENGTHS:
                peak_feats[:n_valid, col] = np.cos((two_pi / wl) * losses).astype(np.float32); col += 1

            peak_feats[:n_valid, col] = np.sqrt(irel).astype(np.float32); col += 1
            peak_feats[:n_valid, col] = (np.log1p(irel * 100.0) / math.log(101.0)).astype(np.float32); col += 1
            peak_feats[:n_valid, col] = isum.astype(np.float32); col += 1
            peak_feats[:n_valid, col] = ranks.astype(np.float32); col += 1

            tols = np.maximum(0.003, mzs * ms2_tol_ppm * 1e-6)
            for shift in (1.0033548, 1.9970499, 1.9979535):
                targets = mzs + shift
                idx_r = np.searchsorted(mzs, targets)
                ratios = np.zeros(n_valid, dtype=np.float64)
                for k in range(n_valid):
                    best = 0.0
                    for cand_i in (idx_r[k] - 1, idx_r[k]):
                        if 0 <= cand_i < n_valid and abs(mzs[cand_i] - targets[k]) <= tols[k]:
                            r = ints[cand_i] / max(ints[k], 1e-12)
                            if r > best:
                                best = r
                    ratios[k] = math.log1p(min(best, 2.0))
                peak_feats[:n_valid, col] = ratios.astype(np.float32); col += 1

            peak_mzs[:n_valid] = mzs.astype(np.float32)
            peak_ints[:n_valid] = irel.astype(np.float32)
            peak_mask[:n_valid] = True

    return peak_feats, peak_mzs, peak_ints, peak_mask, mzs, n_valid


def element_ll_numpy(elem_params: np.ndarray, counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized per-element log-likelihood & expected-count residuals for (C, 12) formulas."""
    c = counts.astype(np.float64)
    mu_ch = elem_params[0:2].astype(np.float64)
    log_sig = np.clip(elem_params[2:4].astype(np.float64), -2.0, 2.0)
    sig = np.exp(log_sig)
    y_ch = np.log1p(c[:, 0:2])
    res_ch = (y_ch - mu_ch) / sig
    ll_ch = -0.5 * (res_ch ** 2) - log_sig - 0.5 * math.log(2.0 * math.pi)

    h_raw = elem_params[4:34].astype(np.float64).reshape(10, 3)
    logit_p = h_raw[:, 0]
    mean = np.logaddexp(0.0, h_raw[:, 1]) + 1e-4
    shape = np.logaddexp(0.0, h_raw[:, 2]) + 1e-4

    p_pres = 1.0 / (1.0 + np.exp(-np.clip(logit_p, -30.0, 30.0)))
    log_p = -np.logaddexp(0.0, -logit_p)
    log_1mp = -np.logaddexp(0.0, logit_p)

    c_h = c[:, 2:12]
    log_zero_nb = shape * (np.log(shape) - np.log(shape + mean))
    log_nb = (
        gammaln(c_h + shape) - gammaln(shape) - gammaln(c_h + 1.0)
        + log_zero_nb
        + c_h * (np.log(mean) - np.log(shape + mean))
    )
    log_one_minus_zero = np.log(np.maximum(-np.expm1(log_zero_nb), 1e-7))
    ll_pos = log_p + log_nb - log_one_minus_zero
    ll_h = np.where(c_h > 0.5, ll_pos, log_1mp)

    e_pos = mean / np.maximum(-np.expm1(log_zero_nb), 1e-7)
    exp_cnt = p_pres * e_pos
    res_h = (c_h - exp_cnt) / (1.0 + exp_cnt)

    return (
        np.concatenate([ll_ch, ll_h], axis=1).astype(np.float32),
        np.concatenate([res_ch, res_h], axis=1).astype(np.float32),
    )


def _stage1a_gen_and_prefilter(
    mf,
    adduct: Adduct,
    ep: np.ndarray,
    config,
    store,
    k_max: int,
    n_protect: int,
    ll_w: float,
    qm: np.ndarray,
    m_pre: int,
):
    """v5 Stage-1A (nogil Numba): candidate generation, MS1 scoring, and top-m_pre pre-filtering."""
    pmz = float(mf.mz)
    ppm = bool(config.ppm)
    tol1 = float(config.ms1_tol)
    tol2 = float(config.ms2_tol)
    tol1_norm = tol1 if ppm else max(tol1 / max(pmz, 1.0) * 1e6, 1.0)

    orig_adduct = mf.adduct
    mf.adduct = adduct
    try:
        mf.data_preprocess(
            ppm, tol1, tol2,
            config.isotope_bin_mztol, config.max_isotope_cnt,
            config.rel_int_denoise_cutoff, config.top_n_per_50_da,
        )
        proc_ms2 = mf.ms2_processed
        proc_ms1 = mf.ms1_processed
    finally:
        mf.adduct = orig_adduct

    p_mzs = proc_ms2.mz_array.astype(np.float64) if proc_ms2 else np.empty(0, dtype=np.float64)
    p_ints = proc_ms2.int_array.astype(np.float64) if proc_ms2 else np.empty(0, dtype=np.float64)
    has_loss = adduct.loss_formula is not None
    loss_arr = adduct.loss_formula.array.astype(np.int16) if has_loss else np.zeros(12, dtype=np.int16)

    c_arr, c_db, c_int, c_err = generate_candidates_fast_numba(
        pmz, p_mzs, p_ints, ppm, tol1, tol2,
        adduct.net_formula.array.astype(np.int16), int(adduct.m), int(adduct.charge),
        float(adduct.net_formula.mass), loss_arr, has_loss,
        config.ele_lower.astype(np.int16), config.ele_upper.astype(np.int16), int(config.db_mode),
        store.basic_mass, store.basic_form, store.basic_idx,
        store.halogen_mass, store.halogen_form, store.halogen_idx,
    )
    n_gen = len(c_arr)
    if n_gen == 0:
        return None

    has_ms1 = bool(proc_ms1) and len(proc_ms1) > 1
    if has_ms1:
        n_iso = min(len(proc_ms1), 4)
        obs_iso_ints = np.ascontiguousarray(proc_ms1.int_array[:n_iso], dtype=np.float64)
        obs_iso_mzs = np.ascontiguousarray(proc_ms1.mz_array[:n_iso], dtype=np.float64)
    else:
        n_iso = 0
        obs_iso_ints = np.zeros(4, dtype=np.float64)
        obs_iso_mzs = np.zeros(4, dtype=np.float64)

    ll_all, res_all = element_ll_numpy(ep, c_arr)
    sum_ll_all = ll_all.sum(axis=1)
    # Negate generate_candidates_fast_numba (theo - obs) -> (obs - theo) matching MS2 signed ppm!
    err_ppm_all = (-c_err.astype(np.float64)) if ppm else (-c_err.astype(np.float64) / max(pmz, 1.0) * 1e6)
    abs_ppm_all = np.abs(err_ppm_all)
    mass_ll_all = -0.5 * ((abs_ppm_all / max(tol1_norm / 3.0, 1e-3)) ** 2)
    phys_score = mass_ll_all + 3.0 * c_int.astype(np.float64) + 2.0 * c_db.astype(np.float64)

    if has_ms1:
        all_ch = np.ascontiguousarray(
            np.maximum(c_arr * int(adduct.m) + adduct.net_formula.array.astype(np.int16), 0), dtype=np.int16
        )
        ms1_all_f = compute_ms1_features_numba(all_ch, obs_iso_ints, obs_iso_mzs, n_iso, abs(int(adduct.charge)))
        phys_score = phys_score + 3.5 * ms1_all_f[:, 2].astype(np.float64)
    else:
        ms1_all_f = None

    if n_gen > m_pre:
        pre1 = phys_score + ll_w * sum_ll_all.astype(np.float64)
        prot_n = min(n_protect, m_pre)
        prot_i = np.argpartition(-phys_score, prot_n - 1)[:prot_n]
        pmask = np.zeros(n_gen, dtype=bool)
        pmask[prot_i] = True
        rem_i = np.flatnonzero(~pmask)
        need_pre = m_pre - prot_n
        top_pre = rem_i[np.argpartition(-pre1[rem_i], need_pre - 1)[:need_pre]] if len(rem_i) > need_pre else rem_i
        pre_idx = np.concatenate([prot_i, top_pre])
    else:
        pre_idx = np.arange(n_gen, dtype=np.intp)

    am5_1 = extract_adduct_meta5(qm)
    am5_pre = np.broadcast_to(am5_1[None, :], (len(pre_idx), 5))
    f_feats_pre = build_formula_features_v3(c_arr[pre_idx], am5_pre)

    return {
        'adduct': adduct, 'n_gen': n_gen, 'tol1_norm': tol1_norm, 'has_ms1': has_ms1,
        'c_arr': c_arr[pre_idx], 'c_db': c_db[pre_idx], 'c_int': c_int[pre_idx],
        'c_err': c_err[pre_idx], 'err_ppm': err_ppm_all[pre_idx],
        'll': ll_all[pre_idx], 'res': res_all[pre_idx], 'sum_ll': sum_ll_all[pre_idx],
        'phys_score': phys_score[pre_idx],
        'ms1_f': (ms1_all_f[pre_idx] if ms1_all_f is not None else None),
        'f_feats_pre': f_feats_pre,
    }


def _stage2_finalize_v5(
    pre: dict,
    cl_pre: np.ndarray,
    ep: np.ndarray,
    pr_p: np.ndarray,
    qm: np.ndarray,
    peak_mzs64: np.ndarray,
    peak_ints64: np.ndarray,
    n_v: int,
    config,
    store,
    k_max: int,
    n_protect: int,
    ll_w: float,
    cl_w: float,
):
    """v5 Stage-2 (nogil Numba): top-K selection with cl_sim + 48-d MS2 physics + 179-d assembly."""
    adduct = pre['adduct']
    n_gen = pre['n_gen']
    tol1_norm = pre['tol1_norm']
    has_ms1 = pre['has_ms1']
    phys_pre = pre['phys_score']
    sum_ll_pre = pre['sum_ll']
    comb_pre = phys_pre + ll_w * sum_ll_pre.astype(np.float64) + cl_w * cl_pre.astype(np.float64)
    n_pre = len(cl_pre)

    if n_pre > k_max:
        prot_n = min(n_protect, k_max)
        prot_loc = np.argpartition(-phys_pre, prot_n - 1)[:prot_n]
        prot_mask = np.zeros(n_pre, dtype=bool)
        prot_mask[prot_loc] = True
        rem_loc = np.flatnonzero(~prot_mask)
        need = k_max - prot_n
        top_rem = rem_loc[np.argpartition(-comb_pre[rem_loc], need - 1)[:need]] if len(rem_loc) > need else rem_loc
        sel = np.concatenate([prot_loc, top_rem])
        sel = sel[np.argsort(-comb_pre[sel], kind='stable')]
    else:
        sel = np.argsort(-comb_pre, kind='stable')

    K = len(sel)
    sub_arr = np.ascontiguousarray(pre['c_arr'][sel], dtype=np.int16)
    sub_db = pre['c_db'][sel]
    sub_int = pre['c_int'][sel]
    sub_err_raw = pre['c_err'][sel]
    sub_err_ppm = pre['err_ppm'][sel].astype(np.float32)
    sub_ll = pre['ll'][sel]
    sub_res = pre['res'][sel]
    sub_sum_ll = sum_ll_pre[sel]
    sub_cl = cl_pre[sel].astype(np.float32)
    sub_ch = np.ascontiguousarray(
        np.maximum(sub_arr * int(adduct.m) + adduct.net_formula.array.astype(np.int16), 0), dtype=np.int16
    )

    ms2_f48, best_frags, best_found = compute_ms2_evidence_fast_v3(
        sub_arr, sub_ch, peak_mzs64, peak_ints64, n_v, int(adduct.charge),
        bool(config.ppm), float(config.ms2_tol),
        store.basic_mass, store.basic_idx, store.halogen_mass, store.halogen_idx,
        store.common_loss_keys, store.common_frag_db,
    )
    ms2_f48 = np.nan_to_num(ms2_f48, nan=0.0, posinf=0.0, neginf=0.0)
    ms2_f = ms2_f48[:, :24]
    ms2_new24 = ms2_f48[:, 24:48]

    cf = np.zeros((K, 112), dtype=np.float32)
    c_f = sub_arr.astype(np.float32)
    ch_f = sub_ch.astype(np.float32)
    masses = c_f.astype(np.float64) @ ELEM_MASSES_F64
    masses_safe = np.maximum(masses, 1.0)
    mass_fracs = (c_f.astype(np.float64) * ELEM_MASSES_F64[None, :]) / masses_safe[:, None]

    dbe = c_f[:, 0] + 1.0 - 0.5 * (c_f[:, 1] + c_f[:, 2] + c_f[:, 3] + c_f[:, 4] + c_f[:, 5] + c_f[:, 6] + c_f[:, 8]) + 0.5 * (c_f[:, 7] + c_f[:, 10])
    c_safe = np.maximum(c_f[:, 0], 1.0)
    tot_atoms = np.maximum(c_f.sum(axis=1), 1.0)
    hc = c_f[:, 1] / c_safe
    nc = c_f[:, 7] / c_safe
    oc = c_f[:, 9] / c_safe
    halo_c = (c_f[:, 2] + c_f[:, 3] + c_f[:, 4] + c_f[:, 5]) / c_safe
    hetero_r = c_f[:, 2:].sum(axis=1) / tot_atoms

    col = 0
    cf[:, col:col + 12] = np.log1p(c_f); col += 12
    cf[:, col:col + 12] = mass_fracs.astype(np.float32); col += 12
    cf[:, col:col + 12] = np.log1p(ch_f); col += 12
    cf[:, col] = dbe / 20.0; col += 1
    cf[:, col] = dbe / c_safe; col += 1
    cf[:, col] = hc; col += 1
    cf[:, col] = nc; col += 1
    cf[:, col] = oc; col += 1
    cf[:, col] = halo_c; col += 1
    cf[:, col] = hetero_r; col += 1
    cf[:, col] = np.log1p(masses_safe / 500.0).astype(np.float32); col += 1
    cf[:, col] = (masses - np.rint(masses)).astype(np.float32); col += 1
    cf[:, col] = sub_db.astype(np.float32); col += 1

    abs_ppm = np.abs(sub_err_ppm)
    tol1_div = max(tol1_norm, 1.0)
    cf[:, col] = sub_err_ppm / tol1_div; col += 1
    cf[:, col] = abs_ppm / tol1_div; col += 1
    cf[:, col] = np.exp(-0.5 * np.minimum((abs_ppm / max(tol1_norm / 3.0, 1e-3)) ** 2, 16.0)); col += 1
    cf[:, col] = sub_int; col += 1

    cf[:, col:col + 12] = np.clip(sub_ll, -20.0, 5.0); col += 12
    cf[:, col] = np.clip(sub_sum_ll, -60.0, 20.0) / 10.0; col += 1
    cf[:, col] = np.clip(sub_ll.min(axis=1), -20.0, 5.0); col += 1
    cf[:, col] = np.clip(sub_ll[:, 2:].sum(axis=1), -50.0, 10.0) / 10.0; col += 1
    cf[:, col:col + 12] = np.clip(sub_res, -10.0, 10.0); col += 12

    cf[:, col] = (dbe - pr_p[0]) / 10.0; col += 1
    cf[:, col] = hc - pr_p[1]; col += 1
    cf[:, col] = hetero_r - pr_p[2]; col += 1

    cf[:, col:col + 24] = ms2_f; col += 24

    min_ppm = float(abs_ppm.min())
    max_ll = float(sub_sum_ll.max())
    max_exp = float(ms2_f[:, 1].max())
    max_dbf = float(ms2_f[:, 7].max())
    max_sub = float(ms2_f[:, 14].max())
    max_bu = float(sub_int.max())
    rk_ll = np.argsort(np.argsort(-sub_sum_ll, kind='stable')).astype(np.float32) / max(K, 1)
    rk_exp = np.argsort(np.argsort(-ms2_f[:, 1], kind='stable')).astype(np.float32) / max(K, 1)
    cf[:, col] = (abs_ppm - min_ppm) / tol1_div; col += 1
    cf[:, col] = np.clip(sub_sum_ll - max_ll, -40.0, 0.0) / 10.0; col += 1
    cf[:, col] = ms2_f[:, 1] - max_exp; col += 1
    cf[:, col] = ms2_f[:, 7] - max_dbf; col += 1
    cf[:, col] = ms2_f[:, 14] - max_sub; col += 1
    cf[:, col] = sub_int - max_bu; col += 1
    cf[:, col] = rk_ll; col += 1
    cf[:, col] = rk_exp; col += 1

    key_cols = (47, 62, 64, 81, 87, 49)
    signs = (-1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    rel12 = np.zeros((K, 12), dtype=np.float32)
    for j, (kc, sgn) in enumerate(zip(key_cols, signs)):
        v = cf[:, kc] * sgn
        if K > 1:
            p2 = np.partition(-v, 1)
            b1, b2 = -float(p2[0]), -float(p2[1])
            ref = np.where(v >= b1 - 1e-6, b2, b1)
        else:
            ref = v
        rel12[:, 2 * j] = np.clip(v - ref, -10.0, 10.0)
        mu = float(v.mean())
        sd = math.sqrt(float(((v - mu) ** 2).mean()) + 1e-4)
        rel12[:, 2 * j + 1] = np.clip((v - mu) / sd, -5.0, 5.0)

    if has_ms1 and pre['ms1_f'] is not None:
        ms1_12 = pre['ms1_f'][sel]
        max_iso = float(ms1_12[:, 2].max())
    else:
        ms1_12 = np.zeros((K, MS1_FEAT_DIM), dtype=np.float32)
        max_iso = 0.0

    sorted_ll = np.sort(sub_sum_ll)[::-1]
    gap_ll = float(sorted_ll[0] - sorted_ll[1]) if K > 1 else 0.0

    has_ms2_exp = (ms2_f[:, 0] > 0.0)
    recal_mean = np.where(has_ms2_exp, np.abs(sub_err_ppm - ms2_new24[:, 0]), abs_ppm)
    recal_med = np.where(has_ms2_exp, np.abs(sub_err_ppm - ms2_new24[:, 1]), abs_ppm)
    min_recal = float(recal_mean.min())
    recal4 = np.stack([
        recal_mean / tol1_div,
        recal_med / tol1_div,
        (abs_ppm - recal_mean) / tol1_div,
        (recal_mean - min_recal) / tol1_div,
    ], axis=1).astype(np.float32)

    max_cl = float(sub_cl.max())
    if K > 1:
        p2 = np.partition(-sub_cl, 1)
        cl_b1, cl_b2 = -float(p2[0]), -float(p2[1])
        cl_ref = np.where(sub_cl >= cl_b1 - 1e-6, cl_b2, cl_b1)
        gap_cl = cl_b1 - cl_b2
    else:
        cl_ref = sub_cl
        gap_cl = 0.0
    cl_mu = float(sub_cl.mean())
    cl_sd = math.sqrt(float(((sub_cl - cl_mu) ** 2).mean()) + 1e-4)
    cl_rk = np.argsort(np.argsort(-sub_cl, kind='stable')).astype(np.float32) / max(K, 1)
    cl_ex = np.exp(np.clip(sub_cl - max_cl, -30.0, 0.0))
    cl_prob = cl_ex / max(float(cl_ex.sum()), 1e-12)
    cl7 = np.stack([
        sub_cl / 10.0,
        (sub_cl - max_cl) / 10.0,
        np.clip((sub_cl - cl_ref) / 10.0, -5.0, 5.0),
        np.clip((sub_cl - cl_mu) / cl_sd, -5.0, 5.0),
        cl_rk,
        (sub_cl / 10.0) - cf[:, 62],
        cl_prob,
    ], axis=1).astype(np.float32)

    new_rel_vals = (ms2_new24[:, 4], ms2_new24[:, 6], ms2_new24[:, 10], -recal4[:, 0])
    rel8 = np.zeros((K, 8), dtype=np.float32)
    for j, v in enumerate(new_rel_vals):
        if K > 1:
            p2 = np.partition(-v, 1)
            b1, b2 = -float(p2[0]), -float(p2[1])
            ref = np.where(v >= b1 - 1e-6, b2, b1)
        else:
            ref = v
        rel8[:, 2 * j] = np.clip(v - ref, -10.0, 10.0)
        mu = float(v.mean())
        sd = math.sqrt(float(((v - mu) ** 2).mean()) + 1e-4)
        rel8[:, 2 * j + 1] = np.clip((v - mu) / sd, -5.0, 5.0)

    cf_full = np.clip(
        np.concatenate([cf, rel12, ms1_12, ms2_new24, recal4, cl7, rel8], axis=1), -1e4, 1e4
    ).astype(np.float32)
    pool_f = np.array([
        math.log1p(n_gen) / 10.0,
        math.log1p(K) / 5.0,
        min_ppm / tol1_div,
        np.clip(max_ll, -60.0, 20.0) / 10.0,
        np.clip(gap_ll, 0.0, 30.0) / 10.0,
        max_exp,
        max_dbf,
        float(n_v) / 64.0,
        1.0 if has_ms1 else 0.0,
        max_iso,
        max_cl / 10.0,
        gap_cl / 10.0,
        min_recal / tol1_div,
        float(ms2_new24[:, 4].max()),
    ], dtype=np.float32)
    spec_ctx = np.concatenate([
        np.clip(ep, -20.0, 20.0),
        np.clip(pr_p, -20.0, 20.0),
        np.clip(qm, -20.0, 20.0),
    ]).astype(np.float32)

    return {
        'adduct': adduct, 'K': K, 'sub_arr': sub_arr, 'sub_ch': sub_ch,
        'sub_db': sub_db, 'sub_int': sub_int, 'sub_err_raw': sub_err_raw,
        'masses': masses, 'ms1_sim': ms1_12[:, 2] if has_ms1 else None,
        'best_frags': best_frags, 'best_found': best_found,
        'peak_mzs64': peak_mzs64, 'n_v': n_v,
        'cf_full': cf_full, 'pool_f': pool_f, 'spec_ctx': spec_ctx,
    }


def _prepare_single_hypothesis(
    mf,
    adduct: Adduct,
    ep: np.ndarray,
    pr_p: np.ndarray,
    peak_mzs64: np.ndarray,
    peak_ints64: np.ndarray,
    n_v: int,
    config,
    store,
    k_max: int,
    n_protect: int,
    ll_w: float,
    qm: Optional[np.ndarray] = None,
    z_sp: Optional[np.ndarray] = None,
    fenc_np: Optional[dict] = None,
    logit_scale: float = 1.0,
    cl_w: float = 0.0,
    format_version: int = 4,
):
    """Run Stage-1 + Stage-2 for one (MetaFeature, Adduct) hypothesis (format_version 4 or 5)."""
    pmz = float(mf.mz)
    ppm = bool(config.ppm)
    tol1 = float(config.ms1_tol)
    tol2 = float(config.ms2_tol)
    tol1_norm = tol1 if ppm else max(tol1 / max(pmz, 1.0) * 1e6, 1.0)

    # Preprocess MS1 & MS2 using this adduct's charge
    orig_adduct = mf.adduct
    mf.adduct = adduct
    try:
        mf.data_preprocess(
            ppm, tol1, tol2,
            config.isotope_bin_mztol, config.max_isotope_cnt,
            config.rel_int_denoise_cutoff, config.top_n_per_50_da,
        )
        proc_ms2 = mf.ms2_processed
        proc_ms1 = mf.ms1_processed
    finally:
        mf.adduct = orig_adduct

    p_mzs = proc_ms2.mz_array.astype(np.float64) if proc_ms2 else np.empty(0, dtype=np.float64)
    p_ints = proc_ms2.int_array.astype(np.float64) if proc_ms2 else np.empty(0, dtype=np.float64)
    has_loss = adduct.loss_formula is not None
    loss_arr = adduct.loss_formula.array.astype(np.int16) if has_loss else np.zeros(12, dtype=np.int16)

    c_arr, c_db, c_int, c_err = generate_candidates_fast_numba(
        pmz, p_mzs, p_ints, ppm, tol1, tol2,
        adduct.net_formula.array.astype(np.int16), int(adduct.m), int(adduct.charge),
        float(adduct.net_formula.mass), loss_arr, has_loss,
        config.ele_lower.astype(np.int16), config.ele_upper.astype(np.int16), int(config.db_mode),
        store.basic_mass, store.basic_form, store.basic_idx,
        store.halogen_mass, store.halogen_form, store.halogen_idx,
    )
    n_gen = len(c_arr)
    if n_gen == 0:
        return None

    has_ms1 = bool(proc_ms1) and len(proc_ms1) > 1
    if has_ms1:
        n_iso = min(len(proc_ms1), 4)
        obs_iso_ints = np.ascontiguousarray(proc_ms1.int_array[:n_iso], dtype=np.float64)
        obs_iso_mzs = np.ascontiguousarray(proc_ms1.mz_array[:n_iso], dtype=np.float64)
    else:
        n_iso = 0
        obs_iso_ints = np.zeros(4, dtype=np.float64)
        obs_iso_mzs = np.zeros(4, dtype=np.float64)

    ll_all, res_all = element_ll_numpy(ep, c_arr)
    sum_ll_all = ll_all.sum(axis=1)
    # generate_candidates_fast_numba returns (theo - obs) for CandidateFormula.mz_error API compatibility;
    # Stage-2/3 features & MS1-MS2 self-recalibration use (obs - theo) matching MS2 signed ppm!
    err_ppm_all = (-c_err.astype(np.float64)) if ppm else (-c_err.astype(np.float64) / max(pmz, 1.0) * 1e6)
    abs_ppm_all = np.abs(err_ppm_all)
    mass_ll_all = -0.5 * ((abs_ppm_all / max(tol1_norm / 3.0, 1e-3)) ** 2)
    phys_score = mass_ll_all + 3.0 * c_int.astype(np.float64) + 2.0 * c_db.astype(np.float64)

    if has_ms1:
        all_ch = np.ascontiguousarray(
            np.maximum(c_arr * int(adduct.m) + adduct.net_formula.array.astype(np.int16), 0), dtype=np.int16
        )
        ms1_all_f = compute_ms1_features_numba(all_ch, obs_iso_ints, obs_iso_mzs, n_iso, abs(int(adduct.charge)))
        phys_score = phys_score + 3.5 * ms1_all_f[:, 2].astype(np.float64)
    else:
        ms1_all_f = None

    is_v5 = (format_version == 5)
    if is_v5:
        # Fast two-pass Stage-1 shortlisting: pre-filter very large pools (>256) by phys + ll
        # (retaining >=99.9% of rec_full!) so FormulaEncoderV3 runs on <=256 formulas (<1.2 ms/spec).
        m_pre = max(256, k_max * 3)
        if n_gen > m_pre:
            pre1 = phys_score + ll_w * sum_ll_all.astype(np.float64)
            prot_n = min(n_protect, m_pre)
            prot_i = np.argpartition(-phys_score, prot_n - 1)[:prot_n]
            pmask = np.zeros(n_gen, dtype=bool)
            pmask[prot_i] = True
            rem_i = np.flatnonzero(~pmask)
            need_pre = m_pre - prot_n
            top_pre = rem_i[np.argpartition(-pre1[rem_i], need_pre - 1)[:need_pre]] if len(rem_i) > need_pre else rem_i
            pre_idx = np.concatenate([prot_i, top_pre])
        else:
            pre_idx = np.arange(n_gen, dtype=np.intp)

        am5_1 = extract_adduct_meta5(qm)
        am5_pre = np.broadcast_to(am5_1[None, :], (len(pre_idx), 5))
        f_feats_pre = build_formula_features_v3(c_arr[pre_idx], am5_pre)
        zf_pre = formula_enc_forward_np(f_feats_pre, fenc_np)
        cl_pre = (zf_pre @ z_sp) * logit_scale

        phys_pre = phys_score[pre_idx]
        comb_pre = phys_pre + ll_w * sum_ll_all[pre_idx].astype(np.float64) + cl_w * cl_pre.astype(np.float64)
        n_pre = len(pre_idx)
        if n_pre > k_max:
            prot_n = min(n_protect, k_max)
            prot_loc = np.argpartition(-phys_pre, prot_n - 1)[:prot_n]
            prot_mask = np.zeros(n_pre, dtype=bool)
            prot_mask[prot_loc] = True
            rem_loc = np.flatnonzero(~prot_mask)
            need = k_max - prot_n
            top_rem = rem_loc[np.argpartition(-comb_pre[rem_loc], need - 1)[:need]] if len(rem_loc) > need else rem_loc
            sel_loc = np.concatenate([prot_loc, top_rem])
            sel_loc = sel_loc[np.argsort(-comb_pre[sel_loc], kind='stable')]
        else:
            sel_loc = np.argsort(-comb_pre, kind='stable')
        sel = pre_idx[sel_loc]
        K = len(sel)
        sub_cl = cl_pre[sel_loc].astype(np.float32)
    else:
        pre_score = ll_w * sum_ll_all + phys_score
        order_main = np.argsort(-pre_score, kind='stable')
        K = min(n_gen, k_max)
        if n_protect > 0 and n_gen > K:
            n_main = max(K - n_protect, 1)
            sel_main = order_main[:n_main]
            picked = np.zeros(n_gen, dtype=bool)
            picked[sel_main] = True
            order_phys = np.argsort(-phys_score, kind='stable')
            rem_phys = order_phys[~picked[order_phys]][:K - n_main]
            sel = np.concatenate([sel_main, rem_phys])
        else:
            sel = order_main[:K]
        sub_cl = None

    sub_arr = np.ascontiguousarray(c_arr[sel], dtype=np.int16)
    sub_db = c_db[sel]
    sub_int = c_int[sel]
    sub_err_raw = c_err[sel]
    sub_err_ppm = err_ppm_all[sel].astype(np.float32)
    sub_ll = ll_all[sel]
    sub_res = res_all[sel]
    sub_sum_ll = sum_ll_all[sel]
    sub_ch = np.ascontiguousarray(
        np.maximum(sub_arr * int(adduct.m) + adduct.net_formula.array.astype(np.int16), 0), dtype=np.int16
    )

    if is_v5:
        ms2_f48, best_frags, best_found = compute_ms2_evidence_fast_v3(
            sub_arr, sub_ch, peak_mzs64, peak_ints64, n_v, int(adduct.charge),
            ppm, tol2,
            store.basic_mass, store.basic_idx, store.halogen_mass, store.halogen_idx,
            store.common_loss_keys, store.common_frag_db,
        )
        ms2_f48 = np.nan_to_num(ms2_f48, nan=0.0, posinf=0.0, neginf=0.0)
        ms2_f = ms2_f48[:, :24]
        ms2_new24 = ms2_f48[:, 24:48]
    else:
        ms2_f, best_frags, best_found = compute_ms2_evidence_fast(
            sub_arr, sub_ch, peak_mzs64, peak_ints64, n_v, int(adduct.charge),
            ppm, tol2,
            store.basic_mass, store.basic_idx, store.halogen_mass, store.halogen_idx,
            store.common_loss_keys, store.common_frag_db,
        )
        ms2_f = np.nan_to_num(ms2_f, nan=0.0, posinf=0.0, neginf=0.0)
        ms2_new24 = None

    # Assemble 112 base + 12 rel12 + 12 ms1_12 features (136 dims; +43 dims for v5 -> 179 dims)
    cf = np.zeros((K, 112), dtype=np.float32)
    c_f = sub_arr.astype(np.float32)
    ch_f = sub_ch.astype(np.float32)
    masses = c_f.astype(np.float64) @ ELEM_MASSES_F64
    masses_safe = np.maximum(masses, 1.0)
    mass_fracs = (c_f.astype(np.float64) * ELEM_MASSES_F64[None, :]) / masses_safe[:, None]

    dbe = c_f[:, 0] + 1.0 - 0.5 * (c_f[:, 1] + c_f[:, 2] + c_f[:, 3] + c_f[:, 4] + c_f[:, 5] + c_f[:, 6] + c_f[:, 8]) + 0.5 * (c_f[:, 7] + c_f[:, 10])
    c_safe = np.maximum(c_f[:, 0], 1.0)
    tot_atoms = np.maximum(c_f.sum(axis=1), 1.0)
    hc = c_f[:, 1] / c_safe
    nc = c_f[:, 7] / c_safe
    oc = c_f[:, 9] / c_safe
    halo_c = (c_f[:, 2] + c_f[:, 3] + c_f[:, 4] + c_f[:, 5]) / c_safe
    hetero_r = c_f[:, 2:].sum(axis=1) / tot_atoms

    col = 0
    cf[:, col:col + 12] = np.log1p(c_f); col += 12
    cf[:, col:col + 12] = mass_fracs.astype(np.float32); col += 12
    cf[:, col:col + 12] = np.log1p(ch_f); col += 12
    cf[:, col] = dbe / 20.0; col += 1
    cf[:, col] = dbe / c_safe; col += 1
    cf[:, col] = hc; col += 1
    cf[:, col] = nc; col += 1
    cf[:, col] = oc; col += 1
    cf[:, col] = halo_c; col += 1
    cf[:, col] = hetero_r; col += 1
    cf[:, col] = np.log1p(masses_safe / 500.0).astype(np.float32); col += 1
    cf[:, col] = (masses - np.rint(masses)).astype(np.float32); col += 1
    cf[:, col] = sub_db.astype(np.float32); col += 1

    abs_ppm = np.abs(sub_err_ppm)
    tol1_div = max(tol1_norm, 1.0)
    cf[:, col] = sub_err_ppm / tol1_div; col += 1
    cf[:, col] = abs_ppm / tol1_div; col += 1
    cf[:, col] = np.exp(-0.5 * np.minimum((abs_ppm / max(tol1_norm / 3.0, 1e-3)) ** 2, 16.0)); col += 1
    cf[:, col] = sub_int; col += 1

    cf[:, col:col + 12] = np.clip(sub_ll, -20.0, 5.0); col += 12
    cf[:, col] = np.clip(sub_sum_ll, -60.0, 20.0) / 10.0; col += 1
    cf[:, col] = np.clip(sub_ll.min(axis=1), -20.0, 5.0); col += 1
    cf[:, col] = np.clip(sub_ll[:, 2:].sum(axis=1), -50.0, 10.0) / 10.0; col += 1
    cf[:, col:col + 12] = np.clip(sub_res, -10.0, 10.0); col += 12

    cf[:, col] = (dbe - pr_p[0]) / 10.0; col += 1
    cf[:, col] = hc - pr_p[1]; col += 1
    cf[:, col] = hetero_r - pr_p[2]; col += 1

    cf[:, col:col + 24] = ms2_f; col += 24

    min_ppm = float(abs_ppm.min())
    max_ll = float(sub_sum_ll.max())
    max_exp = float(ms2_f[:, 1].max())
    max_dbf = float(ms2_f[:, 7].max())
    max_sub = float(ms2_f[:, 14].max())
    max_bu = float(sub_int.max())
    rk_ll = np.argsort(np.argsort(-sub_sum_ll, kind='stable')).astype(np.float32) / max(K, 1)
    rk_exp = np.argsort(np.argsort(-ms2_f[:, 1], kind='stable')).astype(np.float32) / max(K, 1)
    cf[:, col] = (abs_ppm - min_ppm) / tol1_div; col += 1
    cf[:, col] = np.clip(sub_sum_ll - max_ll, -40.0, 0.0) / 10.0; col += 1
    cf[:, col] = ms2_f[:, 1] - max_exp; col += 1
    cf[:, col] = ms2_f[:, 7] - max_dbf; col += 1
    cf[:, col] = ms2_f[:, 14] - max_sub; col += 1
    cf[:, col] = sub_int - max_bu; col += 1
    cf[:, col] = rk_ll; col += 1
    cf[:, col] = rk_exp; col += 1

    key_cols = (47, 62, 64, 81, 87, 49)
    signs = (-1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    rel12 = np.zeros((K, 12), dtype=np.float32)
    for j, (kc, sgn) in enumerate(zip(key_cols, signs)):
        v = cf[:, kc] * sgn
        if K > 1:
            p2 = np.partition(-v, 1)
            b1, b2 = -float(p2[0]), -float(p2[1])
            ref = np.where(v >= b1 - 1e-6, b2, b1)
        else:
            ref = v
        rel12[:, 2 * j] = np.clip(v - ref, -10.0, 10.0)
        mu = float(v.mean())
        sd = math.sqrt(float(((v - mu) ** 2).mean()) + 1e-4)
        rel12[:, 2 * j + 1] = np.clip((v - mu) / sd, -5.0, 5.0)

    if has_ms1 and ms1_all_f is not None:
        ms1_12 = ms1_all_f[sel]
        max_iso = float(ms1_12[:, 2].max())
    else:
        ms1_12 = np.zeros((K, MS1_FEAT_DIM), dtype=np.float32)
        max_iso = 0.0

    sorted_ll = np.sort(sub_sum_ll)[::-1]
    gap_ll = float(sorted_ll[0] - sorted_ll[1]) if K > 1 else 0.0

    if is_v5:
        has_ms2_exp = (ms2_f[:, 0] > 0.0)
        recal_mean = np.where(has_ms2_exp, np.abs(sub_err_ppm - ms2_new24[:, 0]), abs_ppm)
        recal_med = np.where(has_ms2_exp, np.abs(sub_err_ppm - ms2_new24[:, 1]), abs_ppm)
        min_recal = float(recal_mean.min())
        recal4 = np.stack([
            recal_mean / tol1_div,
            recal_med / tol1_div,
            (abs_ppm - recal_mean) / tol1_div,
            (recal_mean - min_recal) / tol1_div,
        ], axis=1).astype(np.float32)

        max_cl = float(sub_cl.max())
        if K > 1:
            p2 = np.partition(-sub_cl, 1)
            cl_b1, cl_b2 = -float(p2[0]), -float(p2[1])
            cl_ref = np.where(sub_cl >= cl_b1 - 1e-6, cl_b2, cl_b1)
            gap_cl = cl_b1 - cl_b2
        else:
            cl_ref = sub_cl
            gap_cl = 0.0
        cl_mu = float(sub_cl.mean())
        cl_sd = math.sqrt(float(((sub_cl - cl_mu) ** 2).mean()) + 1e-4)
        cl_rk = np.argsort(np.argsort(-sub_cl, kind='stable')).astype(np.float32) / max(K, 1)
        cl_ex = np.exp(np.clip(sub_cl - max_cl, -30.0, 0.0))
        cl_prob = cl_ex / max(float(cl_ex.sum()), 1e-12)
        cl7 = np.stack([
            sub_cl / 10.0,
            (sub_cl - max_cl) / 10.0,
            np.clip((sub_cl - cl_ref) / 10.0, -5.0, 5.0),
            np.clip((sub_cl - cl_mu) / cl_sd, -5.0, 5.0),
            cl_rk,
            (sub_cl / 10.0) - cf[:, 62],
            cl_prob,
        ], axis=1).astype(np.float32)

        new_rel_vals = (ms2_new24[:, 4], ms2_new24[:, 6], ms2_new24[:, 10], -recal4[:, 0])
        rel8 = np.zeros((K, 8), dtype=np.float32)
        for j, v in enumerate(new_rel_vals):
            if K > 1:
                p2 = np.partition(-v, 1)
                b1, b2 = -float(p2[0]), -float(p2[1])
                ref = np.where(v >= b1 - 1e-6, b2, b1)
            else:
                ref = v
            rel8[:, 2 * j] = np.clip(v - ref, -10.0, 10.0)
            mu = float(v.mean())
            sd = math.sqrt(float(((v - mu) ** 2).mean()) + 1e-4)
            rel8[:, 2 * j + 1] = np.clip((v - mu) / sd, -5.0, 5.0)

        cf_full = np.clip(
            np.concatenate([cf, rel12, ms1_12, ms2_new24, recal4, cl7, rel8], axis=1), -1e4, 1e4
        ).astype(np.float32)
        pool_f = np.array([
            math.log1p(n_gen) / 10.0,
            math.log1p(K) / 5.0,
            min_ppm / tol1_div,
            np.clip(max_ll, -60.0, 20.0) / 10.0,
            np.clip(gap_ll, 0.0, 30.0) / 10.0,
            max_exp,
            max_dbf,
            float(n_v) / 64.0,
            1.0 if has_ms1 else 0.0,
            max_iso,
            max_cl / 10.0,
            gap_cl / 10.0,
            min_recal / tol1_div,
            float(ms2_new24[:, 4].max()),
        ], dtype=np.float32)
        spec_ctx = np.concatenate([
            np.clip(ep, -20.0, 20.0),
            np.clip(pr_p, -20.0, 20.0),
            np.clip(qm, -20.0, 20.0),
        ]).astype(np.float32)
    else:
        cf_full = np.clip(np.concatenate([cf, rel12, ms1_12], axis=1), -1e4, 1e4).astype(np.float32)
        pool_f = np.array([
            math.log1p(n_gen) / 10.0,
            math.log1p(K) / 5.0,
            min_ppm / tol1_div,
            np.clip(max_ll, -60.0, 20.0) / 10.0,
            np.clip(gap_ll, 0.0, 30.0) / 10.0,
            max_exp,
            max_dbf,
            float(sub_db.mean()),
            1.0 if has_ms1 else 0.0,
            max_iso,
        ], dtype=np.float32)
        spec_ctx = None

    return {
        'adduct': adduct,
        'K': K,
        'sub_arr': sub_arr,
        'sub_ch': sub_ch,
        'sub_db': sub_db,
        'sub_int': sub_int,
        'sub_err_raw': sub_err_raw,
        'masses': masses,
        'ms1_sim': ms1_12[:, 2] if has_ms1 else None,
        'best_frags': best_frags,
        'best_found': best_found,
        'peak_mzs64': peak_mzs64,
        'n_v': n_v,
        'cf_full': cf_full,
        'pool_f': pool_f,
        'spec_ctx': spec_ctx,
    }


def _build_candidate_formula(hyp: dict, k: int, mf, raw_idx_map: Optional[np.ndarray], prob: float,
                             norm_prob: float, fdr: float) -> CandidateFormula:
    adduct = hyp['adduct']
    arr = hyp['sub_arr'][k]
    ch_arr = hyp['sub_ch'][k]
    neutral_f = Formula._from_array(arr.copy(), 0, float(hyp['masses'][k]))
    charged_f = Formula._from_array(ch_arr.copy(), int(adduct.charge))

    cf = CandidateFormula(
        formula=neutral_f,
        charged_formula=charged_f,
        mz_error=float(hyp['sub_err_raw'][k]),
        exp_ms2_sum_int=float(hyp['sub_int'][k]),
        ms2_raw_explanation=None,
        db_existed=bool(hyp['sub_db'][k]),
    )
    # Defer Formula object instantiation for fragment peaks until ms2_raw_explanation is accessed
    found_mask = hyp['best_found'][k, :hyp['n_v']]
    if found_mask.any():
        exp_local = np.flatnonzero(found_mask)
        idx_arr = (raw_idx_map[exp_local] if raw_idx_map is not None else exp_local).astype(np.intp)
        frag_arrs = np.ascontiguousarray(hyp['best_frags'][k, exp_local], dtype=np.int16)
        cf._ms2_raw_explanation_lazy = (idx_arr, frag_arrs, int(adduct.charge))
    cf.adduct = adduct
    if hyp['ms1_sim'] is not None:
        cf.ms1_isotope_similarity = float(hyp['ms1_sim'][k])
    cf.estimated_prob = float(prob)
    cf.normed_estimated_prob = float(norm_prob)
    cf.estimated_fdr = float(fdr)
    return cf


def annotate_batch_v2(batch_data: Sequence, config, gd: dict) -> None:
    """Annotate a batch of MetaFeature objects using the unified msbuddy 2.0 engine."""
    if not batch_data:
        return
    rt = _get_v2_runtime(gd)
    store = get_fast_store(gd)
    k_max = int(config.max_candidates) if config.max_candidates is not None else DEFAULT_K_MAX
    n_protect = min(DEFAULT_PROTECT, max(k_max // 3, 0))
    ms2_tol_ppm = float(config.ms2_tol) if config.ppm else 20.0

    # Expand (feature_index, adduct) hypotheses (1 per feature normally, 5-7 for adduct="auto")
    hyp_specs = []
    peak_cache = {}
    for fi, mf in enumerate(batch_data):
        if mf.ms2_raw is None or len(mf.ms2_raw) == 0:
            raw_mzs = np.empty(0, dtype=np.float64)
            raw_ints = np.empty(0, dtype=np.float64)
        else:
            raw_mzs = np.asarray(mf.ms2_raw.mz_array, dtype=np.float64)
            raw_ints = np.asarray(mf.ms2_raw.int_array, dtype=np.float64)

        pf, pm, pi, pmask, mzs_kept, n_v = featurize_peaks_v2(raw_mzs, raw_ints, float(mf.mz), ms2_tol_ppm=ms2_tol_ppm)
        # Map top-64 sorted peaks back to original raw_mzs indices for MS2Explanation
        raw_idx_map = None
        if n_v > 0 and len(raw_mzs) > 0:
            raw_idx_map = np.array([int(np.argmin(np.abs(raw_mzs - m))) for m in pm[:n_v]], dtype=np.intp)
        peak_cache[fi] = (pf, pm, pi, pmask, raw_mzs, raw_ints, n_v, raw_idx_map)

        if getattr(mf, 'adduct_auto', False):
            adduct_strs = AUTO_ADDUCTS_POS if mf.adduct.charge > 0 else AUTO_ADDUCTS_NEG
            adducts = [Adduct(a, a.endswith('+'), report_invalid=True) for a in adduct_strs]
        else:
            adducts = [mf.adduct]

        inst = getattr(mf, 'instrument', None) or getattr(config, 'ms_instr', None)
        ce = getattr(mf, 'collision_energy', None)
        for ion in adducts:
            qm = build_query_meta_60(
                raw_mzs, raw_ints, float(mf.mz),
                ion.net_formula.array.astype(np.float32),
                int(ion.charge), int(ion.m), float(ion.net_formula.mass),
                instrument=inst, collision_energy=ce, ms2_tol_ppm=ms2_tol_ppm,
            )
            hyp_specs.append((fi, ion, qm))

    # Resolve effective worker thread count for shared-memory parallel execution
    if getattr(config, 'parallel', False):
        req_cpu = int(getattr(config, 'n_cpu', -1))
        n_threads = (os.cpu_count() or 1) if req_cpu <= 0 else req_cpu
    else:
        n_threads = 1
    n_threads = max(1, min(n_threads, max(1, len(hyp_specs))))

    # Stage 0: Batched CPU ONNX inference across all hypotheses (dynamic peak length per batch)
    fv = int(rt['format_version'])
    is_v5 = (fv == 5)
    cand_dim = int(rt['cand_feat_dim'])
    pool_dim = int(rt['pool_feat_dim'])
    n_hyp = len(hyp_specs)
    ep_all = np.zeros((n_hyp, 34), dtype=np.float32)
    pr_all = np.zeros((n_hyp, 3), dtype=np.float32)
    z_all = np.zeros((n_hyp, CL_DIM), dtype=np.float32) if is_v5 else None
    s0_sess = rt['stage0']
    # Sort indices by peak count so short spectra never pay O(64^2) attention cost
    hyp_order = sorted(range(n_hyp), key=lambda idx: peak_cache[hyp_specs[idx][0]][6])
    s0_chunks = [hyp_order[a:a + 32] for a in range(0, n_hyp, 32)]

    def _run_s0_chunk(batch_ids: List[int]) -> None:
        bs = len(batch_ids)
        nb = max(1, max(peak_cache[hyp_specs[idx][0]][6] for idx in batch_ids))
        pf_b = np.empty((bs, nb, PEAK_FEAT_DIM), dtype=np.float32)
        pm_b = np.empty((bs, nb), dtype=np.float32)
        mk_b = np.empty((bs, nb), dtype=bool)
        qm_b = np.empty((bs, QUERY_META_60_DIM), dtype=np.float32)
        for j, idx in enumerate(batch_ids):
            fi, _, qm = hyp_specs[idx]
            pf, pm, _, pmask, _, _, _, _ = peak_cache[fi]
            pf_b[j] = pf[:nb]; pm_b[j] = pm[:nb]; mk_b[j] = pmask[:nb]; qm_b[j] = qm
        outs = s0_sess.run(None, {
            'peak_feats': pf_b, 'peak_mzs': pm_b, 'peak_mask': mk_b, 'query_meta': qm_b,
        })
        ep_o, pr_o = outs[0], outs[1]
        z_o = outs[2] if is_v5 else None
        for j, idx in enumerate(batch_ids):
            ep_all[idx] = ep_o[j]
            pr_all[idx] = pr_o[j]
            if is_v5:
                z_all[idx] = z_o[j]

    def _run_hyp(hi: int):
        fi, ion, qm = hyp_specs[hi]
        _, pm, pi, _, _, _, n_v, _ = peak_cache[fi]
        return _prepare_single_hypothesis(
            batch_data[fi], ion, ep_all[hi], pr_all[hi],
            pm, pi, n_v, config, store, k_max, n_protect, rt['ll_w'],
            qm=qm,
            z_sp=(z_all[hi] if is_v5 else None),
            fenc_np=rt['formula_enc_np'],
            logit_scale=rt['logit_scale'],
            cl_w=rt['cl_w'],
            format_version=fv,
        )

    pool = ThreadPoolExecutor(max_workers=n_threads) if n_threads > 1 else None
    try:
        if pool is not None and len(s0_chunks) > 1:
            list(pool.map(_run_s0_chunk, s0_chunks))
        else:
            for ch in s0_chunks:
                _run_s0_chunk(ch)

        if is_v5:
            m_pre = max(160, k_max * 2)

            def _run_s1a(hi: int):
                fi, ion, qm = hyp_specs[hi]
                return _stage1a_gen_and_prefilter(
                    batch_data[fi], ion, ep_all[hi], config, store,
                    k_max, n_protect, rt['ll_w'], qm, m_pre,
                )

            pre_results = list(pool.map(_run_s1a, range(n_hyp))) if pool is not None else [_run_s1a(hi) for hi in range(n_hyp)]

            # Stage-1B: Batched FormulaEncoderV3 over 4096-row chunks (0.4 ms/spec, zero GIL contention!)
            nonempty_his = [hi for hi, pr in enumerate(pre_results) if pr is not None]
            cl_by_hi: Dict[int, np.ndarray] = {}
            if nonempty_his:
                fenc_np = rt['formula_enc_np']
                lscale = rt['logit_scale']
                f_rows = [pre_results[hi]['f_feats_pre'] for hi in nonempty_his]
                f_lens = [len(r) for r in f_rows]
                X_f = np.concatenate(f_rows, axis=0)
                tot_f = len(X_f)
                Z_f = np.empty((tot_f, CL_DIM), dtype=np.float32)
                chunk_ranges = [(a, min(a + 4096, tot_f)) for a in range(0, tot_f, 4096)]

                def _run_fenc_chunk(rng_ab: Tuple[int, int]) -> None:
                    a, b = rng_ab
                    Z_f[a:b] = formula_enc_forward_np(X_f[a:b], fenc_np)

                if pool is not None and len(chunk_ranges) > 1:
                    list(pool.map(_run_fenc_chunk, chunk_ranges))
                else:
                    for rng_ab in chunk_ranges:
                        _run_fenc_chunk(rng_ab)

                off = 0
                for hi, ln in zip(nonempty_his, f_lens):
                    cl_by_hi[hi] = (Z_f[off:off + ln] @ z_all[hi]) * lscale
                    off += ln

            def _run_s2(hi: int):
                pr = pre_results[hi]
                if pr is None:
                    return None
                fi, _, qm = hyp_specs[hi]
                _, pm, pi, _, _, _, n_v, _ = peak_cache[fi]
                return _stage2_finalize_v5(
                    pr, cl_by_hi[hi], ep_all[hi], pr_all[hi], qm,
                    pm, pi, n_v, config, store, k_max, n_protect, rt['ll_w'], rt['cl_w'],
                )

            hyp_results = list(pool.map(_run_s2, range(n_hyp))) if (pool is not None and len(nonempty_his) > 1) else [_run_s2(hi) for hi in range(n_hyp)]
        else:
            hyp_results = list(pool.map(_run_hyp, range(n_hyp))) if pool is not None else [_run_hyp(hi) for hi in range(n_hyp)]

        valid_hyps = []
        valid_hyp_fi = []
        for hi, hyp in enumerate(hyp_results):
            if hyp is not None:
                valid_hyps.append(hyp)
                valid_hyp_fi.append(hyp_specs[hi][0])

        # Clear features with 0 candidates across all hypotheses
        by_feature: Dict[int, List[dict]] = {fi: [] for fi in range(len(batch_data))}
        if not valid_hyps:
            for mf in batch_data:
                mf.candidate_formula_list = []
            return

        # Stage 3: Batched PureSetReranker ONNX + LightGBM LambdaRank
        s3_sess = rt['stage3_nn']
        s3_lgb = rt['stage3_lgb']
        fc = rt['feat_center']
        fs = rt['feat_scale']
        w_nn = rt['w_nn']
        lgb_scale = rt['lgb_scale']
        temp = rt['temp']
        b_null = rt['b_null']

        max_k_batch = max(h['K'] for h in valid_hyps)
        n_vh = len(valid_hyps)
        nn_logits_all = np.full((n_vh, max_k_batch + 1), -1e9, dtype=np.float32)

        # Stage-3A ONNX in length-bucketed batches of 32 (avoids O(80^2) padding for small K!)
        vh_order = sorted(range(n_vh), key=lambda idx: valid_hyps[idx]['K'])
        s3_chunks = [vh_order[a:a + 32] for a in range(0, n_vh, 32)]

        def _run_s3_chunk(b_ids: List[int]) -> None:
            bs = len(b_ids)
            kb = max(valid_hyps[idx]['K'] for idx in b_ids)
            cf_b = np.zeros((bs, kb, cand_dim), dtype=np.float32)
            cm_b = np.zeros((bs, kb), dtype=bool)
            pf_b = np.zeros((bs, pool_dim), dtype=np.float32)
            zs_b = np.zeros((bs, SPEC_CTX_DIM), dtype=np.float32) if is_v5 else None
            for j, idx in enumerate(b_ids):
                h = valid_hyps[idx]
                k = h['K']
                cf_b[j, :k] = np.clip((h['cf_full'] - fc) / fs, -10.0, 10.0)
                cm_b[j, :k] = True
                pf_b[j] = h['pool_f']
                if is_v5:
                    zs_b[j] = h['spec_ctx']
            feed = {'cand_feats': cf_b, 'cand_mask': cm_b, 'pool_feats': pf_b}
            if is_v5:
                feed['z_spec'] = zs_b
            l_out = s3_sess.run(None, feed)[0]
            for j, idx in enumerate(b_ids):
                k = valid_hyps[idx]['K']
                nn_logits_all[idx, :k] = l_out[j, :k]
                nn_logits_all[idx, max_k_batch] = l_out[j, kb]

        if pool is not None and len(s3_chunks) > 1:
            list(pool.map(_run_s3_chunk, s3_chunks))
        else:
            for ch in s3_chunks:
                _run_s3_chunk(ch)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    # Stage-3B LightGBM over concatenated valid candidates (1 fast vectorized call)
    lgb_rows = []
    offsets = [0]
    for h in valid_hyps:
        k = h['K']
        pf_rep = np.broadcast_to(h['pool_f'][None, :], (k, pool_dim))
        lgb_rows.append(np.concatenate([h['cf_full'], pf_rep], axis=1))
        offsets.append(offsets[-1] + k)
    X_lgb = np.concatenate(lgb_rows, axis=0)
    lgb_preds = s3_lgb.predict(X_lgb, num_threads=n_threads).astype(np.float32)

    for hi, (fi, h) in enumerate(zip(valid_hyp_fi, valid_hyps)):
        k = h['K']
        nn_c = nn_logits_all[hi, :k].astype(np.float64)
        nn_null = float(nn_logits_all[hi, max_k_batch])
        lgb_c = lgb_preds[offsets[hi]:offsets[hi + 1]].astype(np.float64)
        fused_c = w_nn * nn_c + (1.0 - w_nn) * lgb_scale * lgb_c
        # Calibrated log-odds relative to [ABSENT]: (fused - nn_null)/temp - b_null
        h['log_odds'] = (fused_c - nn_null) / temp - b_null
        by_feature[fi].append(h)

    # Stage 4: Calibrated posterior normalization & FDR per MetaFeature (single or adduct="auto")
    for fi, mf in enumerate(batch_data):
        hyps = by_feature[fi]
        if not hyps:
            mf.candidate_formula_list = []
            continue
        raw_idx_map = peak_cache[fi][7]

        # Collect all (hyp, k, log_odds) entries
        is_auto = getattr(mf, 'adduct_auto', False) and len(hyps) > 1
        flat_entries = []
        for h in hyps:
            prior_shift = AUTO_ADDUCT_LOG_PRIORS.get(h['adduct'].string, -1.5) if is_auto else 0.0
            for k in range(h['K']):
                flat_entries.append((h, k, float(h['log_odds'][k]) + prior_shift))
        log_odds_arr = np.array([e[2] for e in flat_entries], dtype=np.float64)
        # Joint softmax with shared [ABSENT] anchor at 0.0
        all_s = np.append(log_odds_arr, 0.0)
        probs_all = np.exp(all_s - logsumexp(all_s))[:-1]

        order = np.argsort(-probs_all, kind='stable')[:k_max]
        cand_list = []
        cum_p = 0.0
        for rank_i, idx in enumerate(order):
            h, k, _ = flat_entries[int(idx)]
            p = float(probs_all[int(idx)])
            cum_p += p
            fdr = max(0.0, min(1.0, 1.0 - cum_p / float(rank_i + 1)))
            cand_list.append(_build_candidate_formula(h, k, mf, raw_idx_map, p, p, fdr))

        mf.candidate_formula_list = cand_list
        if getattr(mf, 'adduct_auto', False) and cand_list:
            mf.adduct = cand_list[0].adduct
            mf.data_preprocess(
                config.ppm, config.ms1_tol, config.ms2_tol,
                config.isotope_bin_mztol, config.max_isotope_cnt,
                config.rel_int_denoise_cutoff, config.top_n_per_50_da,
            )
