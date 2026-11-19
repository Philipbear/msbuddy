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

import hashlib
import json
import math
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
    generate_candidates_fast_numba,
    get_fast_store,
)
from msbuddy.base import Adduct, CandidateFormula, Formula, MS2Explanation

PEAK_WAVELENGTHS = (500.0, 100.0, 20.0, 4.0, 1.0, 0.2, 0.02)
PEAK_FEAT_DIM = 40
CAND_FEAT_DIM = 136
POOL_FEAT_DIM = 10
DEFAULT_K_MAX = 80
DEFAULT_PROTECT = 24
DEFAULT_LL_W = 0.35

AUTO_ADDUCTS_POS = (
    '[M+H]+', '[M+Na]+', '[M+NH4]+', '[M+K]+',
    '[M-H2O+H]+', '[2M+H]+', '[2M+Na]+',
)
AUTO_ADDUCTS_NEG = (
    '[M-H]-', '[M+Cl]-', '[M+CH2O2-H]-', '[M+C2H4O2-H]-', '[2M-H]-',
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    """Verify schema, finite parameters, and SHA-256 checksums for a format_version=4 bundle."""
    if not isinstance(bundle, dict) or bundle.get('format_version') != 4:
        raise ValueError('msbuddy 2.0 bundle must have format_version == 4.')
    if bundle.get('cand_feat_dim') != CAND_FEAT_DIM or bundle.get('pool_feat_dim') != POOL_FEAT_DIM:
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

    center = np.asarray(bundle.get('feat_center'), dtype=np.float32)
    scale = np.asarray(bundle.get('feat_scale'), dtype=np.float32)
    if center.shape != (CAND_FEAT_DIM,) or scale.shape != (CAND_FEAT_DIM,):
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
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
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
                ang = (two_pi / wl) * mzs
                peak_feats[:n_valid, col] = np.sin(ang).astype(np.float32); col += 1
                peak_feats[:n_valid, col] = np.cos(ang).astype(np.float32); col += 1
            for wl in PEAK_WAVELENGTHS:
                ang = (two_pi / wl) * losses
                peak_feats[:n_valid, col] = np.sin(ang).astype(np.float32); col += 1
                peak_feats[:n_valid, col] = np.cos(ang).astype(np.float32); col += 1

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
):
    """Run Stage-1 + Stage-2 for one (MetaFeature, Adduct) hypothesis."""
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
    # Express mass error in ppm for universal scoring even when ppm=False
    err_ppm_all = c_err.astype(np.float64) if ppm else (c_err.astype(np.float64) / max(pmz, 1.0) * 1e6)
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

    ms2_f, best_frags, best_found = compute_ms2_evidence_fast(
        sub_arr, sub_ch, peak_mzs64, peak_ints64, n_v, int(adduct.charge),
        ppm, tol2,
        store.basic_mass, store.basic_idx, store.halogen_mass, store.halogen_idx,
        store.common_loss_keys, store.common_frag_db,
    )

    # Assemble 136-dim candidate features matching train_msbuddy2_suite.py identically
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
    cf[:, col] = sub_err_ppm / max(tol1_norm, 1.0); col += 1
    cf[:, col] = abs_ppm / max(tol1_norm, 1.0); col += 1
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

    cf[:, col:col + 24] = np.nan_to_num(ms2_f, nan=0.0, posinf=0.0, neginf=0.0); col += 24

    min_ppm = float(abs_ppm.min())
    max_ll = float(sub_sum_ll.max())
    max_exp = float(ms2_f[:, 1].max())
    max_dbf = float(ms2_f[:, 7].max())
    max_sub = float(ms2_f[:, 14].max())
    max_bu = float(sub_int.max())
    rk_ll = np.argsort(np.argsort(-sub_sum_ll, kind='stable')).astype(np.float32) / max(K, 1)
    rk_exp = np.argsort(np.argsort(-ms2_f[:, 1], kind='stable')).astype(np.float32) / max(K, 1)
    cf[:, col] = (abs_ppm - min_ppm) / max(tol1_norm, 1.0); col += 1
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

    cf_full = np.clip(np.concatenate([cf, rel12, ms1_12], axis=1), -1e4, 1e4).astype(np.float32)
    sorted_ll = np.sort(sub_sum_ll)[::-1]
    gap_ll = float(sorted_ll[0] - sorted_ll[1]) if K > 1 else 0.0
    pool_f = np.array([
        math.log1p(n_gen) / 10.0,
        math.log1p(K) / 5.0,
        min_ppm / max(tol1_norm, 1.0),
        np.clip(max_ll, -60.0, 20.0) / 10.0,
        np.clip(gap_ll, 0.0, 30.0) / 10.0,
        max_exp,
        max_dbf,
        float(sub_db.mean()),
        1.0 if has_ms1 else 0.0,
        max_iso,
    ], dtype=np.float32)

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
    }


def _build_candidate_formula(hyp: dict, k: int, mf, raw_idx_map: Optional[np.ndarray], prob: float,
                             norm_prob: float, fdr: float) -> CandidateFormula:
    adduct = hyp['adduct']
    arr = hyp['sub_arr'][k]
    ch_arr = hyp['sub_ch'][k]
    neutral_f = Formula._from_array(arr.copy(), 0, float(hyp['masses'][k]))
    charged_f = Formula._from_array(ch_arr.copy(), int(adduct.charge))

    # Build MS2Explanation mapped to raw MS2 indices
    found_mask = hyp['best_found'][k, :hyp['n_v']]
    ms2_exp = None
    if found_mask.any():
        exp_local = np.flatnonzero(found_mask)
        idx_arr = raw_idx_map[exp_local] if raw_idx_map is not None else exp_local
        frag_forms = [
            Formula._from_array(hyp['best_frags'][k, li].copy(), int(adduct.charge))
            for li in exp_local
        ]
        ms2_exp = MS2Explanation(idx_array=idx_arr.astype(np.intp), explanation_list=frag_forms)

    cf = CandidateFormula(
        formula=neutral_f,
        charged_formula=charged_f,
        mz_error=float(hyp['sub_err_raw'][k]),
        exp_ms2_sum_int=float(hyp['sub_int'][k]),
        ms2_raw_explanation=ms2_exp,
        db_existed=bool(hyp['sub_db'][k]),
    )
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

    # Stage 0: Batched CPU ONNX inference across all hypotheses
    n_hyp = len(hyp_specs)
    ep_all = np.zeros((n_hyp, 34), dtype=np.float32)
    pr_all = np.zeros((n_hyp, 3), dtype=np.float32)
    s0_sess = rt['stage0']
    for a in range(0, n_hyp, 64):
        b = min(n_hyp, a + 64)
        bs = b - a
        pf_b = np.empty((bs, 64, PEAK_FEAT_DIM), dtype=np.float32)
        pm_b = np.empty((bs, 64), dtype=np.float32)
        mk_b = np.empty((bs, 64), dtype=bool)
        qm_b = np.empty((bs, QUERY_META_60_DIM), dtype=np.float32)
        for j, (fi, _, qm) in enumerate(hyp_specs[a:b]):
            pf, pm, _, pmask, _, _, _, _ = peak_cache[fi]
            pf_b[j] = pf; pm_b[j] = pm; mk_b[j] = pmask; qm_b[j] = qm
        ep_o, pr_o = s0_sess.run(None, {
            'peak_feats': pf_b, 'peak_mzs': pm_b, 'peak_mask': mk_b, 'query_meta': qm_b,
        })
        ep_all[a:b] = ep_o
        pr_all[a:b] = pr_o

    # Stage 1 & 2: Compiled candidate generation + feature extraction per hypothesis
    valid_hyps = []
    valid_hyp_fi = []
    for hi, (fi, ion, _) in enumerate(hyp_specs):
        _, pm, pi, _, _, _, n_v, _ = peak_cache[fi]
        hyp = _prepare_single_hypothesis(
            batch_data[fi], ion, ep_all[hi], pr_all[hi],
            pm, pi, n_v, config, store, k_max, n_protect, DEFAULT_LL_W,
        )
        if hyp is not None:
            valid_hyps.append(hyp)
            valid_hyp_fi.append(fi)

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

    # Stage-3A ONNX in batches of 64
    for a in range(0, n_vh, 64):
        b = min(n_vh, a + 64)
        bs = b - a
        kb = max(h['K'] for h in valid_hyps[a:b])
        cf_b = np.zeros((bs, kb, CAND_FEAT_DIM), dtype=np.float32)
        cm_b = np.zeros((bs, kb), dtype=bool)
        pf_b = np.zeros((bs, POOL_FEAT_DIM), dtype=np.float32)
        for j, h in enumerate(valid_hyps[a:b]):
            k = h['K']
            cf_b[j, :k] = np.clip((h['cf_full'] - fc) / fs, -10.0, 10.0)
            cm_b[j, :k] = True
            pf_b[j] = h['pool_f']
        l_out = s3_sess.run(None, {'cand_feats': cf_b, 'cand_mask': cm_b, 'pool_feats': pf_b})[0]
        for j, h in enumerate(valid_hyps[a:b]):
            k = h['K']
            nn_logits_all[a + j, :k] = l_out[j, :k]
            nn_logits_all[a + j, max_k_batch] = l_out[j, kb]

    # Stage-3B LightGBM over concatenated valid candidates (1 fast vectorized call)
    lgb_rows = []
    offsets = [0]
    for h in valid_hyps:
        k = h['K']
        pf_rep = np.broadcast_to(h['pool_f'][None, :], (k, POOL_FEAT_DIM))
        lgb_rows.append(np.concatenate([h['cf_full'], pf_rep], axis=1))
        offsets.append(offsets[-1] + k)
    X_lgb = np.concatenate(lgb_rows, axis=0)
    lgb_preds = s3_lgb.predict(X_lgb, num_threads=1).astype(np.float32)

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
        flat_entries = []
        for h in hyps:
            for k in range(h['K']):
                flat_entries.append((h, k, float(h['log_odds'][k])))
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
