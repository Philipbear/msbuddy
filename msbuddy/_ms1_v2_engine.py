# ==============================================================================
# Copyright (C) 2024-2026 Shipei Xing <sxing@fudan.edu.cn>
#
# Licensed under the Apache License 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://github.com/Philipbear/msbuddy/blob/main/LICENSE
# ==============================================================================
"""Compact compiled CPU engine for MS1 isotope-only and precursor mass-only annotation.

Reuses the shared-memory ``FastFormulaDB`` and compiled ``nogil`` Numba kernels
(``generate_candidates_fast_numba`` and ``compute_ms1_features_numba``) from
``msbuddy._fast_db`` so MS1-only and precursor mass-only queries run fast on CPU
with rich 12-d analytical isotope features, pool-relative features, arbitrary
charge/tolerance/adduct support, opt-in ``adduct="auto"``, and calibrated
``[ABSENT]`` posterior probabilities—at a tiny (~1 MB) model footprint.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp

from msbuddy._fast_db import (
    ELEM_MASSES_F64,
    MS1_FEAT_DIM,
    compute_ms1_features_numba,
    generate_candidates_fast_numba,
    get_fast_store,
)
from msbuddy._v2_engine import (
    AUTO_ADDUCT_LOG_PRIORS,
    AUTO_ADDUCTS_NEG,
    AUTO_ADDUCTS_POS,
)
from msbuddy.base import Adduct, CandidateFormula, Formula

MS1_V2_FORMAT_VERSION = 1
MASS_FEAT_DIM = 79
ISO_FEAT_DIM = 99  # 79 mass+formula+adduct+pool + 12 analytical MS1 + 8 pool-relative MS1
CAL_CTX_DIM = 9
DEFAULT_MS1_SCORE_CAP = 500
DEFAULT_MS1_KEEP_K = 80

_EMPTY_F64 = np.empty(0, dtype=np.float64)
_EMPTY_LOSS = np.zeros(12, dtype=np.int16)


def _sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def build_ms1_v2_features(
    c_arr: np.ndarray,
    c_err_raw: np.ndarray,
    adduct: Adduct,
    precursor_mz: float,
    ppm: bool,
    ms1_tol: float,
    n_gen: int,
    ms1_f12: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact feature matrix for MS1-only (99-d) or mass-only (79-d) candidates.

    Shared verbatim between training and inference to guarantee zero feature skew.

    Returns
    -------
    X : (K, 99 or 79) float32 feature matrix
    masses : (K,) float64 neutral monoisotopic masses
    abs_ppm : (K,) float32 absolute precursor mass error in ppm
    """
    K = len(c_arr)
    c_f = c_arr.astype(np.float32)
    c64 = c_arr.astype(np.float64)
    masses = c64 @ ELEM_MASSES_F64
    masses_safe = np.maximum(masses, 1.0)
    mass_fracs = (c64 * ELEM_MASSES_F64[None, :]) / masses_safe[:, None]

    # Element counts: 0:C, 1:H, 2:Br, 3:Cl, 4:F, 5:I, 6:K, 7:N, 8:Na, 9:O, 10:P, 11:S
    C = c_f[:, 0]
    H = c_f[:, 1]
    Br = c_f[:, 2]
    Cl = c_f[:, 3]
    F = c_f[:, 4]
    I = c_f[:, 5]
    K_el = c_f[:, 6]
    N = c_f[:, 7]
    Na = c_f[:, 8]
    O = c_f[:, 9]
    P = c_f[:, 10]
    S = c_f[:, 11]

    hal = Br + Cl + F + I
    c_safe = np.maximum(C, 1.0)
    tot_atoms = np.maximum(c_f.sum(axis=1), 1.0)
    dbe = C + 1.0 - 0.5 * (H + hal + Na + K_el) + 0.5 * (N + P)
    dbe_pos = np.maximum(dbe, 0.0)
    chon_only = ((hal + Na + K_el + P + S) == 0.0).astype(np.float32)
    chonps_only = ((hal + Na + K_el) == 0.0).astype(np.float32)
    n_halo_types = (
        (Br > 0).astype(np.float32) + (Cl > 0).astype(np.float32)
        + (F > 0).astype(np.float32) + (I > 0).astype(np.float32)
    )
    hal_h = np.where(H > 0.0, hal / np.maximum(H, 1.0), 0.0)
    o_p = np.where(P > 0.0, O / np.maximum(P, 1.0) / 3.0, 1.0)
    hetero_r = c_f[:, 2:].sum(axis=1) / tot_atoms
    val_sum = 6.0 * S + 5.0 * P + 4.0 * C + 3.0 * N + 2.0 * O + H + hal + Na + K_el
    senior_excess = (val_sum - (2.0 * tot_atoms - 1.0)) / np.maximum(tot_atoms, 1.0)

    # Precursor mass error in signed ppm: c_err_raw = (theo - obs), negate -> (obs - theo)
    pmz_safe = max(float(precursor_mz), 1.0)
    tol1_ppm = float(ms1_tol) if ppm else max(float(ms1_tol) / pmz_safe * 1e6, 0.5)
    tol1_div = max(tol1_ppm, 0.5)
    signed_ppm = (-c_err_raw.astype(np.float64)) if ppm else (-c_err_raw.astype(np.float64) / pmz_safe * 1e6)
    abs_ppm = np.abs(signed_ppm)
    mass_gauss = np.exp(-0.5 * np.minimum((abs_ppm / max(tol1_div / 3.0, 0.25)) ** 2, 16.0))
    min_ppm = float(abs_ppm.min()) if K > 0 else 0.0
    mu_ppm = float(abs_ppm.mean()) if K > 0 else 0.0
    sd_ppm = math.sqrt(float(((abs_ppm - mu_ppm) ** 2).mean()) + 1e-4) if K > 0 else 1.0
    rk_ppm = np.argsort(np.argsort(abs_ppm, kind='stable')).astype(np.float32) / max(K, 1)

    if K > 1:
        p2 = np.partition(abs_ppm, 1)
        b1, b2 = float(p2[0]), float(p2[1])
        opp_ppm = np.where(abs_ppm <= b1 + 1e-9, b2, b1)
    else:
        opp_ppm = abs_ppm
    ppm_opp_gap = np.clip((opp_ppm - abs_ppm) / tol1_div, -5.0, 5.0)

    # Adduct metadata (9 dims)
    net = adduct.net_formula.array
    pos_mode = 1.0 if adduct.charge > 0 else 0.0
    abs_q = float(abs(adduct.charge))
    mult_m = float(adduct.m)
    net_mass_norm = float(adduct.net_formula.mass) / 50.0
    has_na = 1.0 if net[8] > 0 else 0.0
    has_k = 1.0 if net[6] > 0 else 0.0
    has_n = 1.0 if net[7] > 0 else 0.0
    has_cl = 1.0 if net[3] > 0 else 0.0
    has_loss = 1.0 if adduct.loss_formula is not None else 0.0

    has_ms1 = ms1_f12 is not None
    dim = ISO_FEAT_DIM if has_ms1 else MASS_FEAT_DIM
    X = np.zeros((K, dim), dtype=np.float32)
    col = 0

    # 1. Counts & log counts & mass fractions (36 dims)
    X[:, col:col + 12] = c_f; col += 12
    X[:, col:col + 12] = np.log1p(c_f); col += 12
    X[:, col:col + 12] = mass_fracs.astype(np.float32); col += 12

    # 2. DBE, ratios, chemical plausibility, mass/defect (24 dims)
    X[:, col] = dbe / 20.0; col += 1
    X[:, col] = dbe / c_safe; col += 1
    X[:, col] = np.sqrt(dbe_pos / masses_safe.astype(np.float32)); col += 1
    X[:, col] = dbe / np.power(masses_safe.astype(np.float32) / 100.0, 2.0 / 3.0); col += 1
    X[:, col] = H / c_safe; col += 1
    X[:, col] = N / c_safe; col += 1
    X[:, col] = O / c_safe; col += 1
    X[:, col] = P / c_safe; col += 1
    X[:, col] = S / c_safe; col += 1
    X[:, col] = hal / c_safe; col += 1
    X[:, col] = F / c_safe; col += 1
    X[:, col] = Cl / c_safe; col += 1
    X[:, col] = Br / c_safe; col += 1
    X[:, col] = I / c_safe; col += 1
    X[:, col] = hetero_r; col += 1
    X[:, col] = chon_only; col += 1
    X[:, col] = chonps_only; col += 1
    X[:, col] = n_halo_types / 4.0; col += 1
    X[:, col] = np.clip(hal_h, 0.0, 10.0); col += 1
    X[:, col] = np.clip(o_p, 0.0, 10.0); col += 1
    X[:, col] = np.log1p(masses_safe / 500.0).astype(np.float32); col += 1
    X[:, col] = (masses - np.rint(masses)).astype(np.float32); col += 1
    X[:, col] = tot_atoms / 100.0; col += 1
    X[:, col] = np.clip(senior_excess, -5.0, 10.0); col += 1

    # 3. Adduct & ion metadata (9 dims)
    X[:, col] = pos_mode; col += 1
    X[:, col] = abs_q; col += 1
    X[:, col] = mult_m; col += 1
    X[:, col] = net_mass_norm; col += 1
    X[:, col] = has_na; col += 1
    X[:, col] = has_k; col += 1
    X[:, col] = has_n; col += 1
    X[:, col] = has_cl; col += 1
    X[:, col] = has_loss; col += 1

    # 4. Precursor mass fit & pool-relative context (10 dims) -> 36 + 24 + 9 + 10 = 79
    X[:, col] = np.clip(signed_ppm / tol1_div, -5.0, 5.0).astype(np.float32); col += 1
    X[:, col] = np.clip(abs_ppm / tol1_div, 0.0, 5.0).astype(np.float32); col += 1
    X[:, col] = np.clip(abs_ppm, 0.0, 200.0).astype(np.float32) / 10.0; col += 1
    X[:, col] = mass_gauss.astype(np.float32); col += 1
    X[:, col] = min(tol1_ppm, 200.0) / 10.0; col += 1
    X[:, col] = math.log1p(max(n_gen, 0)) / 8.0; col += 1
    X[:, col] = np.clip((abs_ppm - min_ppm) / tol1_div, 0.0, 5.0).astype(np.float32); col += 1
    X[:, col] = np.clip((abs_ppm - mu_ppm) / sd_ppm, -5.0, 5.0).astype(np.float32); col += 1
    X[:, col] = rk_ppm; col += 1
    X[:, col] = ppm_opp_gap.astype(np.float32); col += 1

    # 5. Analytical 12-d MS1 isotope features + 8 pool-relative MS1 features (20 dims -> 99)
    if has_ms1:
        X[:, col:col + MS1_FEAT_DIM] = ms1_f12; col += MS1_FEAT_DIM
        sim = ms1_f12[:, 2].astype(np.float64)
        iso_ll = ms1_f12[:, 8].astype(np.float64)
        if K > 1:
            ps = np.partition(-sim, 1)
            s1, s2 = -float(ps[0]), -float(ps[1])
            opp_sim = np.where(sim >= s1 - 1e-6, s2, s1)
            pl = np.partition(-iso_ll, 1)
            l1, l2 = -float(pl[0]), -float(pl[1])
            opp_ll = np.where(iso_ll >= l1 - 1e-6, l2, l1)
        else:
            opp_sim = sim
            opp_ll = iso_ll
        sim_mu = float(sim.mean()) if K > 0 else 0.0
        sim_sd = math.sqrt(float(((sim - sim_mu) ** 2).mean()) + 1e-4) if K > 0 else 1.0
        ll_mu = float(iso_ll.mean()) if K > 0 else 0.0
        ll_sd = math.sqrt(float(((iso_ll - ll_mu) ** 2).mean()) + 1e-4) if K > 0 else 1.0
        rk_sim = np.argsort(np.argsort(-sim, kind='stable')).astype(np.float32) / max(K, 1)
        max_ll = float(iso_ll.max()) if K > 0 else 0.0
        joint_phys = mass_gauss + 3.0 * sim + 1.5 * iso_ll
        max_jp = float(joint_phys.max()) if K > 0 else 0.0

        X[:, col] = np.clip(sim - opp_sim, -2.0, 2.0).astype(np.float32); col += 1
        X[:, col] = np.clip((sim - sim_mu) / sim_sd, -5.0, 5.0).astype(np.float32); col += 1
        X[:, col] = rk_sim; col += 1
        X[:, col] = np.clip(iso_ll - max_ll, -20.0, 0.0).astype(np.float32); col += 1
        X[:, col] = np.clip(iso_ll - opp_ll, -10.0, 10.0).astype(np.float32); col += 1
        X[:, col] = np.clip((iso_ll - ll_mu) / ll_sd, -5.0, 5.0).astype(np.float32); col += 1
        X[:, col] = np.clip(joint_phys, -30.0, 10.0).astype(np.float32); col += 1
        X[:, col] = np.clip(joint_phys - max_jp, -30.0, 0.0).astype(np.float32); col += 1

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, masses, abs_ppm.astype(np.float32)


def build_ms1_cal_context(
    scores: np.ndarray,
    abs_ppm: np.ndarray,
    tol1_ppm: float,
    n_gen: int,
    precursor_mz: float,
    adduct_m: int,
    has_ms1: bool,
    n_iso: int,
    ms1_sim: Optional[np.ndarray],
) -> np.ndarray:
    """9-d query calibration context for temperature + [ABSENT] logit scaling."""
    order = np.argsort(-scores, kind='stable')
    top = int(order[0])
    s_top = float(scores[top])
    s_gap = float(scores[top] - scores[order[1]]) if len(scores) > 1 else 0.0
    tol1_div = max(float(tol1_ppm), 0.5)
    top_rel_ppm = min(float(abs_ppm[top]) / tol1_div, 5.0)
    top_sim = float(ms1_sim[top]) if (has_ms1 and ms1_sim is not None) else 0.0
    sim_gap = (
        float(ms1_sim[top] - ms1_sim[order[1]])
        if (has_ms1 and ms1_sim is not None and len(scores) > 1)
        else 0.0
    )
    return np.array([
        s_top,
        s_gap,
        math.log1p(max(n_gen, 0)),
        top_rel_ppm,
        min(float(precursor_mz), 2000.0) / 1000.0,
        float(adduct_m),
        float(n_iso) / 4.0 if has_ms1 else 0.0,
        top_sim,
        sim_gap,
    ], dtype=np.float64)


def calibrated_ms1_log_odds(scores: np.ndarray, context: np.ndarray, cal: dict) -> np.ndarray:
    """Convert raw ranker scores to calibrated log-odds relative to [ABSENT] anchor (0.0)."""
    center = np.asarray(cal['context_center'], dtype=np.float64)
    scale = np.asarray(cal['context_scale'], dtype=np.float64)
    null_coef = np.asarray(cal['null_coefficients'], dtype=np.float64)
    temp_coef = np.asarray(cal['temperature_coefficients'], dtype=np.float64)
    clip = float(cal.get('context_clip', 6.0))
    z = np.clip((context - center) / scale, -clip, clip)
    inv_temp = float(np.exp(np.clip(temp_coef[0] + np.dot(temp_coef[1:], z), math.log(0.05), math.log(15.0))))
    null_logit = float(np.clip(null_coef[0] + np.dot(null_coef[1:], z), -20.0, 20.0))
    shifted = (np.asarray(scores, dtype=np.float64) - float(np.max(scores))) * inv_temp
    return shifted - null_logit


def validate_ms1_v2_spec(spec: dict) -> None:
    """Validate checksums, feature dimensions, and calibration parameters for ms1_v2."""
    if not isinstance(spec, dict) or spec.get('format_version') != MS1_V2_FORMAT_VERSION:
        raise ValueError('Unsupported ms1_v2 model specification version.')
    if spec.get('isotope_feat_dim') != ISO_FEAT_DIM or spec.get('mass_feat_dim') != MASS_FEAT_DIM:
        raise ValueError('ms1_v2 feature dimensions do not match engine schema.')
    for txt_key, sha_key in (
        ('isotope_lgb_txt', 'isotope_lgb_sha256'),
        ('mass_lgb_txt', 'mass_lgb_sha256'),
    ):
        txt = spec.get(txt_key)
        sha = spec.get(sha_key)
        if not isinstance(txt, str) or not isinstance(sha, str) or _sha256_str(txt) != sha:
            raise ValueError(f'ms1_v2 checksum verification failed for {txt_key}.')
    for cal_key in ('isotope_calibration', 'mass_calibration'):
        cal = spec.get(cal_key)
        if not isinstance(cal, dict):
            raise ValueError(f'Missing {cal_key} in ms1_v2 specification.')
        for arr_key, exp_len in (
            ('context_center', CAL_CTX_DIM),
            ('context_scale', CAL_CTX_DIM),
            ('null_coefficients', CAL_CTX_DIM + 1),
            ('temperature_coefficients', CAL_CTX_DIM + 1),
        ):
            arr = np.asarray(cal.get(arr_key), dtype=np.float64)
            if arr.shape != (exp_len,) or not np.all(np.isfinite(arr)):
                raise ValueError(f'Invalid {arr_key} in ms1_v2 {cal_key}.')
            if arr_key == 'context_scale' and np.any(arr <= 0.0):
                raise ValueError(f'Non-positive context_scale in ms1_v2 {cal_key}.')


def _get_ms1_v2_runtime(gd: dict) -> dict:
    cache = gd.setdefault('model_ms1_v2_cache', {})
    if 'isotope' not in cache:
        from lightgbm import Booster
        spec = gd['model_ms1_v2']
        cache['isotope'] = Booster(model_str=spec['isotope_lgb_txt'])
        cache['mass'] = Booster(model_str=spec['mass_lgb_txt'])
        cache['isotope_cal'] = spec['isotope_calibration']
        cache['mass_cal'] = spec['mass_calibration']
    return cache


def _prepare_ms1_hypothesis(mf, adduct: Adduct, config, store, score_cap: int):
    pmz = float(mf.mz)
    ppm = bool(config.ppm)
    tol1 = float(config.ms1_tol)
    tol2 = float(config.ms2_tol)

    orig_adduct = mf.adduct
    mf.adduct = adduct
    try:
        mf.data_preprocess(
            ppm, tol1, tol2,
            config.isotope_bin_mztol, config.max_isotope_cnt,
            config.rel_int_denoise_cutoff, config.top_n_per_50_da,
        )
        proc_ms1 = mf.ms1_processed
    finally:
        mf.adduct = orig_adduct

    has_loss = adduct.loss_formula is not None
    loss_arr = adduct.loss_formula.array.astype(np.int16) if has_loss else _EMPTY_LOSS
    c_arr, _, _, c_err = generate_candidates_fast_numba(
        pmz, _EMPTY_F64, _EMPTY_F64, ppm, tol1, tol2,
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
        obs_ints = np.ascontiguousarray(proc_ms1.int_array[:n_iso], dtype=np.float64)
        obs_mzs = np.ascontiguousarray(proc_ms1.mz_array[:n_iso], dtype=np.float64)
        all_ch = np.ascontiguousarray(
            np.maximum(c_arr * int(adduct.m) + adduct.net_formula.array.astype(np.int16), 0),
            dtype=np.int16,
        )
        ms1_f12 = compute_ms1_features_numba(all_ch, obs_ints, obs_mzs, n_iso, abs(int(adduct.charge)))
    else:
        n_iso = 0
        ms1_f12 = None

    # Cap extreme pools (>score_cap) by fast physics score before LightGBM
    if n_gen > score_cap:
        tol1_ppm = tol1 if ppm else max(tol1 / max(pmz, 1.0) * 1e6, 0.5)
        abs_ppm = np.abs(c_err.astype(np.float64)) if ppm else np.abs(c_err.astype(np.float64)) / max(pmz, 1.0) * 1e6
        phys = -0.5 * ((abs_ppm / max(tol1_ppm / 3.0, 0.25)) ** 2)
        if ms1_f12 is not None:
            phys = phys + 3.5 * ms1_f12[:, 2].astype(np.float64) + 1.5 * ms1_f12[:, 8].astype(np.float64)
        keep = np.argpartition(-phys, score_cap - 1)[:score_cap]
        c_arr = np.ascontiguousarray(c_arr[keep], dtype=np.int16)
        c_err = c_err[keep]
        if ms1_f12 is not None:
            ms1_f12 = np.ascontiguousarray(ms1_f12[keep], dtype=np.float32)

    X, masses, abs_ppm = build_ms1_v2_features(
        c_arr, c_err, adduct, pmz, ppm, tol1, n_gen, ms1_f12=ms1_f12,
    )
    sub_ch = np.ascontiguousarray(
        np.maximum(c_arr * int(adduct.m) + adduct.net_formula.array.astype(np.int16), 0),
        dtype=np.int16,
    )
    tol1_ppm = tol1 if ppm else max(tol1 / max(pmz, 1.0) * 1e6, 0.5)
    return {
        'adduct': adduct,
        'K': len(c_arr),
        'n_gen': n_gen,
        'has_ms1': has_ms1,
        'n_iso': n_iso,
        'tol1_ppm': tol1_ppm,
        'pmz': pmz,
        'c_arr': c_arr,
        'sub_ch': sub_ch,
        'c_err': c_err,
        'masses': masses,
        'abs_ppm': abs_ppm,
        'ms1_sim': (ms1_f12[:, 2] if ms1_f12 is not None else None),
        'X': X,
    }


def annotate_batch_ms1_v2(batch_data: Sequence, config, gd: dict) -> None:
    """Annotate a batch of MS1-only and/or mass-only MetaFeature queries via compiled ms1_v2."""
    if not batch_data:
        return
    rt = _get_ms1_v2_runtime(gd)
    store = get_fast_store(gd)
    k_keep = int(config.max_candidates) if config.max_candidates is not None else DEFAULT_MS1_KEEP_K
    score_cap = max(DEFAULT_MS1_SCORE_CAP, k_keep)

    hyp_specs: List[Tuple[int, Adduct]] = []
    for fi, mf in enumerate(batch_data):
        if getattr(mf, 'adduct_auto', False) and abs(mf.adduct.charge) == 1:
            adduct_strs = AUTO_ADDUCTS_POS if mf.adduct.charge > 0 else AUTO_ADDUCTS_NEG
            adducts = [Adduct(a, a.endswith('+'), report_invalid=True) for a in adduct_strs]
        else:
            adducts = [mf.adduct]
        for ion in adducts:
            hyp_specs.append((fi, ion))

    if getattr(config, 'parallel', False):
        req_cpu = int(getattr(config, 'n_cpu', -1))
        n_threads = (os.cpu_count() or 1) if req_cpu <= 0 else req_cpu
    else:
        n_threads = 1
    n_threads = max(1, min(n_threads, max(1, len(hyp_specs))))

    def _run_prep(idx: int):
        fi, ion = hyp_specs[idx]
        return _prepare_ms1_hypothesis(batch_data[fi], ion, config, store, score_cap)

    if n_threads > 1 and len(hyp_specs) > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            prep_res = list(pool.map(_run_prep, range(len(hyp_specs))))
    else:
        prep_res = [_run_prep(i) for i in range(len(hyp_specs))]

    # Batch LightGBM prediction separately for MS1-only (99-d) and mass-only (79-d)
    iso_hyps: List[Tuple[int, dict]] = []
    mass_hyps: List[Tuple[int, dict]] = []
    for (fi, _), hyp in zip(hyp_specs, prep_res):
        if hyp is None:
            continue
        if hyp['has_ms1']:
            iso_hyps.append((fi, hyp))
        else:
            mass_hyps.append((fi, hyp))

    by_feature: Dict[int, List[dict]] = {fi: [] for fi in range(len(batch_data))}
    for group_hyps, booster_key, cal_key in (
        (iso_hyps, 'isotope', 'isotope_cal'),
        (mass_hyps, 'mass', 'mass_cal'),
    ):
        if not group_hyps:
            continue
        X_cat = np.concatenate([h['X'] for _, h in group_hyps], axis=0)
        preds = rt[booster_key].predict(X_cat, num_threads=n_threads).astype(np.float64)
        cal = rt[cal_key]
        off = 0
        for fi, h in group_hyps:
            k = h['K']
            sc = preds[off:off + k]
            off += k
            ctx = build_ms1_cal_context(
                sc, h['abs_ppm'], h['tol1_ppm'], h['n_gen'], h['pmz'],
                int(h['adduct'].m), h['has_ms1'], h['n_iso'], h['ms1_sim'],
            )
            h['log_odds'] = calibrated_ms1_log_odds(sc, ctx, cal)
            by_feature[fi].append(h)

    for fi, mf in enumerate(batch_data):
        hyps = by_feature[fi]
        if not hyps:
            mf.candidate_formula_list = []
            continue
        is_auto = getattr(mf, 'adduct_auto', False) and len(hyps) > 1
        flat = []
        for h in hyps:
            prior_shift = AUTO_ADDUCT_LOG_PRIORS.get(h['adduct'].string, -1.5) if is_auto else 0.0
            for k in range(h['K']):
                flat.append((h, k, float(h['log_odds'][k]) + prior_shift))
        log_odds_arr = np.array([e[2] for e in flat], dtype=np.float64)
        all_s = np.append(log_odds_arr, 0.0)
        probs = np.exp(all_s - logsumexp(all_s))[:-1]

        order = np.argsort(-probs, kind='stable')[:k_keep]
        cand_list = []
        cum_p = 0.0
        for rank_i, idx in enumerate(order):
            h, k, _ = flat[int(idx)]
            p = float(probs[int(idx)])
            cum_p += p
            fdr = max(0.0, min(1.0, 1.0 - cum_p / float(rank_i + 1)))
            adduct = h['adduct']
            neutral_f = Formula._from_array(h['c_arr'][k].copy(), 0, float(h['masses'][k]))
            charged_f = Formula._from_array(h['sub_ch'][k].copy(), int(adduct.charge))
            cf = CandidateFormula(
                formula=neutral_f,
                charged_formula=charged_f,
                mz_error=float(h['c_err'][k]),
                exp_ms2_sum_int=0.0,
                ms2_raw_explanation=None,
                db_existed=True,
            )
            cf.adduct = adduct
            if h['ms1_sim'] is not None:
                cf.ms1_isotope_similarity = float(h['ms1_sim'][k])
            cf.estimated_prob = p
            cf.normed_estimated_prob = p
            cf.estimated_fdr = fdr
            cand_list.append(cf)

        mf.candidate_formula_list = cand_list
        if is_auto and cand_list:
            mf.adduct = cand_list[0].adduct
            mf.data_preprocess(
                config.ppm, config.ms1_tol, config.ms2_tol,
                config.isotope_bin_mztol, config.max_isotope_cnt,
                config.rel_int_denoise_cutoff, config.top_n_per_50_da,
            )
