# ==============================================================================
# Copyright (C) 2024-2026 Shipei Xing <sxing@fudan.edu.cn>
#
# Licensed under the Apache License 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://github.com/Philipbear/msbuddy/blob/main/LICENSE
# ==============================================================================
"""Compiled Numba database search, bottom-up MS2 stitching, subformula evidence,
analytical MS1 isotope simulation, and exact-mass chemistry featurization for msbuddy 2.0.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from numba import njit

from msbuddy._subformula import closest_subformulas

# Element order matching msbuddy.utils.SUPPORTED_ELEMENTS:
# 0:C, 1:H, 2:Br, 3:Cl, 4:F, 5:I, 6:K, 7:N, 8:Na, 9:O, 10:P, 11:S
ELEM_MASSES_F64 = np.array([
    12.0, 1.007825, 78.918336, 34.968853, 18.998403, 126.904473,
    38.963707, 14.003074, 22.989769, 15.994915, 30.973762, 31.972071,
], dtype=np.float64)

MASS_H = 1.007825
MASS_E = 0.00054858
MASS_PROTON = 1.007276
NA_H_DELTA = 22.989769 - 1.007825
K_H_DELTA = 38.963707 - 1.007825

UNIVERSAL_DELTAS = np.array([
    18.010565,  # H2O
    17.026549,  # NH3
    27.994915,  # CO
    43.989830,  # CO2
    14.015650,  # CH2
    28.031300,  # C2H4
    20.006228,  # HF
    35.976678,  # HCl
    79.926161,  # HBr
    127.912297, # HI
    49.996806,  # CF2
    63.961901,  # SO2
    79.956816,  # SO3
    79.966331,  # HPO3
], dtype=np.float64)

HALIDE_NEUTRAL_MASSES = np.array([34.968853, 78.918336, 126.904473], dtype=np.float64)
OXYANION_NEUTRAL_MASSES = np.array([78.958507, 96.969072, 79.956816, 96.959553], dtype=np.float64)

QUERY_META_60_DIM = 60
MS1_FEAT_DIM = 12


@njit(cache=True)
def _pack_formula_hash(arr: np.ndarray) -> Tuple[np.uint64, np.uint64]:
    """Bijective 128-bit key for any 12-element formula (counts 0..1023)."""
    lo = np.uint64(0)
    hi = np.uint64(0)
    for i in range(6):
        lo |= (np.uint64(arr[i]) & np.uint64(0x3FF)) << np.uint64(10 * i)
    for i in range(6):
        hi |= (np.uint64(arr[6 + i]) & np.uint64(0x3FF)) << np.uint64(10 * i)
    return lo, hi


@njit(cache=True)
def pack_common_key(arr: np.ndarray) -> np.uint64:
    """Exact 64-bit bijective packing for small common neutral losses."""
    k = np.uint64(0)
    k |= np.uint64(arr[0] & 0x7F)
    k |= np.uint64(arr[1] & 0xFF) << np.uint64(7)
    k |= np.uint64(arr[2] & 0x1F) << np.uint64(15)
    k |= np.uint64(arr[3] & 0x1F) << np.uint64(20)
    k |= np.uint64(arr[4] & 0x3F) << np.uint64(25)
    k |= np.uint64(arr[5] & 0x1F) << np.uint64(31)
    k |= np.uint64(arr[6] & 0x0F) << np.uint64(36)
    k |= np.uint64(arr[7] & 0x3F) << np.uint64(40)
    k |= np.uint64(arr[8] & 0x0F) << np.uint64(46)
    k |= np.uint64(arr[9] & 0x3F) << np.uint64(50)
    k |= np.uint64(arr[10] & 0x0F) << np.uint64(56)
    k |= np.uint64(arr[11] & 0x0F) << np.uint64(60)
    return k


@njit(cache=True)
def common_nl_lookup_fast(arr: np.ndarray, sorted_keys: np.ndarray) -> bool:
    """O(log n) lookup for common neutral loss in sorted packed uint64 keys."""
    if len(sorted_keys) == 0:
        return False
    if (arr[0] < 0 or arr[0] > 127 or arr[1] < 0 or arr[1] > 255
            or arr[2] < 0 or arr[2] > 31 or arr[3] < 0 or arr[3] > 31
            or arr[4] < 0 or arr[4] > 63 or arr[5] < 0 or arr[5] > 31
            or arr[6] < 0 or arr[6] > 15 or arr[7] < 0 or arr[7] > 63
            or arr[8] < 0 or arr[8] > 15 or arr[9] < 0 or arr[9] > 63
            or arr[10] < 0 or arr[10] > 15 or arr[11] < 0 or arr[11] > 15):
        return False
    key = pack_common_key(arr)
    lo = 0
    hi = len(sorted_keys)
    while lo < hi:
        mid = (lo + hi) >> 1
        v = sorted_keys[mid]
        if v == key:
            return True
        if v < key:
            lo = mid + 1
        else:
            hi = mid
    return False


@njit(cache=True)
def _dbe_12(arr: np.ndarray) -> float:
    return (
        float(arr[0]) + 1.0
        - 0.5 * float(arr[1] + arr[4] + arr[3] + arr[2] + arr[5] + arr[8] + arr[6])
        + 0.5 * float(arr[7] + arr[10])
    )


@njit(cache=True)
def _formula_valid_precursor(arr: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                             loss_arr: np.ndarray, has_loss: bool) -> bool:
    atoms = 0
    for i in range(12):
        v = int(arr[i])
        if v < int(lower[i]) or v > int(upper[i]):
            return False
        if has_loss and v < int(loss_arr[i]):
            return False
        atoms += v
    if atoms <= 0:
        return False
    if arr[10] > 0 and arr[9] < 2 * arr[10]:
        return False
    dbe = _dbe_12(arr)
    if dbe < -0.5 or (dbe % 1.0) != 0.0:
        return False
    val = (
        6 * int(arr[11]) + 5 * int(arr[10]) + 4 * int(arr[0]) + 3 * int(arr[7])
        + 2 * int(arr[9]) + int(arr[1]) + int(arr[4]) + int(arr[3])
        + int(arr[2]) + int(arr[5]) + int(arr[8]) + int(arr[6])
    )
    return val >= 2 * (atoms - 1)


@njit(cache=True)
def _binsearch_window(masses: np.ndarray, index: np.ndarray, target: float, tol: float) -> Tuple[int, int]:
    """Return exact [lo, hi) row slice in sorted masses matching |m - target| <= tol."""
    n = len(masses)
    if n == 0 or target + tol < 0.0:
        return 0, 0
    s_bin = int(math.floor((target - tol) * 10.0))
    e_bin = int(math.floor((target + tol) * 10.0)) + 1
    n_idx = len(index)
    if n_idx == 0:
        lo_bound = 0
        hi_bound = n
    else:
        lo_bound = 0 if s_bin <= 0 else int(index[min(s_bin, n_idx - 1)])
        if e_bin <= 0:
            return 0, 0
        hi_bound = n if e_bin >= n_idx else int(index[e_bin])
    if lo_bound >= hi_bound:
        return 0, 0

    t_lo = target - tol
    t_hi = target + tol
    l = lo_bound
    r = hi_bound
    while l < r:
        mid = (l + r) >> 1
        if masses[mid] < t_lo:
            l = mid + 1
        else:
            r = mid
    start = l

    l = start
    r = hi_bound
    while l < r:
        mid = (l + r) >> 1
        if masses[mid] <= t_hi:
            l = mid + 1
        else:
            r = mid
    return start, l


@njit(cache=True)
def mass_exists_fast(masses: np.ndarray, index: np.ndarray, target: float, tol: float) -> bool:
    """Bit-exact mass-only existence check without decoding formula rows."""
    lo, hi = _binsearch_window(masses, index, target, tol)
    return hi > lo


@njit(cache=True)
def _query_fragnl_arrays(
    mass: float, fragment: bool, pos_mode: bool, na_contain: bool, k_contain: bool,
    mz_tol: float, ppm: bool, db_mode: int,
    b_mass: np.ndarray, b_form: np.ndarray, b_idx: np.ndarray,
    h_mass: np.ndarray, h_form: np.ndarray, h_idx: np.ndarray,
):
    """Exact compiled equivalent of msbuddy.query.query_fragnl_mass returning sorted arrays."""
    mass_tol = mass * mz_tol / 1e6 if ppm else mz_tol
    ion_mode_int = 1 if pos_mode else -1

    win_db = np.empty(32, dtype=np.int8)
    win_lo = np.empty(32, dtype=np.int64)
    win_hi = np.empty(32, dtype=np.int64)
    win_rad = np.empty(32, dtype=np.int8)
    win_conv = np.empty(32, dtype=np.int8)
    n_win = 0
    total = 0

    n_db = 2 if db_mode > 0 else 1
    for conv_idx in range(3):
        if conv_idx == 1 and not na_contain:
            continue
        if conv_idx == 2 and not k_contain:
            continue
        cd = 0.0 if conv_idx == 0 else (NA_H_DELTA if conv_idx == 1 else K_H_DELTA)
        for rad in range(2):
            if fragment:
                if rad == 0:
                    tm = mass - cd - MASS_PROTON if pos_mode else mass - cd + MASS_PROTON
                else:
                    tm = mass - cd + MASS_E if pos_mode else mass - cd - MASS_E
            else:
                if rad == 0:
                    tm = mass - cd
                else:
                    tm = mass - cd - MASS_H if pos_mode else mass - cd + MASS_H
            for dbi in range(n_db):
                lo, hi = (_binsearch_window(b_mass, b_idx, tm, mass_tol)
                          if dbi == 0 else _binsearch_window(h_mass, h_idx, tm, mass_tol))
                if hi > lo:
                    win_db[n_win] = dbi
                    win_lo[n_win] = lo
                    win_hi[n_win] = hi
                    win_rad[n_win] = rad
                    win_conv[n_win] = conv_idx
                    n_win += 1
                    total += (hi - lo)

    out_arr = np.empty((total, 12), dtype=np.int16)
    out_mass = np.empty(total, dtype=np.float64)
    out_dbe = np.empty(total, dtype=np.float32)
    if total == 0:
        return out_arr, out_mass, out_dbe

    pos = 0
    for wi in range(n_win):
        dbi = win_db[wi]
        lo = win_lo[wi]
        hi = win_hi[wi]
        rad = win_rad[wi]
        ct = win_conv[wi]
        src_m = b_mass if dbi == 0 else h_mass
        src_f = b_form if dbi == 0 else h_form
        for r in range(lo, hi):
            for e in range(12):
                out_arr[pos, e] = np.int16(src_f[r, e])
            m0 = float(src_m[r])
            if fragment:
                if ct == 1:
                    out_arr[pos, 8] += 1
                    if rad == 1:
                        out_arr[pos, 1] -= 1
                        out_mass[pos] = m0 - ion_mode_int * MASS_E + NA_H_DELTA
                    else:
                        out_arr[pos, 1] = out_arr[pos, 1] - 1 + ion_mode_int
                        out_mass[pos] = m0 + ion_mode_int * MASS_PROTON + NA_H_DELTA
                elif ct == 2:
                    out_arr[pos, 6] += 1
                    if rad == 1:
                        out_arr[pos, 1] -= 1
                        out_mass[pos] = m0 - ion_mode_int * MASS_E + K_H_DELTA
                    else:
                        out_arr[pos, 1] = out_arr[pos, 1] - 1 + ion_mode_int
                        out_mass[pos] = m0 + ion_mode_int * MASS_PROTON + K_H_DELTA
                else:
                    if rad == 1:
                        out_mass[pos] = m0 - ion_mode_int * MASS_E
                    else:
                        out_arr[pos, 1] += ion_mode_int
                        out_mass[pos] = m0 + ion_mode_int * MASS_PROTON
            else:
                if ct == 1:
                    out_arr[pos, 8] += 1
                    if rad == 1:
                        out_arr[pos, 1] = out_arr[pos, 1] - 1 + ion_mode_int
                        out_mass[pos] = m0 + ion_mode_int * MASS_H + NA_H_DELTA
                    else:
                        out_arr[pos, 1] -= 1
                        out_mass[pos] = m0 + NA_H_DELTA
                elif ct == 2:
                    out_arr[pos, 6] += 1
                    if rad == 1:
                        out_arr[pos, 1] = out_arr[pos, 1] - 1 + ion_mode_int
                        out_mass[pos] = m0 + ion_mode_int * MASS_H + K_H_DELTA
                    else:
                        out_arr[pos, 1] -= 1
                        out_mass[pos] = m0 + K_H_DELTA
                else:
                    if rad == 1:
                        out_arr[pos, 1] += ion_mode_int
                        out_mass[pos] = m0 + ion_mode_int * MASS_H
                    else:
                        out_mass[pos] = m0
            out_dbe[pos] = np.float32(_dbe_12(out_arr[pos]))
            pos += 1

    order = np.argsort(out_mass)
    return out_arr[order], out_mass[order], out_dbe[order]


@njit(cache=True)
def generate_candidates_fast_numba(
    precursor_mz: float,
    ms2_mzs: np.ndarray,
    ms2_ints: np.ndarray,
    ppm: bool,
    ms1_tol: float,
    ms2_tol: float,
    adduct_net: np.ndarray,
    adduct_m: int,
    adduct_charge: int,
    adduct_net_mass: float,
    adduct_loss: np.ndarray,
    has_loss: bool,
    lower_limit: np.ndarray,
    upper_limit: np.ndarray,
    db_mode: int,
    b_mass: np.ndarray, b_form: np.ndarray, b_idx: np.ndarray,
    h_mass: np.ndarray, h_form: np.ndarray, h_idx: np.ndarray,
):
    """Ultra-fast compiled candidate generation: precursor DB + binary-search MS2 stitching."""
    pos_mode = adduct_charge > 0
    abs_q = abs(adduct_charge)
    na_contain = adduct_net[8] > 0
    k_contain = adduct_net[6] > 0
    mz_tol_da = precursor_mz * ms1_tol / 1e6 if ppm else ms1_tol
    neutral_tol = mz_tol_da * float(abs_q) / float(adduct_m)
    target_neutral = (precursor_mz * float(abs_q) + float(adduct_charge) * MASS_E - adduct_net_mass) / float(adduct_m)

    cap = 512
    cand_arr = np.empty((cap, 12), dtype=np.int16)
    cand_lo = np.empty(cap, dtype=np.uint64)
    cand_hi = np.empty(cap, dtype=np.uint64)
    cand_db = np.zeros(cap, dtype=np.int8)
    cand_int = np.zeros(cap, dtype=np.float32)
    cand_last_pk = np.full(cap, -1, dtype=np.int32)
    n_cand = 0

    ht_size = 2048
    ht_mask = ht_size - 1
    ht_slot = np.full(ht_size, -1, dtype=np.int32)

    tmp_neutral = np.empty(12, dtype=np.int16)

    # 1. Precursor DB candidates
    n_db = 2 if db_mode > 0 else 1
    for dbi in range(n_db):
        lo, hi = (_binsearch_window(b_mass, b_idx, target_neutral, neutral_tol)
                  if dbi == 0 else _binsearch_window(h_mass, h_idx, target_neutral, neutral_tol))
        src_f = b_form if dbi == 0 else h_form
        for r in range(lo, hi):
            for e in range(12):
                tmp_neutral[e] = np.int16(src_f[r, e])
            if not _formula_valid_precursor(tmp_neutral, lower_limit, upper_limit, adduct_loss, has_loss):
                continue
            klo, khi = _pack_formula_hash(tmp_neutral)
            if (n_cand + 1) * 2 >= ht_size:
                ht_size *= 2
                ht_mask = ht_size - 1
                ht_slot = np.full(ht_size, -1, dtype=np.int32)
                for ci in range(n_cand):
                    hv = (cand_lo[ci] ^ (cand_hi[ci] * np.uint64(0x9E3779B97F4A7C15))) & np.uint64(ht_mask)
                    s = int(hv)
                    while ht_slot[s] != -1:
                        s = (s + 1) & ht_mask
                    ht_slot[s] = ci
            hv = (klo ^ (khi * np.uint64(0x9E3779B97F4A7C15))) & np.uint64(ht_mask)
            s = int(hv)
            found = -1
            while ht_slot[s] != -1:
                ci = ht_slot[s]
                if cand_lo[ci] == klo and cand_hi[ci] == khi:
                    found = ci
                    break
                s = (s + 1) & ht_mask
            if found == -1:
                if n_cand == cap:
                    cap *= 2
                    new_arr = np.empty((cap, 12), dtype=np.int16); new_arr[:n_cand] = cand_arr; cand_arr = new_arr
                    new_lo = np.empty(cap, dtype=np.uint64); new_lo[:n_cand] = cand_lo; cand_lo = new_lo
                    new_hi = np.empty(cap, dtype=np.uint64); new_hi[:n_cand] = cand_hi; cand_hi = new_hi
                    new_db = np.zeros(cap, dtype=np.int8); new_db[:n_cand] = cand_db; cand_db = new_db
                    new_int = np.zeros(cap, dtype=np.float32); new_int[:n_cand] = cand_int; cand_int = new_int
                    new_lp = np.full(cap, -1, dtype=np.int32); new_lp[:n_cand] = cand_last_pk; cand_last_pk = new_lp
                for e in range(12):
                    cand_arr[n_cand, e] = tmp_neutral[e]
                cand_lo[n_cand] = klo
                cand_hi[n_cand] = khi
                cand_db[n_cand] = 1
                cand_int[n_cand] = 0.0
                cand_last_pk[n_cand] = -1
                ht_slot[s] = n_cand
                n_cand += 1

    # 2. Bottom-up MS2 stitching via sorted binary search (singly charged species only)
    n_peaks = len(ms2_mzs)
    if n_peaks > 0 and abs_q == 1:
        sum_int = 0.0
        for p in range(n_peaks):
            sum_int += float(ms2_ints[p])
        if sum_int <= 0.0:
            sum_int = 1.0

        for p in range(n_peaks):
            fmz = float(ms2_mzs[p])
            nl_mz = precursor_mz - fmz
            if nl_mz <= 0.0:
                continue
            rel_int = float(ms2_ints[p]) / sum_int

            f_arr, f_mass, f_dbe = _query_fragnl_arrays(
                fmz, True, pos_mode, na_contain, k_contain, ms2_tol, ppm, db_mode,
                b_mass, b_form, b_idx, h_mass, h_form, h_idx,
            )
            if len(f_mass) == 0:
                continue
            l_arr, l_mass, l_dbe = _query_fragnl_arrays(
                nl_mz, False, pos_mode, na_contain, k_contain, ms2_tol, ppm, db_mode,
                b_mass, b_form, b_idx, h_mass, h_form, h_idx,
            )
            n_l = len(l_mass)
            if n_l == 0:
                continue

            for fi in range(len(f_mass)):
                fm = f_mass[fi]
                fd = f_dbe[fi]
                lo_t = precursor_mz - mz_tol_da - fm
                hi_t = precursor_mz + mz_tol_da - fm

                l = 0; r = n_l
                while l < r:
                    mid = (l + r) >> 1
                    if l_mass[mid] < lo_t:
                        l = mid + 1
                    else:
                        r = mid
                li_start = l

                for li in range(li_start, n_l):
                    if l_mass[li] > hi_t:
                        break
                    if (fd + l_dbe[li]) < -0.5:
                        continue
                    ok_div = True
                    for e in range(12):
                        v = int(f_arr[fi, e]) + int(l_arr[li, e]) - int(adduct_net[e])
                        if v < 0 or (v % adduct_m) != 0:
                            ok_div = False
                            break
                        tmp_neutral[e] = np.int16(v // adduct_m)
                    if not ok_div:
                        continue
                    if not _formula_valid_precursor(tmp_neutral, lower_limit, upper_limit, adduct_loss, has_loss):
                        continue

                    klo, khi = _pack_formula_hash(tmp_neutral)
                    if (n_cand + 1) * 2 >= ht_size:
                        ht_size *= 2
                        ht_mask = ht_size - 1
                        ht_slot = np.full(ht_size, -1, dtype=np.int32)
                        for ci in range(n_cand):
                            hv = (cand_lo[ci] ^ (cand_hi[ci] * np.uint64(0x9E3779B97F4A7C15))) & np.uint64(ht_mask)
                            s = int(hv)
                            while ht_slot[s] != -1:
                                s = (s + 1) & ht_mask
                            ht_slot[s] = ci
                    hv = (klo ^ (khi * np.uint64(0x9E3779B97F4A7C15))) & np.uint64(ht_mask)
                    s = int(hv)
                    found = -1
                    while ht_slot[s] != -1:
                        ci = ht_slot[s]
                        if cand_lo[ci] == klo and cand_hi[ci] == khi:
                            found = ci
                            break
                        s = (s + 1) & ht_mask
                    if found == -1:
                        if n_cand == cap:
                            cap *= 2
                            new_arr = np.empty((cap, 12), dtype=np.int16); new_arr[:n_cand] = cand_arr; cand_arr = new_arr
                            new_lo = np.empty(cap, dtype=np.uint64); new_lo[:n_cand] = cand_lo; cand_lo = new_lo
                            new_hi = np.empty(cap, dtype=np.uint64); new_hi[:n_cand] = cand_hi; cand_hi = new_hi
                            new_db = np.zeros(cap, dtype=np.int8); new_db[:n_cand] = cand_db; cand_db = new_db
                            new_int = np.zeros(cap, dtype=np.float32); new_int[:n_cand] = cand_int; cand_int = new_int
                            new_lp = np.full(cap, -1, dtype=np.int32); new_lp[:n_cand] = cand_last_pk; cand_last_pk = new_lp
                        for e in range(12):
                            cand_arr[n_cand, e] = tmp_neutral[e]
                        cand_lo[n_cand] = klo
                        cand_hi[n_cand] = khi
                        cand_db[n_cand] = 0
                        cand_int[n_cand] = np.float32(rel_int)
                        cand_last_pk[n_cand] = p
                        ht_slot[s] = n_cand
                        n_cand += 1
                    else:
                        if cand_last_pk[found] != p:
                            cand_int[found] = np.float32(float(cand_int[found]) + rel_int)
                            cand_last_pk[found] = p

    out_arr = cand_arr[:n_cand].copy()
    out_db = cand_db[:n_cand].copy()
    out_int = cand_int[:n_cand].copy()
    out_err = np.empty(n_cand, dtype=np.float32)
    for ci in range(n_cand):
        nm = 0.0
        for e in range(12):
            nm += float(out_arr[ci, e]) * ELEM_MASSES_F64[e]
        theo_mz = (nm * float(adduct_m) + adduct_net_mass - float(adduct_charge) * MASS_E) / float(abs_q)
        if ppm:
            out_err[ci] = np.float32((theo_mz - precursor_mz) / precursor_mz * 1e6)
        else:
            out_err[ci] = np.float32(theo_mz - precursor_mz)

    return out_arr, out_db, out_int, out_err


@njit(cache=True)
def compute_ms2_evidence_fast(
    cand_neutral: np.ndarray,
    cand_charged: np.ndarray,
    peak_mzs: np.ndarray,
    peak_ints: np.ndarray,
    n_valid_peaks: int,
    adduct_charge: int,
    ppm: bool,
    ms2_tol: float,
    b_mass: np.ndarray, b_idx: np.ndarray,
    h_mass: np.ndarray, h_idx: np.ndarray,
    common_loss_keys: np.ndarray,
    common_frag_db: np.ndarray,
):
    """Vectorized Numba Stage-2 subformula assignment + 24 MS2 evidence features + explanations."""
    K = cand_neutral.shape[0]
    N = peak_mzs.shape[0]
    ms2_feats = np.zeros((K, 24), dtype=np.float32)
    best_frags = np.zeros((K, N, 12), dtype=np.int16)
    best_found = np.zeros((K, N), dtype=np.bool_)

    if K == 0 or n_valid_peaks == 0:
        return ms2_feats, best_frags, best_found

    pos_mode = adduct_charge > 0
    total_int = 0.0
    for i in range(n_valid_peaks):
        total_int += float(peak_ints[i])
    if total_int <= 0.0:
        total_int = 1.0

    # Prepare target masses and tolerances per valid peak
    targets = np.empty(n_valid_peaks, dtype=np.float64)
    tols = np.empty(n_valid_peaks, dtype=np.float64)
    for i in range(n_valid_peaks):
        pmz = float(peak_mzs[i])
        targets[i] = pmz + float(adduct_charge) * MASS_E
        tols[i] = pmz * ms2_tol / 1e6 if ppm else ms2_tol

    tmp_f = np.empty(12, dtype=np.int16)
    tmp_nl = np.empty(12, dtype=np.int16)
    delta_arr = np.empty(12, dtype=np.int16)
    exp_arrs = np.empty((n_valid_peaks, 12), dtype=np.int16)

    for k in range(K):
        ch = cand_charged[k]
        pre_dbe = _dbe_12(ch)
        pre_h2c = float(ch[1]) / float(ch[0]) if ch[0] > 0 else 0.0

        n_exp = 0
        exp_int_sum = 0.0
        db_f_cnt = 0; db_nl_cnt = 0; db_fnl_cnt = 0
        db_f_int = 0.0; db_nl_int = 0.0; db_fnl_int = 0.0
        cm_f_cnt = 0; cm_nl_cnt = 0; cm_fnl_cnt = 0
        cm_f_int = 0.0; cm_nl_int = 0.0; cm_fnl_int = 0.0
        rad_cnt = 0
        w_dbe = 0.0; w_h2c = 0.0; w_err = 0.0; w_nl_dbe_diff = 0.0

        for i in range(n_valid_peaks):
            sub_arr = closest_subformulas(ch, ELEM_MASSES_F64, targets[i], tols[i], True, False)
            if len(sub_arr) > 0:
                best_found[k, i] = True
                for e in range(12):
                    best_frags[k, i, e] = np.int16(sub_arr[0, e])
                    exp_arrs[n_exp, e] = np.int16(sub_arr[0, e])
                n_exp += 1
                exp_int_sum += float(peak_ints[i])

        if n_exp == 0:
            ms2_feats[k, 21] = float(n_valid_peaks)
            continue

        inv_exp_int = 1.0 / max(exp_int_sum, 1e-12)
        for i in range(n_valid_peaks):
            if not best_found[k, i]:
                continue
            pint = float(peak_ints[i])
            w_p = pint * inv_exp_int
            f_mass = 0.0
            for e in range(12):
                v = int(best_frags[k, i, e])
                tmp_f[e] = np.int16(v)
                tmp_nl[e] = np.int16(int(ch[e]) - v)
                f_mass += float(v) * ELEM_MASSES_F64[e]
            f_mz = f_mass - float(adduct_charge) * MASS_E
            err = (float(peak_mzs[i]) - f_mz) / max(f_mz, 1.0) * 1e6 if ppm else (float(peak_mzs[i]) - f_mz)
            sd = max(ms2_tol / 3.0, 1e-6)
            log_p = -0.5 * min((err / sd) ** 2, 25.0)

            f_dbe = _dbe_12(tmp_f)
            nl_dbe = _dbe_12(tmp_nl)
            f_h2c = float(tmp_f[1]) / float(tmp_f[0]) if tmp_f[0] > 0 else 0.0
            is_rad = (int(round(f_dbe * 2.0)) % 2) == 0
            if is_rad:
                rad_cnt += 1

            w_dbe += w_p * (f_dbe - pre_dbe)
            w_h2c += w_p * (f_h2c - pre_h2c)
            w_err += w_p * log_p
            w_nl_dbe_diff += w_p * (nl_dbe - pre_dbe)

            # Fast mass-only DB existence + common check for fragment
            f_halo = (tmp_f[2] + tmp_f[3] + tmp_f[4] + tmp_f[5]) > 0
            tmp_f[1] = tmp_f[1] + tmp_f[8] + tmp_f[6]
            tmp_f[8] = 0; tmp_f[6] = 0
            if not is_rad:
                if pos_mode:
                    tmp_f[1] -= 1
                else:
                    tmp_f[1] += 1
            f_nm = 0.0
            for e in range(12):
                f_nm += float(tmp_f[e]) * ELEM_MASSES_F64[e]
            f_db = mass_exists_fast(h_mass, h_idx, f_nm, 1e-4) if f_halo else mass_exists_fast(b_mass, b_idx, f_nm, 1e-4)

            if tmp_f[0] == 0:
                f_cm = False
                for cf_i in range(common_frag_db.shape[0]):
                    hd = int(common_frag_db[cf_i, 1]) - int(tmp_f[1])
                    if -1 <= hd <= 1:
                        eq_cf = True
                        for e in range(2, 12):
                            if common_frag_db[cf_i, e] != tmp_f[e]:
                                eq_cf = False
                                break
                        if eq_cf:
                            f_cm = True
                            break
                    elif hd > 1:
                        break
            else:
                f_cm = common_nl_lookup_fast(tmp_f, common_loss_keys)

            # Fast mass-only DB existence + common check for neutral loss
            nl_halo = (tmp_nl[2] + tmp_nl[3] + tmp_nl[4] + tmp_nl[5]) > 0
            tmp_nl[1] = tmp_nl[1] + tmp_nl[8] + tmp_nl[6]
            tmp_nl[8] = 0; tmp_nl[6] = 0
            nl_nm = 0.0
            for e in range(12):
                nl_nm += float(tmp_nl[e]) * ELEM_MASSES_F64[e]
            nl_db = mass_exists_fast(h_mass, h_idx, nl_nm, 1e-4) if nl_halo else mass_exists_fast(b_mass, b_idx, nl_nm, 1e-4)
            nl_cm = common_nl_lookup_fast(tmp_nl, common_loss_keys)

            if f_db:
                db_f_cnt += 1; db_f_int += pint
            if nl_db:
                db_nl_cnt += 1; db_nl_int += pint
            if f_db or nl_db:
                db_fnl_cnt += 1; db_fnl_int += pint
            if f_cm:
                cm_f_cnt += 1; cm_f_int += pint
            if nl_cm:
                cm_nl_cnt += 1; cm_nl_int += pint
            if f_cm or nl_cm:
                cm_fnl_cnt += 1; cm_fnl_int += pint

        sub_cnt = 0
        sub_cm_cnt = 0
        if n_exp > 1:
            for a in range(n_exp):
                for b in range(a + 1, n_exp):
                    all_ge = True
                    sum_non_c = 0
                    sum_non_n = 0
                    for e in range(12):
                        d = exp_arrs[b, e] - exp_arrs[a, e]
                        if d < 0:
                            all_ge = False
                            break
                        delta_arr[e] = d
                        if e != 0:
                            sum_non_c += d
                        if e != 7:
                            sum_non_n += d
                    if all_ge and sum_non_c > 0 and sum_non_n > 0:
                        sub_cnt += 1
                        if common_nl_lookup_fast(delta_arr, common_loss_keys):
                            sub_cm_cnt += 1
            denom = float(n_exp * (n_exp - 1))
            sub_score = 2.0 * sub_cnt / denom
            sub_cm_score = 2.0 * sub_cm_cnt / denom
        else:
            sub_score = 0.0
            sub_cm_score = 0.0

        inv_nv = 1.0 / float(n_valid_peaks)
        inv_ti = 1.0 / total_int
        exp_cnt_pct = float(n_exp) * inv_nv
        exp_int_pct = exp_int_sum * inv_ti

        ms2_feats[k, 0] = exp_cnt_pct
        ms2_feats[k, 1] = exp_int_pct
        ms2_feats[k, 2] = float(db_f_cnt) * inv_nv
        ms2_feats[k, 3] = float(db_nl_cnt) * inv_nv
        ms2_feats[k, 4] = float(db_fnl_cnt) * inv_nv
        ms2_feats[k, 5] = db_f_int * inv_ti
        ms2_feats[k, 6] = db_nl_int * inv_ti
        ms2_feats[k, 7] = db_fnl_int * inv_ti
        ms2_feats[k, 8] = float(cm_f_cnt) * inv_nv
        ms2_feats[k, 9] = float(cm_nl_cnt) * inv_nv
        ms2_feats[k, 10] = float(cm_fnl_cnt) * inv_nv
        ms2_feats[k, 11] = cm_f_int * inv_ti
        ms2_feats[k, 12] = cm_nl_int * inv_ti
        ms2_feats[k, 13] = cm_fnl_int * inv_ti
        ms2_feats[k, 14] = sub_score
        ms2_feats[k, 15] = sub_cm_score
        ms2_feats[k, 16] = float(rad_cnt) / float(n_exp)
        ms2_feats[k, 17] = w_dbe
        ms2_feats[k, 18] = w_h2c
        ms2_feats[k, 19] = w_err
        ms2_feats[k, 20] = w_nl_dbe_diff
        ms2_feats[k, 21] = float(n_valid_peaks)
        ms2_feats[k, 22] = math.sqrt(exp_cnt_pct)
        ms2_feats[k, 23] = math.sqrt(exp_int_pct)

    return ms2_feats, best_frags, best_found


# -----------------------------------------------------------------------------
# Analytical Numba MS1 Isotope Engine (exact M0..M4 convolution + mass shifts)
# -----------------------------------------------------------------------------
@njit(cache=True)
def _convolve_trunc5(dist_p, dist_m, q0, q1, q2, dm1, dm2, count):
    if count <= 0:
        return
    bp = np.zeros(5, dtype=np.float64)
    bm = np.zeros(5, dtype=np.float64)
    bp[0] = q0; bp[1] = q1; bp[2] = q2
    bm[1] = q1 * dm1; bm[2] = q2 * dm2

    rp = np.zeros(5, dtype=np.float64)
    rm = np.zeros(5, dtype=np.float64)
    rp[0] = 1.0

    n = count
    while n > 0:
        if n & 1:
            np_ = np.zeros(5, dtype=np.float64)
            nm_ = np.zeros(5, dtype=np.float64)
            for i in range(5):
                pi = rp[i]
                if pi == 0.0:
                    continue
                mi = rm[i]
                for j in range(5 - i):
                    pj = bp[j]
                    if pj == 0.0:
                        continue
                    np_[i + j] += pi * pj
                    nm_[i + j] += mi * pj + pi * bm[j]
            rp = np_; rm = nm_
        n >>= 1
        if n > 0:
            np_ = np.zeros(5, dtype=np.float64)
            nm_ = np.zeros(5, dtype=np.float64)
            for i in range(5):
                pi = bp[i]
                if pi == 0.0:
                    continue
                mi = bm[i]
                for j in range(5 - i):
                    pj = bp[j]
                    if pj == 0.0:
                        continue
                    np_[i + j] += pi * pj
                    nm_[i + j] += mi * pj + pi * bm[j]
            bp = np_; bm = nm_

    out_p = np.zeros(5, dtype=np.float64)
    out_m = np.zeros(5, dtype=np.float64)
    for i in range(5):
        pi = dist_p[i]
        if pi == 0.0:
            continue
        mi = dist_m[i]
        for j in range(5 - i):
            pj = rp[j]
            if pj == 0.0:
                continue
            out_p[i + j] += pi * pj
            out_m[i + j] += mi * pj + pi * rm[j]
    for i in range(5):
        dist_p[i] = out_p[i]
        dist_m[i] = out_m[i]


@njit(cache=True)
def calc_isotope_envelope_numba(formula_arr: np.ndarray):
    """Return exact normalized M0..M4 intensities and mean mass shifts (Da)."""
    dp = np.zeros(5, dtype=np.float64)
    dm = np.zeros(5, dtype=np.float64)
    dp[0] = 1.0

    if formula_arr[0] > 0:
        _convolve_trunc5(dp, dm, 0.9893, 0.0107, 0.0, 1.003354835, 0.0, int(formula_arr[0]))
    if formula_arr[1] > 0:
        _convolve_trunc5(dp, dm, 0.999885, 0.000115, 0.0, 1.006276746, 0.0, int(formula_arr[1]))
    if formula_arr[2] > 0:
        _convolve_trunc5(dp, dm, 0.5069, 0.0, 0.4931, 0.0, 1.9979535, int(formula_arr[2]))
    if formula_arr[3] > 0:
        _convolve_trunc5(dp, dm, 0.7576, 0.0, 0.2424, 0.0, 1.9970499, int(formula_arr[3]))
    if formula_arr[6] > 0:
        _convolve_trunc5(dp, dm, 0.932581, 0.000117, 0.067302, 1.000287, 1.998119, int(formula_arr[6]))
    if formula_arr[7] > 0:
        _convolve_trunc5(dp, dm, 0.99636, 0.00364, 0.0, 0.997034893, 0.0, int(formula_arr[7]))
    if formula_arr[9] > 0:
        _convolve_trunc5(dp, dm, 0.99757, 0.00038, 0.00205, 1.004217, 2.004245, int(formula_arr[9]))
    if formula_arr[11] > 0:
        _convolve_trunc5(dp, dm, 0.9499, 0.0075, 0.0425, 0.999387, 1.995796, int(formula_arr[11]))

    shifts = np.zeros(5, dtype=np.float64)
    s = 0.0
    for i in range(5):
        if dp[i] > 1e-15:
            shifts[i] = dm[i] / dp[i]
        s += dp[i]
    if s > 0.0:
        for i in range(5):
            dp[i] /= s
    return dp, shifts


@njit(cache=True)
def compute_ms1_features_numba(
    cand_charged: np.ndarray,
    exp_iso_ints: np.ndarray,
    exp_iso_mzs: np.ndarray,
    n_exp_iso: int,
    abs_charge: int,
):
    """Compute 12-dim rich MS1 isotope-fit features for (K, 12) charged candidates."""
    K = cand_charged.shape[0]
    out = np.zeros((K, MS1_FEAT_DIM), dtype=np.float32)
    if n_exp_iso <= 1 or K == 0:
        return out

    n_use = min(n_exp_iso, 4)
    exp_p = np.zeros(4, dtype=np.float64)
    exp_sum = 0.0
    for j in range(n_use):
        v = max(float(exp_iso_ints[j]), 0.0)
        exp_p[j] = v
        exp_sum += v
    if exp_sum <= 0.0 or exp_p[0] <= 0.0:
        return out

    exp_norm = exp_p / exp_sum
    exp_r1 = exp_p[1] / exp_p[0] if n_use >= 2 else 0.0
    exp_r2 = exp_p[2] / exp_p[0] if n_use >= 3 else 0.0
    exp_r3 = exp_p[3] / exp_p[0] if n_use >= 4 else 0.0

    has_mz = len(exp_iso_mzs) >= n_use and exp_iso_mzs[0] > 0.0
    obs_dm1 = (exp_iso_mzs[1] - exp_iso_mzs[0]) * abs_charge if (has_mz and n_use >= 2 and exp_iso_mzs[1] > 0) else 0.0
    obs_dm2 = (exp_iso_mzs[2] - exp_iso_mzs[0]) * abs_charge if (has_mz and n_use >= 3 and exp_iso_mzs[2] > 0) else 0.0

    max_sim = -1e9
    for k in range(K):
        dp, shifts = calc_isotope_envelope_numba(cand_charged[k])
        th_sum = 0.0
        for j in range(n_use):
            th_sum += dp[j]
        if th_sum <= 0.0 or dp[0] <= 1e-12:
            continue
        l1_diff = 0.0
        for j in range(n_use):
            l1_diff += abs(exp_norm[j] - dp[j] / th_sum)
        sim = 1.0 - l1_diff
        if sim > max_sim:
            max_sim = sim

        th_r1 = dp[1] / dp[0]
        th_r2 = dp[2] / dp[0]
        th_r3 = dp[3] / dp[0]

        e1 = (exp_r1 - th_r1) / (0.05 + th_r1) if n_use >= 2 else 0.0
        e2 = (exp_r2 - th_r2) / (0.05 + th_r2) if n_use >= 3 else 0.0
        e3 = (exp_r3 - th_r3) / (0.05 + th_r3) if n_use >= 4 else 0.0

        iso_ll = 0.0
        if n_use >= 2:
            s1 = 0.025 + 0.12 * th_r1
            iso_ll -= 0.5 * min(((exp_r1 - th_r1) / s1) ** 2, 25.0)
        if n_use >= 3:
            s2 = 0.020 + 0.15 * th_r2
            iso_ll -= 0.5 * min(((exp_r2 - th_r2) / s2) ** 2, 25.0)
        if n_use >= 4:
            s3 = 0.020 + 0.20 * th_r3
            iso_ll -= 0.5 * min(((exp_r3 - th_r3) / s3) ** 2, 25.0)

        m1_mda = 0.0
        if obs_dm1 > 0.5 and exp_r1 > 0.02 and shifts[1] > 0.5:
            m1_mda = max(-20.0, min(20.0, (obs_dm1 - shifts[1]) * 1000.0)) / 10.0
        m2_mda = 0.0
        if obs_dm2 > 1.5 and exp_r2 > 0.02 and shifts[2] > 1.5:
            m2_mda = max(-20.0, min(20.0, (obs_dm2 - shifts[2]) * 1000.0)) / 10.0

        out[k, 0] = 1.0
        out[k, 1] = float(n_use) / 4.0
        out[k, 2] = float(sim)
        out[k, 3] = float(min(abs(e1), 5.0))
        out[k, 4] = float(min(abs(e2), 5.0))
        out[k, 5] = float(min(abs(e3), 5.0))
        out[k, 6] = float(max(-5.0, min(5.0, e1)))
        out[k, 7] = float(max(-5.0, min(5.0, e2)))
        out[k, 8] = float(iso_ll) / 10.0
        out[k, 9] = float(m1_mda)
        out[k, 10] = float(m2_mda)
        out[k, 11] = 0.0

    if max_sim > -1e8:
        for k in range(K):
            out[k, 11] = float(out[k, 2] - max_sim)
    return out


# -----------------------------------------------------------------------------
# General Exact-Mass Spectral Summary (32 dims) + Universal Query Meta (60 dims)
# -----------------------------------------------------------------------------
@njit(cache=True)
def general_chemistry_summary_numba(
    mzs: np.ndarray,
    ints: np.ndarray,
    precursor_mz: float,
    charge: int,
    ms2_tol_ppm: float = 20.0,
) -> np.ndarray:
    """32-dim general exact-mass chemistry summary across pos/neg & all 12 elements."""
    out = np.zeros(32, dtype=np.float32)
    n_raw = len(mzs)
    if n_raw == 0 or precursor_mz <= 0.0:
        return out

    max_int = 0.0
    for i in range(n_raw):
        m = mzs[i]; v = ints[i]
        if m > 0.0 and v > 0.0 and m < precursor_mz - 0.5 and v > max_int:
            max_int = v
    if max_int <= 0.0:
        return out

    cut = max_int * 0.001
    cnt = 0
    for i in range(n_raw):
        if mzs[i] > 0.0 and ints[i] >= cut and mzs[i] < precursor_mz - 0.5:
            cnt += 1
    if cnt == 0:
        return out

    pm = np.empty(cnt, dtype=np.float64)
    pv = np.empty(cnt, dtype=np.float64)
    c = 0
    for i in range(n_raw):
        if mzs[i] > 0.0 and ints[i] >= cut and mzs[i] < precursor_mz - 0.5:
            pm[c] = mzs[i]; pv[c] = ints[i]; c += 1

    if cnt > 128:
        order = np.argsort(-pv)
        pm = pm[order[:128]]; pv = pv[order[:128]]; cnt = 128

    wsum = 0.0; wmax = 0.0
    for i in range(cnt):
        wsum += pv[i]
        if pv[i] > wmax:
            wmax = pv[i]
    w = pv / max(wsum, 1e-12)

    f_mz = 0.0; f_def = 0.0; f_abs = 0.0; f_neg = 0.0
    for i in range(cnt):
        d = pm[i] - round(pm[i])
        f_mz += w[i] * pm[i]; f_def += w[i] * d; f_abs += w[i] * abs(d)
        if d < 0.0:
            f_neg += w[i]
    f_var = 0.0
    for i in range(cnt):
        d = pm[i] - round(pm[i])
        f_var += w[i] * (d - f_def) ** 2

    out[0] = math.log1p(cnt) / 6.0
    out[1] = f_mz / 1000.0
    out[2] = f_def
    out[3] = math.sqrt(max(f_var, 0.0))
    out[4] = f_abs
    out[5] = f_neg

    l_mz = 0.0; l_def = 0.0; l_neg = 0.0
    for i in range(cnt):
        ls = precursor_mz - pm[i]
        d = ls - round(ls)
        l_mz += w[i] * ls; l_def += w[i] * d
        if d < 0.0:
            l_neg += w[i]
    l_var = 0.0
    for i in range(cnt):
        ls = precursor_mz - pm[i]
        d = ls - round(ls)
        l_var += w[i] * (d - l_def) ** 2

    out[6] = l_mz / 1000.0
    out[7] = l_def
    out[8] = math.sqrt(max(l_var, 0.0))
    out[9] = l_neg

    if cnt > 64:
        ord64 = np.argsort(-pv)[:64]
        sm = pm[ord64]; sv = pv[ord64]; sw = w[ord64]
    else:
        sm = pm; sv = pv; sw = w
    ord_m = np.argsort(sm)
    sm = sm[ord_m]; sv = sv[ord_m]; sw = sw[ord_m]
    n64 = len(sm)

    pair_tot = 1e-12
    delta_hits = np.zeros(14, dtype=np.float64)
    for i in range(n64):
        ls = precursor_mz - sm[i]
        tol = max(0.0025, precursor_mz * ms2_tol_ppm * 1e-6)
        wi = sw[i]
        pair_tot += wi
        for d_idx in range(14):
            if abs(ls - UNIVERSAL_DELTAS[d_idx]) <= tol:
                delta_hits[d_idx] += wi

    iso_13c = 0.0; iso_37cl = 0.0; iso_81br = 0.0; iso_34s = 0.0
    for i in range(n64):
        mi = sm[i]; vi = sv[i]; wi = sw[i]
        for j in range(i + 1, n64):
            diff = sm[j] - mi
            if diff > 130.0:
                break
            pw = math.sqrt(wi * sw[j])
            pair_tot += pw
            tol = max(0.0025, math.hypot(mi, sm[j]) * ms2_tol_ppm * 1e-6)
            for d_idx in range(14):
                if abs(diff - UNIVERSAL_DELTAS[d_idx]) <= tol:
                    delta_hits[d_idx] += pw
            if diff < 2.2:
                ratio = sv[j] / max(vi, 1e-12)
                if abs(diff - 1.003355) <= tol and ratio < 1.2:
                    iso_13c += pw
                if abs(diff - 1.997050) <= tol and 0.10 <= ratio <= 1.2:
                    lr = math.log(ratio / 0.32)
                    iso_37cl += pw * math.exp(-0.5 * (lr / 0.5) ** 2)
                if abs(diff - 1.997952) <= tol and 0.35 <= ratio <= 2.2:
                    lr = math.log(ratio / 0.97)
                    iso_81br += pw * math.exp(-0.5 * (lr / 0.45) ** 2)
                if abs(diff - 1.995796) <= tol and 0.01 <= ratio <= 0.25:
                    lr = math.log(ratio / 0.045)
                    iso_34s += pw * math.exp(-0.5 * (lr / 0.6) ** 2)

    for d_idx in range(14):
        out[10 + d_idx] = math.log1p(10.0 * delta_hits[d_idx] / pair_tot)

    out[24] = math.log1p(10.0 * iso_13c / pair_tot)
    out[25] = math.log1p(10.0 * iso_37cl / pair_tot)
    out[26] = math.log1p(10.0 * iso_81br / pair_tot)
    out[27] = math.log1p(10.0 * iso_34s / pair_tot)

    sgn = 1.0 if charge > 0 else -1.0
    hal_w = 0.0; oxy_w = 0.0
    for i in range(n64):
        m = sm[i]
        tol = max(0.0025, m * ms2_tol_ppm * 1e-6)
        for hm in HALIDE_NEUTRAL_MASSES:
            if abs(m - (hm - sgn * MASS_E)) <= tol:
                hal_w += sw[i]
        for om in OXYANION_NEUTRAL_MASSES:
            if abs(m - (om - sgn * MASS_E)) <= tol:
                oxy_w += sw[i]
    out[28] = hal_w
    out[29] = oxy_w

    ent = 0.0
    if cnt > 1:
        for i in range(cnt):
            if w[i] > 1e-12:
                ent -= w[i] * math.log(w[i])
        ent /= math.log(float(cnt))
    out[30] = ent
    out[31] = wmax / max(wsum, 1e-12)
    return out


def build_query_meta_60(
    mzs: np.ndarray,
    ints: np.ndarray,
    precursor_mz: float,
    adduct_net_counts: np.ndarray,
    adduct_charge: int,
    adduct_m: int,
    adduct_mass: float,
    instrument: Optional[str] = None,
    collision_energy: Optional[float] = None,
    ms2_tol_ppm: float = 20.0,
) -> np.ndarray:
    """Build 60-dim universal query_meta = 28-dim adduct/precursor/inst/CE + 32-dim general chemistry."""
    meta = np.zeros(QUERY_META_60_DIM, dtype=np.float32)
    meta[:12] = np.asarray(adduct_net_counts, dtype=np.float32) / 4.0
    meta[12] = float(adduct_charge)
    meta[13] = float(adduct_m) / 2.0
    meta[14] = 1.0 if adduct_charge > 0 else 0.0
    meta[15] = float(adduct_mass) / 50.0

    m_est = (precursor_mz * abs(adduct_charge) - adduct_mass + adduct_charge * MASS_E) / max(adduct_m, 1)
    m_est = max(m_est, 1.0)
    two_pi = 2.0 * math.pi
    meta[16] = float(m_est - round(m_est))
    meta[17] = math.sin(two_pi * m_est)
    meta[18] = math.cos(two_pi * m_est)
    meta[19] = math.sin(two_pi * m_est / 0.1)
    meta[20] = math.cos(two_pi * m_est / 0.1)
    meta[21] = math.log1p(m_est / 500.0)

    inst = str(instrument or '').strip().lower()
    if 'orbitrap' in inst or 'qexactive' in inst or 'hcd' in inst:
        meta[22] = 1.0
    elif 'tof' in inst:
        meta[23] = 1.0
    elif 'fticr' in inst or 'ftms' in inst or 'ft' in inst:
        meta[24] = 1.0
    else:
        meta[25] = 1.0

    if collision_energy is not None and math.isfinite(collision_energy) and collision_energy >= 0:
        meta[26] = min(float(collision_energy) / 100.0, 2.0)
        meta[27] = 1.0

    mzs64 = np.ascontiguousarray(mzs, dtype=np.float64)
    ints64 = np.ascontiguousarray(ints, dtype=np.float64)
    meta[28:60] = general_chemistry_summary_numba(mzs64, ints64, float(precursor_mz), int(adduct_charge), float(ms2_tol_ppm))
    return meta


class FastFormulaStore:
    """In-RAM contiguous int16 formula store + packed common-loss keys built lazily from gd."""

    def __init__(self, gd: dict):
        self.basic_mass = np.ascontiguousarray(gd['basic_db_mass'], dtype=np.float64)
        self.basic_idx = np.ascontiguousarray(gd['basic_db_idx'], dtype=np.int64)
        bf_obj = gd['basic_db_formula']
        bf = bf_obj[:] if not isinstance(bf_obj, np.ndarray) else bf_obj
        self.basic_form = np.ascontiguousarray(bf, dtype=np.int16)

        self.halogen_mass = np.ascontiguousarray(gd['halogen_db_mass'], dtype=np.float64)
        self.halogen_idx = np.ascontiguousarray(gd['halogen_db_idx'], dtype=np.int64)
        hf_obj = gd['halogen_db_formula']
        hf = hf_obj[:] if not isinstance(hf_obj, np.ndarray) else hf_obj
        self.halogen_form = np.ascontiguousarray(hf, dtype=np.int16)

        cf_obj = gd['common_frag_db']
        self.common_frag_db = np.ascontiguousarray(
            cf_obj[:] if not isinstance(cf_obj, np.ndarray) else cf_obj, dtype=np.int16
        )
        cl_obj = gd['common_loss_db']
        cl = np.ascontiguousarray(
            cl_obj[:] if not isinstance(cl_obj, np.ndarray) else cl_obj, dtype=np.int16
        )
        if len(cl) > 0:
            keys = np.array([pack_common_key(row) for row in cl], dtype=np.uint64)
            self.common_loss_keys = np.ascontiguousarray(np.unique(keys))
        else:
            self.common_loss_keys = np.empty(0, dtype=np.uint64)


def get_fast_store(gd: dict) -> FastFormulaStore:
    """Return cached FastFormulaStore for a dependencies dict, materializing once."""
    store = gd.get('_fast_store')
    if store is None:
        store = FastFormulaStore(gd)
        gd['_fast_store'] = store
    return store


def get_common_loss_keys(gd: dict) -> np.ndarray:
    """Return cached sorted uint64 packed common_loss keys without materializing full formula DB."""
    keys = gd.get('_common_loss_keys')
    if keys is None:
        cl_obj = gd['common_loss_db']
        cl = np.ascontiguousarray(
            cl_obj[:] if not isinstance(cl_obj, np.ndarray) else cl_obj, dtype=np.int16
        )
        if len(cl) > 0:
            keys = np.ascontiguousarray(np.unique(np.array([pack_common_key(row) for row in cl], dtype=np.uint64)))
        else:
            keys = np.empty(0, dtype=np.uint64)
        gd['_common_loss_keys'] = keys
    return keys
