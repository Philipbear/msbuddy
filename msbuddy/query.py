# ==============================================================================
# Copyright (C) 2023 Shipei Xing <s1xing@health.ucsd.edu>
#
# Licensed under the Apache License 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://github.com/Philipbear/msbuddy/blob/main/LICENSE
# ==============================================================================
"""
File: query.py
Author: Shipei Xing
Email: s1xing@health.ucsd.edu
GitHub: Philipbear
Description: query mass in database
"""

from math import ceil
from typing import List, Tuple, Union

import numpy as np
from numba import njit

from msbuddy.base import Adduct, Formula, calc_formula_mass

# constants
na_h_delta = 22.989769 - 1.007825
k_h_delta = 38.963707 - 1.007825


def _get_formula_db_idx(start_idx, end_idx, db_mode: int, gd) -> Tuple[int, int]:
    """
    get formula database index
    :param start_idx: start index of candidate space
    :param end_idx: end index of candidate space
    :param db_mode: database label (0: basic, 1: halogen)
    :param gd: global dependencies dictionary
    :return: database start index, database end index
    """
    if db_mode == 0:
        if start_idx >= 15000:
            db_start_idx = gd['basic_db_idx'][-1]
        else:
            db_start_idx = gd['basic_db_idx'][start_idx]
        if end_idx >= 15000:
            db_end_idx = len(gd['basic_db_idx']) - 1
        else:
            db_end_idx = gd['basic_db_idx'][end_idx]
    else:
        if start_idx >= 15000:
            db_start_idx = gd['halogen_db_idx'][-1]
        else:
            db_start_idx = gd['halogen_db_idx'][start_idx]
        if end_idx >= 15000:
            db_end_idx = len(gd['halogen_db_idx']) - 1
        else:
            db_end_idx = gd['halogen_db_idx'][end_idx]

    return int(db_start_idx), int(db_end_idx)


def query_neutral_mass(mass: float, mz_tol: float, ppm: bool, gd) -> List[Formula]:
    """
    search neutral mass in neutral formula database
    :param mass: mass to search
    :param mz_tol: mass tolerance
    :param ppm: whether ppm is used
    :param gd: global dependencies dictionary
    :return: list of Formula
    """
    # calculate mass tolerance
    mass_tol = mass * mz_tol / 1e6 if ppm else mz_tol
    # search in database
    target_mass = mass
    # formulas to return
    formulas = []

    # query database, quick filter by in-memory index array
    # quick filter by in-memory index array
    start_idx = int((target_mass - mass_tol) * 10)
    end_idx = ceil((target_mass + mass_tol) * 10)

    db_start_idx, db_end_idx = _get_formula_db_idx(start_idx, end_idx, 0, gd)
    results_basic_mass = gd['basic_db_mass'][db_start_idx:db_end_idx]
    results_basic_formula = gd['basic_db_formula'][db_start_idx:db_end_idx]
    forms_basic = _func_a(results_basic_mass, results_basic_formula, target_mass, mass_tol, None)
    formulas.extend(forms_basic)

    db_start_idx, db_end_idx = _get_formula_db_idx(start_idx, end_idx, 1, gd)
    results_halogen_mass = gd['halogen_db_mass'][db_start_idx:db_end_idx]
    results_halogen_formula = gd['halogen_db_formula'][db_start_idx:db_end_idx]
    forms_halogen = _func_a(results_halogen_mass, results_halogen_formula, target_mass, mass_tol, None)
    formulas.extend(forms_halogen)

    return formulas


def check_formula_existence(formula: Formula, pos_mode: bool, gd) -> bool:
    """
    check whether this formula exists in the database
    :param formula: formula to check
    :param gd: global dependencies dictionary
    :return: True if this formula exists in the database
    """
    form_arr = formula.array
    radical_bool = formula.dbe % 2 == 0
    halogen_bool = (form_arr[2] + form_arr[3] + form_arr[4] + form_arr[5]) > 0

    # Na, K => H
    form_arr = convert_na_k(form_arr)

    # if not a radical fragment, convert to neutral form
    if not radical_bool:
        form_arr = convert_neutral(form_arr, pos_mode)

    # recalculated target mass
    target_mass = calc_formula_mass(form_arr, 0, 0)

    # query database, use a tiny mass tolerance (1e-5)
    mass_tol = 1e-4
    db_mode = 0 if not halogen_bool else 1
    start_idx = int((target_mass - mass_tol) * 10)
    end_idx = ceil((target_mass + mass_tol) * 10)

    db_start_idx, db_end_idx = _get_formula_db_idx(start_idx, end_idx, db_mode, gd)
    if db_mode == 0:
        results_mass = gd['basic_db_mass'][db_start_idx:db_end_idx]
        results_formula = gd['basic_db_formula'][db_start_idx:db_end_idx]
    else:
        results_mass = gd['halogen_db_mass'][db_start_idx:db_end_idx]
        results_formula = gd['halogen_db_formula'][db_start_idx:db_end_idx]
    forms = _func_a(results_mass, results_formula, target_mass, mass_tol, None)

    return len(forms) > 0


def query_precursor_mass(mass: float, adduct: Adduct, mz_tol: float,
                         ppm: bool, db_mode: int, gd) -> List[Formula]:
    """
    search precursor mass in neutral database
    :param mass: mass to search
    :param adduct: adduct type
    :param mz_tol: mass tolerance
    :param ppm: whether ppm is used
    :param db_mode: database label (0: basic, 1: halogen)
    :param gd: global dependencies dictionary
    :return: list of Formula
    """
    # calculate mass tolerance
    mass_tol = mass * mz_tol / 1e6 if ppm else mz_tol
    # search in database
    ion_mode_int = 1 if adduct.pos_mode else -1
    target_mass = (mass * abs(adduct.charge) + ion_mode_int * 0.00054858 - adduct.net_formula.mass) / adduct.m

    # formulas to return
    formulas = []

    # query database, quick filter by in-memory index array
    # quick filter by in-memory index array
    start_idx = int((target_mass - mass_tol) * 10)
    end_idx = ceil((target_mass + mass_tol) * 10)

    db_start_idx, db_end_idx = _get_formula_db_idx(start_idx, end_idx, 0, gd)
    results_basic_mass = gd['basic_db_mass'][db_start_idx:db_end_idx]
    results_basic_formula = gd['basic_db_formula'][db_start_idx:db_end_idx]
    forms_basic = _func_a(results_basic_mass, results_basic_formula, target_mass, mass_tol, adduct.loss_formula)
    formulas.extend(forms_basic)

    if db_mode > 0:
        db_start_idx, db_end_idx = _get_formula_db_idx(start_idx, end_idx, 1, gd)
        results_halogen_mass = gd['halogen_db_mass'][db_start_idx:db_end_idx]
        results_halogen_formula = gd['halogen_db_formula'][db_start_idx:db_end_idx]
        forms_halogen = _func_a(results_halogen_mass, results_halogen_formula,
                                target_mass, mass_tol, adduct.loss_formula)
        formulas.extend(forms_halogen)

    return formulas


def query_fragnl_mass(mass: float, fragment: bool, pos_mode: bool, na_contain: bool, k_contain: bool,
                      mz_tol: float, ppm: bool, db_mode: int, gd) -> List[Formula]:
    """
    search fragment or neutral loss mass in neutral database
    by default, both radical and non-radical formulas are searched
    for fragments, return charged formulas; for neutral losses, return neutral formulas
    :param mass: mass to search
    :param fragment: whether this is a fragment or neutral loss
    :param pos_mode: whether this is a frag in positive ion mode
    :param na_contain: whether Na is contained in the adduct form
    :param k_contain: whether K is contained in the adduct form
    :param mz_tol: mass tolerance
    :param ppm: whether ppm is used
    :param db_mode: database label (0: basic, 1: halogen)
    :param gd: global dependencies dictionary
    :return: list of Formula
    """
    # calculate mass tolerance
    mass_tol = mass * mz_tol / 1e6 if ppm else mz_tol

    # formulas to return
    formulas = []

    # search in database
    # consider non-radical ions (even-electron)
    t_mass, start_idx, end_idx = _calc_t_mass_db_idx(mass, fragment, False, 0, pos_mode, mass_tol)
    formulas.extend(_func_b(t_mass, mass_tol, start_idx, end_idx, fragment, False,
                            False, False, pos_mode, db_mode, gd))

    # consider radical ions (odd-electron)
    t_mass, start_idx, end_idx = _calc_t_mass_db_idx(mass, fragment, True, 0, pos_mode, mass_tol)
    formulas.extend(_func_b(t_mass, mass_tol, start_idx, end_idx, fragment, True,
                            False, False, pos_mode, db_mode, gd))

    # consider Na and K
    if na_contain:
        t_mass, start_idx, end_idx = _calc_t_mass_db_idx(mass, fragment, False, na_h_delta, pos_mode, mass_tol)
        formulas.extend(_func_b(t_mass, mass_tol, start_idx, end_idx, fragment, False,
                                True, False, pos_mode, db_mode, gd))

        t_mass, start_idx, end_idx = _calc_t_mass_db_idx(mass, fragment, True, na_h_delta, pos_mode, mass_tol)
        formulas.extend(_func_b(t_mass, mass_tol, start_idx, end_idx, fragment, True,
                                True, False, pos_mode, db_mode, gd))

    if k_contain:
        t_mass, start_idx, end_idx = _calc_t_mass_db_idx(mass, fragment, False, k_h_delta, pos_mode, mass_tol)
        formulas.extend(_func_b(t_mass, mass_tol, start_idx, end_idx, fragment, False,
                                False, True, pos_mode, db_mode, gd))

        t_mass, start_idx, end_idx = _calc_t_mass_db_idx(mass, fragment, True, k_h_delta, pos_mode, mass_tol)
        formulas.extend(_func_b(t_mass, mass_tol, start_idx, end_idx, fragment, True,
                                False, True, pos_mode, db_mode, gd))

    return formulas


def check_common_frag(formula: Formula, gd) -> bool:
    """
    check whether this formula is a common fragment in Buddy.common_frag_db (C=0)
    :param formula: formula to check
    :param gd: global dependencies dictionary
    :return: True if this formula is a common fragment
    """
    form_arr = formula.array
    # carbon = 0
    if form_arr[0] != 0:
        return False

    # Na, K => H
    form_arr_1 = convert_na_k(form_arr)
    return common_frag_from_array(form_arr_1, gd['common_frag_db'])


def check_common_nl(formula: Formula, gd) -> bool:
    """
    check whether this formula is a common neutral loss in Buddy.common_nl_db
    :param formula: formula to check
    :param gd: global dependencies dictionary
    :return: True if this formula is a common neutral loss
    """
    form_arr = formula.array

    # Na, K => H
    form_arr_1 = convert_na_k(form_arr)
    return common_nl_from_array(form_arr_1, gd['common_loss_db'])


def _calc_t_mass_db_idx(mass: float, fragment: bool, radical: bool, convert_mass: float, pos_mode: bool,
                        mass_tol: float) -> Tuple[float, int, int]:
    """
    calculate target mass and calculate database index
    :param mass: mass to search
    :param fragment: whether this is a fragment ion or neutral loss
    :param radical: whether this is a radical ion
    :param convert_mass: mass to convert
    :param pos_mode: whether this is a frag in positive ion mode
    :param mass_tol: mass tolerance
    :return: target mass, database indices
    """
    if fragment:
        if not radical:
            t_mass = mass - convert_mass - 1.007276 if pos_mode else mass - convert_mass + 1.007276
        else:
            t_mass = mass - convert_mass + 0.00054858 if pos_mode else mass - convert_mass - 0.00054858
    else:
        if not radical:
            t_mass = mass - convert_mass
        else:
            t_mass = mass - convert_mass - 1.007825 if pos_mode else mass - convert_mass + 1.007825

    start_idx = int((t_mass - mass_tol) * 10)
    end_idx = ceil((t_mass + mass_tol) * 10)
    return t_mass, start_idx, end_idx


def _func_a(results_mass, results_formula, target_mass: float, mass_tol: float,
            adduct_loss_form: Union[Formula, None]) -> List[Formula]:
    """
    a helper function for query_precursor_mass
    filter and convert the sql query results to Formula objects
    :param results_mass: mass array
    :param results_formula: formula array
    :param target_mass: target mass
    :param mass_tol: mass tolerance
    :param adduct_loss_form: adduct loss formula
    :return: list of Formula
    """
    # filter by mass
    all_idx = np.where(np.abs(results_mass - target_mass) <= mass_tol)[0]

    if len(all_idx) == 0:
        return []

    # convert to Formula in neutral form
    formulas = []
    # if adduct has loss formula
    if adduct_loss_form is not None:
        l_arr = adduct_loss_form.array
        for idx in all_idx:
            if np.all(results_formula[idx] >= l_arr):
                formulas.append(Formula(results_formula[idx], charge=0, mass=results_mass[idx]))
    # if adduct has no loss formula
    else:
        for idx in all_idx:
            formulas.append(Formula(results_formula[idx], charge=0, mass=results_mass[idx]))
    return formulas


def _func_b(target_mass, mass_tol, start_idx, end_idx, fragment: bool, radical: bool,
            na_contain: bool, k_contain: bool, pos_mode: bool, db_mode: int, gd) -> List[Formula]:
    """
    a helper function for "query_fragnl_mass"
    first query the database, then convert the results to Formula objects
    :param target_mass: target mass
    :param mass_tol: mass tolerance
    :param start_idx: database start index
    :param end_idx: database end index
    :param fragment: whether this is a fragment ion or neutral loss
    :param radical: whether this is a radical ion
    :param na_contain: whether Na is contained in the adduct form
    :param k_contain: whether K is contained in the adduct form
    :param pos_mode: whether this is a frag in positive ion mode
    :param db_mode: database label (0: basic, 1: halogen, 2: all)
    :param gd: global dependencies dictionary
    :return: list of Formula
    """

    forms = []
    forms_basic = _func_c(target_mass, mass_tol, start_idx, end_idx, fragment, radical, na_contain,
                          k_contain, pos_mode, 0, gd)
    forms.extend(forms_basic)
    if db_mode > 0:
        forms_halogen = _func_c(target_mass, mass_tol, start_idx, end_idx, fragment, radical, na_contain,
                                k_contain, pos_mode, 1, gd)
        forms.extend(forms_halogen)
    return forms


def _func_c(target_mass, mass_tol, start_idx, end_idx, fragment: bool, radical: bool,
            na_contain: bool, k_contain: bool, pos_mode: bool, db_mode: int, gd) -> List[Formula]:
    """
    a helper function for _func_b
    query the database
    :param target_mass: target mass
    :param mass_tol: mass tolerance
    :param start_idx: database start index
    :param end_idx: database end index
    :param fragment: whether this is a fragment ion or neutral loss
    :param radical: whether this is a radical ion
    :param na_contain: whether Na is contained in the adduct form
    :param k_contain: whether K is contained in the adduct form
    :param pos_mode: whether this is a frag in positive ion mode
    :param db_mode: database label (0: basic, 1: halogen)
    :param gd: global dependencies dictionary
    :return: list of Formula
    """
    db_start_idx, db_end_idx = _get_formula_db_idx(start_idx, end_idx, db_mode, gd)
    if db_mode == 0:
        results_mass = gd['basic_db_mass'][db_start_idx:db_end_idx]
        results_formula = gd['basic_db_formula'][db_start_idx:db_end_idx]
    else:
        results_mass = gd['halogen_db_mass'][db_start_idx:db_end_idx]
        results_formula = gd['halogen_db_formula'][db_start_idx:db_end_idx]

    all_idx = np.where(np.abs(results_mass - target_mass) <= mass_tol)[0]

    forms = []
    if len(all_idx) == 0:
        return forms

    ion_mode_int = 1 if pos_mode else -1

    if fragment:
        if na_contain:
            if radical:
                for idx in all_idx:
                    arr = results_formula[idx].copy()
                    arr[8] += 1
                    arr[1] -= 1
                    forms.append(Formula(arr, charge=ion_mode_int,
                                         mass=results_mass[idx] - ion_mode_int * 0.00054858 + na_h_delta))
            else:
                for idx in all_idx:
                    arr = results_formula[idx].copy()
                    arr[8] += 1
                    arr[1] = arr[1] - 1 + ion_mode_int
                    forms.append(Formula(arr, charge=ion_mode_int,
                                         mass=results_mass[idx] + ion_mode_int * 1.007276 + na_h_delta))
        elif k_contain:
            if radical:
                for idx in all_idx:
                    arr = results_formula[idx].copy()
                    arr[6] += 1
                    arr[1] -= 1
                    forms.append(Formula(arr, charge=ion_mode_int,
                                         mass=results_mass[idx] - ion_mode_int * 0.00054858 + k_h_delta))
            else:
                for idx in all_idx:
                    arr = results_formula[idx].copy()
                    arr[6] += 1
                    arr[1] = arr[1] - 1 + ion_mode_int
                    forms.append(Formula(arr, charge=ion_mode_int,
                                         mass=results_mass[idx] + ion_mode_int * 1.007276 + k_h_delta))
        else:
            if radical:
                for idx in all_idx:
                    forms.append(Formula(results_formula[idx], charge=ion_mode_int,
                                         mass=results_mass[idx] - ion_mode_int * 0.00054858))
            else:
                for idx in all_idx:
                    arr = results_formula[idx].copy()
                    arr[1] += ion_mode_int
                    forms.append(Formula(arr, charge=ion_mode_int, mass=results_mass[idx] + ion_mode_int * 1.007276))
    else:  # neutral loss
        if na_contain:
            if radical:
                for idx in all_idx:
                    arr = results_formula[idx].copy()
                    arr[8] += 1
                    arr[1] = arr[1] - 1 + ion_mode_int
                    forms.append(Formula(arr, charge=0,
                                         mass=results_mass[idx] + ion_mode_int * 1.007825 + na_h_delta))
            else:
                for idx in all_idx:
                    arr = results_formula[idx].copy()
                    arr[8] += 1
                    arr[1] -= 1
                    forms.append(Formula(arr, charge=0, mass=results_mass[idx] + na_h_delta))
        elif k_contain:
            if radical:
                for idx in all_idx:
                    arr = results_formula[idx].copy()
                    arr[6] += 1
                    arr[1] = arr[1] - 1 + ion_mode_int
                    forms.append(Formula(arr, charge=0,
                                         mass=results_mass[idx] + ion_mode_int * 1.007825 + k_h_delta))
            else:
                for idx in all_idx:
                    arr = results_formula[idx].copy()
                    arr[6] += 1
                    arr[1] -= 1
                    forms.append(Formula(arr, charge=0, mass=results_mass[idx] + k_h_delta))
        else:
            if radical:
                for idx in all_idx:
                    arr = results_formula[idx].copy()
                    arr[1] += ion_mode_int
                    forms.append(Formula(arr, charge=0, mass=results_mass[idx] + ion_mode_int * 1.007825))
            else:
                for idx in all_idx:
                    forms.append(Formula(results_formula[idx], charge=0, mass=results_mass[idx]))

    return forms


@njit
def convert_na_k(form_arr: np.array) -> np.array:
    """
    convert formula to Na K converted form, Na K into H
    :param form_arr: 12-dim array
    :return: 12-dim array
    """
    # convert Na K into H
    form_arr[1] = form_arr[1] + form_arr[8] + form_arr[6]
    form_arr[8] = 0
    form_arr[6] = 0
    return form_arr


@njit
def convert_neutral(form_arr: np.array, pos_mode: bool) -> np.array:
    """
    convert charged formula into neutral formula, for fragments
    :param form_arr: 12-dim array
    :param pos_mode: whether this is a frag in positive ion mode
    :return: 12-dim array
    """
    # convert charged formula into neutral formula
    if pos_mode:
        form_arr[1] -= 1
    else:
        form_arr[1] += 1
    return form_arr



@njit
def common_frag_from_array(form_arr: np.array, frag_db: np.array) -> bool:
    """
    a helper function for checking whether a formula is a common fragment, numba accelerated
    :param form_arr: 15-dim array, Na K converted
    :param frag_db: common fragment database
    :return: True if it is a common fragment, False otherwise
    """
    # check, H tolerance: +/- 1
    for frag in frag_db:
        h_diff = frag[1] - form_arr[1]
        if abs(h_diff) <= 1:
            if (frag[2:] == form_arr[2:]).all():
                return True
        # if H diff > 1, no need to check further, since the common frag db is sorted
        elif h_diff > 1:
            break
    return False


@njit
def common_nl_from_array(form_arr: np.array, nl_db: np.array) -> bool:
    """
    a helper function for checking whether a formula is a common neutral loss, numba accelerated
    :param form_arr: 15-dim array, Na K converted
    :return: True if it is a common neutral loss, False otherwise
    """
    # check
    for nl in nl_db:
        if nl[0] == form_arr[0]:
            if (nl == form_arr).all():
                return True
        # no need to check further, since the common nl db is sorted
        elif nl[0] > form_arr[0]:
            break
    return False
