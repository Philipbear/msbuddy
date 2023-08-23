import numpy as np
from typing import List, Tuple
from msbuddy.base import Adduct, Formula
from math import ceil
from numba import njit
from msbuddy.utils import dependencies

# constants
na_h_delta = 22.989769 - 1.007825
k_h_delta = 38.963707 - 1.007825


def _get_formula_db_idx(start_idx, end_idx, db_mode: int) -> Tuple[int, int]:
    """
    get formula database index
    :param start_idx: start index of candidate space
    :param end_idx: end index of candidate space
    :param db_mode: database label (0: basic, 1: halogen)
    :return: database start index, database end index
    """
    if db_mode == 0:
        if start_idx >= 15000:
            db_start_idx = dependencies['basic_db_idx'][-1]
        else:
            db_start_idx = dependencies['basic_db_idx'][start_idx]
        if end_idx >= 15000:
            db_end_idx = len(dependencies['basic_db_idx']) - 1
        else:
            db_end_idx = dependencies['basic_db_idx'][end_idx]
    else:
        if start_idx >= 15000:
            db_start_idx = dependencies['halogen_db_idx'][-1]
        else:
            db_start_idx = dependencies['halogen_db_idx'][start_idx]
        if end_idx >= 15000:
            db_end_idx = len(dependencies['halogen_db_idx']) - 1
        else:
            db_end_idx = dependencies['halogen_db_idx'][end_idx]

    return int(db_start_idx), int(db_end_idx)


def query_precursor_mass(mass: float, adduct: Adduct, mz_tol: float, ppm: bool, db_mode: int) \
        -> List[Formula]:
    """
    search precursor mass in neutral database
    :param mass: mass to search
    :param adduct: adduct type
    :param mz_tol: mass tolerance
    :param ppm: whether ppm is used
    :param db_mode: database label (0: basic, 1: halogen)
    :return: list of Formula
    """
    # calculate mass tolerance
    mass_tol = mass * mz_tol / 1e6 if ppm else mz_tol
    # search in database
    target_mass = (mass * abs(adduct.charge) - adduct.net_formula.mass) / adduct.m

    # formulas to return
    formulas = []

    # query database, quick filter by in-memory index array
    # quick filter by in-memory index array
    start_idx = int((target_mass - mass_tol) * 10)
    end_idx = ceil((target_mass + mass_tol) * 10)

    db_start_idx, db_end_idx = _get_formula_db_idx(start_idx, end_idx, 0)
    results_basic_mass = dependencies['basic_db_mass'][db_start_idx:db_end_idx]
    results_basic_formula = dependencies['basic_db_formula'][db_start_idx:db_end_idx]
    forms_basic = _func_a(results_basic_mass, results_basic_formula, target_mass, mass_tol, adduct.loss_formula)
    formulas.extend(forms_basic)

    if db_mode > 0:
        db_start_idx, db_end_idx = _get_formula_db_idx(start_idx, end_idx, 1)
        results_halogen_mass = dependencies['halogen_db_mass'][db_start_idx:db_end_idx]
        results_halogen_formula = dependencies['halogen_db_formula'][db_start_idx:db_end_idx]
        forms_halogen = _func_a(results_halogen_mass, results_halogen_formula,
                                target_mass, mass_tol, adduct.loss_formula)
        formulas.extend(forms_halogen)

    return formulas


def query_fragnl_mass(mass: float, fragment: bool, pos_mode: bool, na_contain: bool, k_contain: bool,
                      mz_tol: float, ppm: bool, db_mode: int) -> List[Formula]:
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
    :return: list of Formula
    """
    # calculate mass tolerance
    mass_tol = mass * mz_tol / 1e6 if ppm else mz_tol

    # formulas to return
    formulas = []

    # search in database
    # consider non-radical ions (even-electron)
    t_mass, start_idx, end_idx = _calc_t_mass_db_idx(mass, fragment, False, 0, pos_mode, mass_tol)
    formulas.extend(_func_b(t_mass, mass_tol, start_idx, end_idx, fragment, False, False, False, pos_mode, db_mode))

    # consider radical ions (odd-electron)
    t_mass, start_idx, end_idx = _calc_t_mass_db_idx(mass, fragment, True, 0, pos_mode, mass_tol)
    formulas.extend(_func_b(t_mass, mass_tol, start_idx, end_idx, fragment, True, False, False, pos_mode, db_mode))

    # consider Na and K
    if na_contain:
        t_mass, start_idx, end_idx = _calc_t_mass_db_idx(mass, fragment, False, na_h_delta, pos_mode, mass_tol)
        formulas.extend(_func_b(t_mass, mass_tol, start_idx, end_idx, fragment, False, True, False, pos_mode, db_mode))

        t_mass, start_idx, end_idx = _calc_t_mass_db_idx(mass, fragment, True, na_h_delta, pos_mode, mass_tol)
        formulas.extend(_func_b(t_mass, mass_tol, start_idx, end_idx, fragment, True, True, False, pos_mode, db_mode))

    if k_contain:
        t_mass, start_idx, end_idx = _calc_t_mass_db_idx(mass, fragment, False, k_h_delta, pos_mode, mass_tol)
        formulas.extend(_func_b(t_mass, mass_tol, start_idx, end_idx, fragment, False, False, True, pos_mode, db_mode))

        t_mass, start_idx, end_idx = _calc_t_mass_db_idx(mass, fragment, True, k_h_delta, pos_mode, mass_tol)
        formulas.extend(_func_b(t_mass, mass_tol, start_idx, end_idx, fragment, True, False, True, pos_mode, db_mode))

    return formulas


def check_common_frag(formula: Formula) -> bool:
    """
    check whether this formula is a common fragment in Buddy.common_frag_db (C=0)
    :param formula: formula to check
    :return: True if this formula is a common fragment
    """
    form_arr = formula.array
    # carbon = 0
    if form_arr[0] != 0:
        return False

    # Na, K => H
    form_arr_1 = convert_na_k(form_arr)
    return common_frag_from_array(form_arr_1, dependencies['common_frag_db'])


def check_common_nl(formula: Formula) -> bool:
    """
    check whether this formula is a common neutral loss in Buddy.common_nl_db
    :param formula: formula to check
    :return: True if this formula is a common neutral loss
    """
    form_arr = formula.array

    # Na, K => H
    form_arr_1 = convert_na_k(form_arr)
    return common_nl_from_array(form_arr_1, dependencies['common_loss_db'])


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


def _func_a(results_mass, results_formula, target_mass: float, mass_tol: float, adduct_loss_form: Formula)\
        -> List[Formula]:
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
            na_contain: bool, k_contain: bool, pos_mode: bool, db_mode: int) -> List[Formula]:
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
    :return: list of Formula
    """

    forms = []
    forms_basic = _func_c(target_mass, mass_tol, start_idx, end_idx, fragment, radical, na_contain,
                          k_contain, pos_mode, 0)
    forms.extend(forms_basic)
    if db_mode > 0:
        forms_halogen = _func_c(target_mass, mass_tol, start_idx, end_idx, fragment, radical, na_contain,
                                k_contain, pos_mode, 1)
        forms.extend(forms_halogen)
    return forms


def _func_c(target_mass, mass_tol, start_idx, end_idx, fragment: bool, radical: bool,
            na_contain: bool, k_contain: bool, pos_mode: bool, db_mode: int) -> List[Formula]:
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
    :return: list of Formula
    """
    db_start_idx, db_end_idx = _get_formula_db_idx(start_idx, end_idx, db_mode)
    if db_mode == 0:
        results_mass = dependencies['basic_db_mass'][db_start_idx:db_end_idx]
        results_formula = dependencies['basic_db_formula'][db_start_idx:db_end_idx]
    else:
        results_mass = dependencies['halogen_db_mass'][db_start_idx:db_end_idx]
        results_formula = dependencies['halogen_db_formula'][db_start_idx:db_end_idx]

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


def convert_na_k(form_arr: np.array) -> np.array:
    """
    convert formula to Na K converted form, Na K into H
    :param form_arr: 12-dim array
    :return: 12-dim array
    """
    # convert Na K into H
    form_arr[1] += form_arr[8] + form_arr[6]
    form_arr[8] = 0
    form_arr[6] = 0
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
            if np.equal(frag[2:], form_arr[2:]).all():
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
            if np.equal(nl, form_arr).all():
                return True
        # no need to check further, since the common nl db is sorted
        elif nl[0] > form_arr[0]:
            break
    return False
