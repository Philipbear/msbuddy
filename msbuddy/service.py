import numpy as np
from typing import List, Tuple
from msbuddy.base_class import Adduct, Formula
from math import ceil
from numba import njit
from msbuddy.utils import dependencies

# constants
na_h_delta = 22.989769 - 1.007825
k_h_delta = 38.963707 - 1.007825


class FormulaModel:
    """
    formula database model
    """
    def __init__(self,
                 # formula_str: str,
                 mass: float,
                 formula_arr: np.array
                 # pubchem: int,
                 # other_db: int
                 ):
        # self.formula_str = formula_str
        self.mass = mass
        self.formula_arr = formula_arr
        # self.pubchem = pubchem
        # self.other_db = other_db



def _get_formula_db_idx(start_idx, end_idx, db_mode: int) -> Tuple[int, int]:
    """
    get formula database index
    :param start_idx: start index of candidate space
    :param end_idx: end index of candidate space
    :param db_mode: database label (0: basic, 1: halogen)
    :param basic_db_idx: basic formula database index, CHNOPS
    :param halogen_db_idx: halogen formula database index, CHNOPSFClBrI
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
    elif db_mode == 1:
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
    results_basic = dependencies['basic_db'][db_start_idx:db_end_idx]
    forms_basic = _func_a(results_basic, target_mass, mass_tol, adduct.loss_formula)
    formulas.extend(forms_basic)

    if db_mode > 0:
        db_start_idx, db_end_idx = _get_formula_db_idx(start_idx, end_idx, 1)
        results_halogen = dependencies['halogen_db'][db_start_idx:db_end_idx]
        forms_halogen = _func_a(results_halogen, target_mass, mass_tol, adduct.loss_formula)
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


def _func_a(results, target_mass: float, mass_tol: float, adduct_loss_form: Formula) -> List[Formula]:
    """
    a helper function for query_precursor_mass
    filter and convert the sql query results to Formula objects
    :param results: sql query results
    :param target_mass: target mass
    :param mass_tol: mass tolerance
    :param adduct_loss_form: adduct loss formula
    :return: list of Formula
    """
    # filter by mass
    results = [form for form in results if abs(form.mass - target_mass) <= mass_tol]

    if len(results) == 0:
        return []

    # if adduct has loss formula
    if adduct_loss_form is not None:
        l_arr = adduct_loss_form.array
        bool_arr = [True] * len(results)
        for i, form in enumerate(results):
            if np.any(form.formula_arr < l_arr):
                bool_arr[i] = False
        results = [form for i, form in enumerate(results) if bool_arr[i]]

    # convert to Formula in neutral form
    formulas = []
    for result in results:
        formulas.append(Formula(result.formula_arr, charge=0, mass=result.mass))
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
        results = [f for f in dependencies['basic_db'][db_start_idx:db_end_idx]
                   if abs(f.mass - target_mass) <= mass_tol]
    elif db_mode == 1:
        results = [f for f in dependencies['halogen_db'][db_start_idx:db_end_idx]
                   if abs(f.mass - target_mass) <= mass_tol]

    forms = []
    if len(results) == 0:
        return forms

    ion_mode_int = 1 if pos_mode else -1

    if fragment:
        if na_contain:
            if radical:
                for result in results:
                    arr = result.formula_arr.copy()
                    arr[8] += 1
                    arr[1] -= 1
                    forms.append(Formula(arr, charge=ion_mode_int,
                                         mass=result.mass - ion_mode_int * 0.00054858 + na_h_delta))
            else:
                for result in results:
                    arr = result.formula_arr.copy()
                    arr[8] += 1
                    arr[1] = arr[1] - 1 + ion_mode_int
                    forms.append(Formula(arr, charge=ion_mode_int,
                                         mass=result.mass + ion_mode_int * 1.007276 + na_h_delta))
        elif k_contain:
            if radical:
                for result in results:
                    arr = result.formula_arr.copy()
                    arr[6] += 1
                    arr[1] -= 1
                    forms.append(Formula(arr, charge=ion_mode_int,
                                         mass=result.mass - ion_mode_int * 0.00054858 + k_h_delta))
            else:
                for result in results:
                    arr = result.formula_arr.copy()
                    arr[6] += 1
                    arr[1] = arr[1] - 1 + ion_mode_int
                    forms.append(Formula(arr, charge=ion_mode_int,
                                         mass=result.mass + ion_mode_int * 1.007276 + k_h_delta))
        else:
            if radical:
                for result in results:
                    forms.append(Formula(result.formula_arr, charge=ion_mode_int, mass=result.mass - ion_mode_int * 0.00054858))
            else:
                for result in results:
                    arr = result.formula_arr.copy()
                    arr[1] += ion_mode_int
                    forms.append(Formula(arr, charge=ion_mode_int, mass=result.mass + ion_mode_int * 1.007276))
    else:  # neutral loss
        if na_contain:
            if radical:
                for result in results:
                    arr = result.formula_arr.copy()
                    arr[8] += 1
                    arr[1] = arr[1] - 1 + ion_mode_int
                    forms.append(Formula(arr, charge=0,
                                         mass=result.mass + ion_mode_int * 1.007825 + na_h_delta))
            else:
                for result in results:
                    arr = result.formula_arr.copy()
                    arr[8] += 1
                    arr[1] -= 1
                    forms.append(Formula(arr, charge=0, mass=result.mass + na_h_delta))
        elif k_contain:
            if radical:
                for result in results:
                    arr = result.formula_arr.copy()
                    arr[6] += 1
                    arr[1] = arr[1] - 1 + ion_mode_int
                    forms.append(Formula(arr, charge=0,
                                         mass=result.mass + ion_mode_int * 1.007825 + k_h_delta))
            else:
                for result in results:
                    arr = result.formula_arr.copy()
                    arr[6] += 1
                    arr[1] -= 1
                    forms.append(Formula(arr, charge=0, mass=result.mass + k_h_delta))
        else:
            if radical:
                for result in results:
                    arr = result.formula_arr.copy()
                    arr[1] += ion_mode_int
                    forms.append(Formula(arr, charge=0, mass=result.mass + ion_mode_int * 1.007825))
            else:
                for result in results:
                    forms.append(Formula(result.formula_arr, charge=0, mass=result.mass))

    return forms


# @njit
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
            if np.array_equal(frag[2:], form_arr[2:]):
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
            if np.array_equal(nl, form_arr):
                return True
        # no need to check further, since the common nl db is sorted
        elif nl[0] > form_arr[0]:
            break
    return False


# test
if __name__ == '__main__':
    import time
    from file_io import init_db

    start = time.time()
    init_db(0)
    print('init db time: ', time.time() - start)

    start = time.time()
    # adduct_ = Adduct(string='[M + H]+', pos_mode=True)
    # for i in range(10000):
    #     formulas_ = query_precursor_mass(300, adduct_, 0.01, False, 1)
    # print(len(formulas_))
    # for formula in formulas_:
    #     print(formula)

    # for i in range(1000):
    #     subforms = query_fragnl_mass(300, fragment=False, pos_mode=True, na_contain=False, k_contain=False,
    #                                  mz_tol=0.02, ppm=False, db_mode=1)
    # print(len(subforms))


    # for subform in subforms:
    #     print(subform)


    # check common frag
    form = Formula(np.array([10, 14, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]), charge=0)
    for i in range(100000):
        a = check_common_frag(form)
        b = check_common_nl(form)
    print(a, b)

    print(time.time() - start)
