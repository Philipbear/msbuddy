# ==============================================================================
# Copyright (C) 2023 Shipei Xing <s1xing@health.ucsd.edu>
#
# Licensed under the Apache License 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://github.com/Philipbear/msbuddy/blob/main/LICENSE
# ==============================================================================
"""
File: cand.py
Author: Shipei Xing
Email: s1xing@health.ucsd.edu
GitHub: Philipbear
Description: generate candidate formula space for a metabolic feature; mass search and bottom-up MS/MS interrogation
"""

from typing import Union, List, Tuple

import numpy as np
from brainpy import isotopic_variants
from numba import njit

from msbuddy.base import Formula, CandidateFormula, MS2Explanation, MetaFeature, check_adduct, Adduct
from msbuddy.query import check_common_frag, check_common_nl, query_precursor_mass, query_fragnl_mass
from msbuddy.utils import form_arr_to_str, enumerate_subformula, read_formula, SubformulaResult, FormulaResult


class FragExplanation:
    """
    FragExplanation is a class for storing all potential fragment/nl explanations for a single MS/MS peak.
    It contains a list of fragment formulas, neutral loss formulas, and the index of the fragment.
    """

    def __init__(self, idx: int, frag: Formula, nl: Formula):
        self.idx = idx  # raw MS2 peak index
        self.frag_list = [frag]  # List[Formula]
        self.nl_list = [nl]  # List[Formula]
        self.optim_nl = None
        self.optim_frag = None

    def direct_assign_optim(self):
        """
        Directly assign the first explanation as the optimized explanation.
        """
        self.optim_frag = self.frag_list[0]
        self.optim_nl = self.nl_list[0]

    def add_frag_nl(self, frag: Formula, nl: Formula):
        self.frag_list.append(frag)
        self.nl_list.append(nl)

    def __len__(self):
        return len(self.frag_list)

    def refine_explanation_v1(self, raw_ms2_mz_arr: np.array, gd):
        """
        Refine the MS2 explanation by selecting the most reasonable explanation.
        :param raw_ms2_mz_arr: raw MS2 m/z array
        :param gd: global dictionary
        :return: fill in self.optim_frag, self.optim_nl
        """
        if len(self) == 1:
            self.optim_frag = self.frag_list[0]
            self.optim_nl = self.nl_list[0]
            return

        # if multiple explanations, select the most reasonable one
        # 1. check common neutral loss/fragment
        # 2. select the closest to the raw MS2 m/z

        # either common frag or common nl is True
        common_bool = [check_common_frag(frag, gd) or check_common_nl(nl, gd) for frag, nl in
                       zip(self.frag_list, self.nl_list)]
        # if only one common frag/nl, select it
        if sum(common_bool) == 1:
            idx = common_bool.index(True)
            self.optim_frag = self.frag_list[idx]
            self.optim_nl = self.nl_list[idx]
            return
        # if multiple common frag/nl, select the mz closest one
        elif sum(common_bool) > 1:
            # get the mass-closest one
            idx = np.argmin(np.abs(raw_ms2_mz_arr[self.idx] - np.array([f.mass for i, f in enumerate(self.frag_list)
                                                                        if common_bool[i]])))
            self.optim_frag = [f for i, f in enumerate(self.frag_list) if common_bool[i]][idx]
            self.optim_nl = [f for i, f in enumerate(self.nl_list) if common_bool[i]][idx]
            return
        # if no common frag/nl, select the closest one
        else:
            idx = np.argmin(np.abs(raw_ms2_mz_arr[self.idx] - np.array([f.mass for f in self.frag_list])))
            self.optim_frag = self.frag_list[idx]
            self.optim_nl = self.nl_list[idx]
            return

    def refine_explanation(self, raw_ms2_mz_arr: np.array):
        """
        Refine the MS2 explanation by selecting the most reasonable explanation.
        :param raw_ms2_mz_arr: raw MS2 m/z array
        :return: fill in self.optim_frag, self.optim_nl
        """
        if len(self) == 1:
            self.optim_frag = self.frag_list[0]
            self.optim_nl = self.nl_list[0]
            return

        # if multiple explanations, select the closest to the raw MS2 m/z
        idx = _find_closest_mass_idx(raw_ms2_mz_arr[self.idx], np.array([f.mass for f in self.frag_list]))
        # idx = np.argmin(np.abs(raw_ms2_mz_arr[self.idx] - np.array([f.mass for f in self.frag_list])))
        self.optim_frag = self.frag_list[idx]
        self.optim_nl = self.nl_list[idx]
        return


@njit
def _find_closest_mass_idx(mass: float, mass_arr: np.array) -> int:
    """
    find the closest mass in mass_arr to mass
    :param mass: float
    :param mass_arr: np.array
    :return: index of the closest mass
    """
    idx = np.argmin(np.abs(mass - mass_arr))
    return idx


class CandidateSpace:
    """
    CandidateSpace is a class for bottom-up MS/MS interrogation.
    It contains a precursor candidate and a list of FragExplanations.
    """

    def __init__(self, pre_neutral_array: np.array, pre_charged_array: np.array,
                 frag_exp_ls: Union[List[FragExplanation], None] = None):
        self.pre_neutral_array = np.int16(pre_neutral_array)  # precursor neutral array
        self.pre_charged_array = np.int16(pre_charged_array)  # used for ms2 global optim.
        self.neutral_mass = float(np.sum(pre_neutral_array * Formula.mass_arr))
        self.frag_exp_list = frag_exp_ls  # List[FragExplanation]

    def add_frag_exp(self, frag_exp: FragExplanation):
        self.frag_exp_list.append(frag_exp)

    def refine_explanation(self, meta_feature: MetaFeature,
                           ms2_iso_tol: float) -> CandidateFormula:
        """
        Refine the MS2 explanation by selecting the most reasonable explanation.
        Explain other fragments as isotope peaks.
        Convert into a CandidateFormula.
        :param meta_feature: MetaFeature object
        :param ms2_iso_tol: MS2 tolerance for isotope peaks, in Da
        :return: CandidateFormula
        """
        ms2_raw = meta_feature.ms2_raw
        ms2_processed = meta_feature.ms2_processed
        # for each frag_exp, refine the explanation, select the most reasonable frag/nl
        for frag_exp in self.frag_exp_list:
            frag_exp.refine_explanation(ms2_raw.mz_array)

        # consider to further explain other fragments as isotope peaks
        explained_idx = [f.idx for f in self.frag_exp_list]
        for m, exp_idx in enumerate(explained_idx):
            # last peak, skip
            if m + 1 == len(explained_idx):
                continue
            # idx of next exp peak
            next_exp_idx = explained_idx[m + 1]
            # if the next peak is already explained
            if next_exp_idx in explained_idx:
                continue
            # if this idx is not in idx_array of ms2_processed, skip
            if next_exp_idx not in ms2_processed.idx_array:
                continue
            # if the next peak is close enough, add it as an isotope peak
            if (ms2_raw.mz_array[next_exp_idx] - ms2_raw.mz_array[exp_idx] - 1.003355) <= ms2_iso_tol and \
                    ms2_raw.int_array[next_exp_idx] <= ms2_raw.int_array[exp_idx]:
                # if the next peak is close enough, add it as an isotope peak
                this_frag = self.frag_exp_list[m].optim_frag
                this_nl = self.frag_exp_list[m].optim_nl
                # add a new FragExplanation
                new_frag = Formula(this_frag.array, this_frag.charge, this_frag.mass + 1.003355, 1)
                # for iso peak, the neutral loss is actually the same as the previous one (M+0)
                # but we denote it as '-1' isotope peak, so that frag mass + nl mass is still precursor mass
                new_nl = Formula(this_nl.array, 0, this_nl.mass - 1.003355, -1)
                new_frag_exp = FragExplanation(next_exp_idx, new_frag, new_nl)
                new_frag_exp.direct_assign_optim()
                self.frag_exp_list.append(new_frag_exp)

        # sort the frag_exp_list by idx
        self.frag_exp_list = sorted(self.frag_exp_list, key=lambda x: x.idx)

        # convert into a CandidateFormula
        # construct MS2Explanation first
        ms2_raw_exp = MS2Explanation(idx_array=np.array([f.idx for f in self.frag_exp_list], dtype=np.int16),
                                     explanation_array=[f.optim_frag for f in self.frag_exp_list])

        return CandidateFormula(formula=Formula(self.pre_neutral_array, 0, self.neutral_mass),
                                ms2_raw_explanation=ms2_raw_exp)


def calc_isotope_pattern(formula: Formula,
                         iso_peaks: Union[int, None] = 4) -> np.array:
    """
    calculate isotope pattern of a neutral formula with a given adduct
    :param formula: Formula object
    :param iso_peaks: number of isotope peaks to calculate
    :return: intensity array of isotope pattern
    """
    # mapping to a dictionary
    arr_dict = {}
    for i, element in enumerate(Formula.alphabet):
        arr_dict[element] = formula.array[i]

    # calculate isotope pattern
    isotope_pattern = isotopic_variants(arr_dict, npeaks=iso_peaks)
    int_arr = np.array([iso.intensity for iso in isotope_pattern], dtype=np.float32)

    return int_arr


@njit
def calc_isotope_similarity(int_arr_x, int_arr_y, iso_num: int) -> float:
    """
    calculate isotope similarity between two ms1 isotope patterns
    :param int_arr_x: intensity array of theoretical isotope pattern
    :param int_arr_y: intensity array of experimental isotope pattern
    :param iso_num: number of isotope peaks to calculate
    :return: isotope similarity, a float between 0 and 1
    """
    min_len = min(len(int_arr_x), iso_num)
    int_arr_x = int_arr_x[:min_len]  # theoretical isotope pattern
    if len(int_arr_y) > min_len:  # experimental isotope pattern
        int_arr_y = int_arr_y[:min_len]
    if len(int_arr_y) < min_len:
        int_arr_y = np.append(int_arr_y, np.zeros(min_len - len(int_arr_y), dtype=np.float32))

    # normalize
    int_arr_x = int_arr_x / np.sum(int_arr_x, dtype=np.float32)
    int_arr_y = int_arr_y / np.sum(int_arr_y, dtype=np.float32)

    # calculate the similarity
    int_diff_arr = np.abs(int_arr_y - int_arr_x)
    sim_score = 1 - np.sum(int_diff_arr)

    return sim_score


def gen_candidate_formula(mf: MetaFeature, ppm: bool, ms1_tol: float, ms2_tol: float,
                          db_mode: int, ele_lower_limit: np.array, ele_upper_limit: np.array,
                          max_isotope_cnt: int, gd: dict) -> MetaFeature:
    """
    Generate candidate formulas for a metabolic feature.
    :param mf: MetaFeature object
    :param ppm: whether to use ppm as the unit of tolerance
    :param ms1_tol: mz tolerance for precursor ion
    :param ms2_tol: mz tolerance for fragment ions / neutral losses
    :param db_mode: database mode (int, 0: basic; 1: halogen)
    :param ele_lower_limit: lower limit of each element
    :param ele_upper_limit: upper limit of each element
    :param max_isotope_cnt: maximum isotope count, used for MS1 isotope pattern matching
    :param gd: global dictionary
    :return: fill in list of candidate formulas (CandidateFormula) in metaFeature
    """

    # if MS2 data missing or non-singly charged species, query precursor mass directly
    if not mf.ms2_processed or abs(mf.adduct.charge) > 1:
        cf_list, _ = _gen_candidate_formula_from_mz(mf, ppm, ms1_tol,
                                                    ele_lower_limit, ele_upper_limit, db_mode, gd)

    else:
        # if MS2 data available, generate candidate space with MS2 data
        ms2_cand_form_ls, ms2_cand_form_str_ls = _gen_candidate_formula_from_ms2(mf, ppm, ms1_tol, ms2_tol,
                                                                                 ele_lower_limit,
                                                                                 ele_upper_limit,
                                                                                 db_mode, gd)

        # query precursor mass, for fill in db_existed
        ms1_cand_form_ls, ms1_cand_form_str_ls = _gen_candidate_formula_from_mz(mf, ppm, ms1_tol,
                                                                                ele_lower_limit,
                                                                                ele_upper_limit, db_mode, gd)
        if len(ms2_cand_form_ls) <= 10:
            # if ms2 candidate space <= 10, merge candidate formulas
            cf_list = _merge_cand_form_list(ms1_cand_form_ls, ms2_cand_form_ls,
                                            ms1_cand_form_str_ls, ms2_cand_form_str_ls)
        else:
            # fill in db_existed
            cf_list = _fill_in_db_existence(ms1_cand_form_ls, ms2_cand_form_ls,
                                            ms1_cand_form_str_ls, ms2_cand_form_str_ls)

    # retain top candidate formulas
    # calculate neutral mass of the precursor ion
    ion_mode = 1 if mf.adduct.pos_mode else -1
    t_neutral_mass = (mf.mz - mf.adduct.net_formula.mass - ion_mode * 0.0005486) / mf.adduct.m
    mf.candidate_formula_list = _retain_top_cand_form(t_neutral_mass, cf_list, 500)

    # if MS1 isotope data is available and >1 iso peaks, calculate isotope similarity
    if mf.ms1_processed and len(mf.ms1_processed) > 1:
        for k, cf in enumerate(mf.candidate_formula_list):
            mf.candidate_formula_list[k].ms1_isotope_similarity = _calc_ms1_iso_sim(cf, mf, max_isotope_cnt)

    return mf


@njit
def _element_check(form_array: np.array, lower_limit: np.array, upper_limit: np.array) -> bool:
    """
    check whether a formula satisfies the element restriction
    :param form_array: 12-dim array
    :param lower_limit: 12-dim array
    :param upper_limit: 12-dim array
    :return: True if satisfies, False otherwise
    """
    if np.any(form_array < lower_limit) or np.any(form_array > upper_limit):
        return False
    return True


@njit
def _senior_rules(form: np.array) -> bool:
    """
    check whether a formula satisfies the senior rules
    :param form: 12-dim array
    :return: True if satisfies, False otherwise
    """
    # ["C", "H", "Br", "Cl", "F", "I", "K", "N", "Na", "O", "P", "S"]
    # int senior_1_1 = 6 * s + 5 * p + 4 * c + 3 * n + 2 * o + h + f + cl + br + i + na + k
    # int senior_1_2 = p + n + h + f + cl + br + i + na + k
    # int senior_2 = c + h + n + o + p + f + cl + br + i + s + na + k

    senior_1_1 = (6 * form[11] + 5 * form[10] + 4 * form[0] + 3 * form[7] + 2 * form[9] + form[1] + form[4] +
                  form[3] + form[2] + form[5] + form[8] + form[6])
    senior_1_1 = np.float32(senior_1_1)
    senior_1_2 = form[10] + form[7] + form[1] + form[4] + form[3] + form[2] + form[5] + form[8] + form[6]
    senior_1_2 = np.float32(senior_1_2)

    # The sum of valences or the total number of atoms having odd valences is even
    if senior_1_1 % 2 != 0 or senior_1_2 % 2 != 0:
        return False

    senior_2 = np.float32(np.sum(form))
    # The sum of valences is greater than or equal to twice the number of atoms minus 1
    if senior_1_1 < 2 * (senior_2 - 1):
        return False
    return True


@njit
def _o_p_check(form: np.array) -> bool:
    """
    check whether a formula satisfies the O/P ratio rule
    :param form: 12-dim array
    :return: True if satisfies, False otherwise
    """
    # ["C", "H", "Br", "Cl", "F", "I", "K", "N", "Na", "O", "P", "S"]
    if form[10] == 0:
        return True
    if form[9] < 3 * form[10]:
        return False
    return True


@njit
def _dbe_check(form: np.array) -> bool:
    """
    check whether a formula DBE >= 0
    :param form: 12-dim array
    :return: True if satisfies, False otherwise
    """
    dbe = form[0] + 1 - (form[1] + form[4] + form[3] + form[2] + form[5] + form[8] +
                         form[6]) / 2.0 + (form[7] + form[10]) / 2.0
    if dbe < 0:
        return False
    return True


def _adduct_loss_check(form: np.array, adduct_loss_form) -> bool:
    """
    check whether a precursor neutral formula contains the adduct loss
    :param form: 12-dim array
    :param adduct_loss_form: Formula object or None
    :return: True if contains, False otherwise
    """
    if adduct_loss_form is None:
        return True
    if np.any(form - adduct_loss_form.array < 0):
        return False
    return True


def _calc_ms1_iso_sim(cand_form, meta_feature, max_isotope_cnt) -> float:
    """
    calculate isotope similarity for a neutral precursor formula (CandidateFormula object)
    :param cand_form: CandidateFormula object
    :param meta_feature: MetaFeature object
    :param max_isotope_cnt: maximum isotope count, used for MS1 isotope pattern matching
    :return: ms1 isotope similarity
    """
    # convert neutral formula into charged form
    charged_form = Formula(cand_form.formula.array * meta_feature.adduct.m + meta_feature.adduct.net_formula.array,
                           meta_feature.adduct.charge)
    # calculate theoretical isotope pattern
    theo_isotope_pattern = calc_isotope_pattern(charged_form, max_isotope_cnt)

    # calculate ms1 isotope similarity
    ms1_isotope_sim = calc_isotope_similarity(meta_feature.ms1_processed.int_array, theo_isotope_pattern,
                                              max_isotope_cnt)

    return ms1_isotope_sim


def _gen_candidate_formula_from_mz(meta_feature: MetaFeature,
                                   ppm: bool, ms1_tol: float,
                                   lower_limit: np.array, upper_limit: np.array,
                                   db_mode: int, gd: dict) -> Tuple[List[CandidateFormula], List[str]]:
    """
    Generate candidate formulas for a metabolic feature with precursor mz only
    :param meta_feature: MetaFeature object
    :param ppm: whether to use ppm as the unit of tolerance
    :param ms1_tol: mz tolerance for precursor ions
    :param lower_limit: lower limit of each element
    :param upper_limit: upper limit of each element
    :param db_mode: database mode
    :param gd: global dictionary
    :return: list of candidate formulas (CandidateFormula), list of candidate formula strings
    """
    # query precursor mz
    formulas = query_precursor_mass(meta_feature.mz, meta_feature.adduct, ms1_tol, ppm, db_mode, gd)
    # filter out formulas that exceed element limits
    forms = [f for f in formulas if _element_check(f.array, lower_limit, upper_limit)
             and _senior_rules(f.array) and _o_p_check(f.array) and _dbe_check(f.array) and
             _adduct_loss_check(f.array, meta_feature.adduct.loss_formula)]

    # convert neutral formulas into CandidateFormula objects
    cand_form_list = [CandidateFormula(form, db_existed=True) for form in forms]
    cand_form_str_list = [form_arr_to_str(cf.formula.array) for cf in cand_form_list]

    return cand_form_list, cand_form_str_list


def _gen_candidate_formula_from_ms2(mf: MetaFeature, ppm: bool, ms1_tol: float, ms2_tol: float,
                                    lower_limit: np.array, upper_limit: np.array,
                                    db_mode: int, gd) -> Tuple[List[CandidateFormula], List[str]]:
    """
    Generate candidate formulas for a metabolic feature with MS2 data, then apply element limits
    :param mf: MetaFeature object
    :param ppm: whether to use ppm as the unit of tolerance
    :param ms1_tol: mz tolerance for precursor ions
    :param ms2_tol: mz tolerance for fragment ions / neutral losses
    :param lower_limit: lower limit of each element
    :param upper_limit: upper limit of each element
    :param db_mode: database mode
    :param gd: global dictionary
    :return: list of candidate formulas (CandidateFormula), list of candidate formula strings
    """

    # normalize MS2 intensity
    mf.ms2_processed.normalize_intensity(method='sum')

    # check whether Na and K are contained in the adduct
    na_bool = True if mf.adduct.net_formula.array[8] > 0 else False
    k_bool = True if mf.adduct.net_formula.array[6] > 0 else False

    # calculate absolute MS1 tolerance
    ms1_abs_tol = ms1_tol if not ppm else ms1_tol * mf.mz * 1e-6

    candidate_space_list = []
    existing_cand_str_list = []
    for i in range(len(mf.ms2_processed.mz_array)):
        # fragment ion m/z
        frag_mz = mf.ms2_processed.mz_array[i]
        # neutral loss m/z
        nl_mz = mf.mz - frag_mz

        # query mass in formula database
        frag_form_list, nl_form_list = _query_frag_nl_pair(frag_mz, nl_mz, mf.adduct.pos_mode, na_bool, k_bool,
                                                           ms2_tol, ppm, db_mode, gd)
        if frag_form_list is None or nl_form_list is None:
            continue

        # formula stitching
        # iterate fragment formula list and neutral loss formula list
        for frag in frag_form_list:
            for nl in nl_form_list:
                # DBE check, sum of DBE should be a non-integer
                if (frag.dbe + nl.dbe) % 1 == 0 or (frag.dbe + nl.dbe) < 0:
                    continue
                # sum mass check
                if abs(frag.mass + nl.mass - mf.mz) > ms1_abs_tol:
                    continue

                # generate precursor formula & check adduct M
                # NOTE: pre_form_arr is in neutral form
                pre_form_arr = (frag.array + nl.array - mf.adduct.net_formula.array) / mf.adduct.m
                valid_pre_form = _valid_precursor_array(pre_form_arr)
                if not valid_pre_form:
                    continue
                # if valid, convert to int16
                pre_form_arr = np.int16(pre_form_arr)

                # add to candidate space list
                candidate_space_list, existing_cand_str_list = _add_to_candidate_space_list(candidate_space_list,
                                                                                            existing_cand_str_list,
                                                                                            pre_form_arr.astype(int),
                                                                                            frag.array, nl.array)

    # element limit check, SENIOR rules, O/P check, DBE check
    candidate_list = [cs for cs in candidate_space_list
                      if _element_check(cs.pre_neutral_array, lower_limit, upper_limit)
                      and _senior_rules(cs.pre_neutral_array) and _o_p_check(cs.pre_neutral_array)
                      and _dbe_check(cs.pre_neutral_array)
                      and _adduct_loss_check(cs.pre_neutral_array, mf.adduct.loss_formula)]

    # remove candidate space variable to save memory
    del candidate_space_list

    # generate CandidateFormula object
    candidate_formula_list = [CandidateFormula(formula=Formula(cs.pre_neutral_array, 0, cs.neutral_mass),
                                               ms2_raw_explanation=None) for cs in candidate_list]
    cand_form_str_list = [form_arr_to_str(cf.formula.array) for cf in candidate_formula_list]

    return candidate_formula_list, cand_form_str_list


def _query_frag_nl_pair(frag_mz: float, nl_mz: float, pos_mode: bool, na_bool: bool, k_bool: bool,
                        ms2_tol: float, ppm: bool,
                        db_mode: int, gd) -> Tuple[Union[List[Formula], None], Union[List[Formula], None]]:
    """
    query fragment and neutral loss formulas from database
    :param frag_mz: fragment m/z
    :param nl_mz: neutral loss m/z
    :param pos_mode: positive mode or negative mode
    :param na_bool: whether Na is contained in the adduct
    :param k_bool: whether K is contained in the adduct
    :param ms2_tol: MS2 tolerance
    :param ppm: whether to use ppm as the unit of tolerance
    :param db_mode: database mode
    :param gd: global dictionary
    :return: fragment formula list, neutral loss formula list
    """
    if nl_mz < frag_mz:
        # search neutral loss first, for faster search
        nl_form_list = query_fragnl_mass(nl_mz, False, pos_mode, na_bool, k_bool,
                                         ms2_tol, ppm, db_mode, gd)
        if nl_form_list:
            frag_form_list = query_fragnl_mass(frag_mz, True, pos_mode,
                                               na_bool, k_bool, ms2_tol, ppm, db_mode, gd)
        else:
            return None, None
    else:
        frag_form_list = query_fragnl_mass(frag_mz, True, pos_mode, na_bool, k_bool,
                                           ms2_tol, ppm, db_mode, gd)
        if frag_form_list:
            nl_form_list = query_fragnl_mass(nl_mz, False, pos_mode, na_bool, k_bool,
                                             ms2_tol, ppm, db_mode, gd)
        else:
            return None, None

    return frag_form_list, nl_form_list


def _add_to_candidate_space_list(candidate_space_list: List[CandidateSpace], existing_cand_str_list: List[str],
                                 pre_form_arr: np.array, frag_arr: np.array,
                                 nl_arr: np.array) -> Tuple[List[CandidateSpace], List[str]]:
    """
    add a new candidate formula to the candidate space list
    :param candidate_space_list: candidate space list
    :param existing_cand_str_list: existing candidate formula string list
    :param pre_form_arr: precursor formula array
    :param frag_arr: fragment formula array
    :param nl_arr: neutral loss formula array
    :return: updated candidate space list
    """
    # check whether the precursor formula is already in the candidate space list
    this_pre_str = form_arr_to_str(pre_form_arr)
    candidate_exist = True if this_pre_str in existing_cand_str_list else False
    # this precursor formula has not been added to the candidate space list
    if not candidate_exist:
        candidate_space_list.append(CandidateSpace(pre_form_arr, frag_arr + nl_arr))
        existing_cand_str_list.append(this_pre_str)

    return candidate_space_list, existing_cand_str_list


@njit
def _valid_precursor_array(pre_arr: np.array) -> bool:
    """
    check adduct M
    :param pre_arr: precursor formula array
    :return: precursor formula array or None
    """
    for i in range(len(pre_arr)):
        if pre_arr[i] < 0 or pre_arr[i] % 1 != 0:
            return False
    return True


def _merge_cand_form_list(ms1_cand_list: List[CandidateFormula], ms2_cand_list: List[CandidateFormula],
                          ms1_cand_str_list: List[str], ms2_cand_str_list: List[str]) -> List[CandidateFormula]:
    """
    Merge MS1 and MS2 candidate formula lists.
    Map MS2 candidate formulas to MS1 candidate formulas (db_existed=True)
    :param ms1_cand_list: candidate formula list from MS1 mz search
    :param ms2_cand_list: candidate formula list from MS2 interrogation
    :param ms1_cand_str_list: candidate formula string list from MS1 mz search
    :param ms2_cand_str_list: candidate formula string list from MS2 interrogation
    :return: merged candidate formula list, remove duplicates
    """
    out_list = ms1_cand_list.copy()
    for m, cf in enumerate(ms2_cand_list):
        found = False
        for n, cf2 in enumerate(ms1_cand_list):
            if ms1_cand_str_list[n] == ms2_cand_str_list[m]:
                found = True
                break
        if not found:
            out_list.append(cf)

    return out_list


def _fill_in_db_existence(ms1_cand_list: List[CandidateFormula], ms2_cand_list: List[CandidateFormula],
                          ms1_cand_str_list: List[str], ms2_cand_str_list: List[str]) -> List[CandidateFormula]:
    """
    Fill in DB existence for MS2 candidate formulas.
    :param ms1_cand_list: candidate formula list from MS1 mz search
    :param ms2_cand_list: candidate formula list from MS2 interrogation
    :param ms1_cand_str_list: candidate formula string list from MS1 mz search
    :param ms2_cand_str_list: candidate formula string list from MS2 interrogation
    :return: ms2 candidate formula list with db_existed filled in
    """
    for m, cf in enumerate(ms2_cand_list):
        found = False
        for n, cf2 in enumerate(ms1_cand_list):
            if ms1_cand_str_list[n] == ms2_cand_str_list[m]:
                found = True
                break
        if found:
            ms2_cand_list[m].db_existed = True

    return ms2_cand_list


def _retain_top_cand_form(t_mass: float, cf_list: List[CandidateFormula], top_n: int) -> List[CandidateFormula]:
    """
    Retain top candidate formulas.
    :param t_mass: target neutral mass of the precursor ion
    :param cf_list: candidate formula list
    :param top_n: number of top candidate formulas to retain
    :return: retained candidate formula list
    """
    if len(cf_list) <= top_n:
        return cf_list
    else:
        # sort candidate list by mz difference (increasing)
        cf_list.sort(key=lambda x: abs(x.formula.mass - t_mass))
        return cf_list[:top_n]


@njit
def _form_array_equal(arr1: np.array, arr2: np.array) -> bool:
    """
    check whether two formula arrays are equal
    :param arr1: 12-dim array
    :param arr2: 12-dim array
    :return: True if equal, False otherwise
    """
    return True if np.equal(arr1, arr2).all() else False


def assign_subformula_cand_form(mf: MetaFeature, ppm: bool, ms2_tol: float) -> MetaFeature:
    """
    Assign subformula to all candidate formulas in a MetaFeature object.
    :param mf: MetaFeature object
    :param ppm: whether to use ppm as the unit of tolerance
    :param ms2_tol: mz tolerance for fragment ions / neutral losses
    :return: MetaFeature object
    """

    for k, cf in enumerate(mf.candidate_formula_list):
        # enumerate all subformulas
        pre_charged_arr = cf.formula.array * mf.adduct.m + mf.adduct.net_formula.array
        subform_arr = enumerate_subformula(pre_charged_arr)

        # mono mass
        mass_arr = _calc_subform_mass(subform_arr, mf.adduct.charge)
        # assign ms2 explanation
        mf.candidate_formula_list[k] = _assign_ms2_explanation(mf, cf, pre_charged_arr, subform_arr, mass_arr,
                                                               ppm, ms2_tol)

    return mf


@njit
def _calc_subform_mass(subform_arr: np.array, adduct_charge: int) -> np.array:
    """
    Calculate mass of each subformula.
    :param subform_arr: 2D array, each row is a subformula array
    :param adduct_charge: adduct charge
    :return: 1D array, mass of each subformula
    """
    mass_arr = np.empty(subform_arr.shape[0], dtype=np.float32)
    ele_mass_arr = np.array([12.000000, 1.007825, 78.918336, 34.968853, 18.998403, 126.904473, 38.963707, 14.003074,
                             22.989769, 15.994915, 30.973762, 31.972071], dtype=np.float32)
    for i in range(subform_arr.shape[0]):
        # element wise multiplication
        mass_arr[i] = np.sum(subform_arr[i, :] * ele_mass_arr, dtype=np.float32) - np.float32(adduct_charge * 0.0005486)
    return mass_arr


@njit
def _dbe_subform_filter(subform_arr: np.array, cutoff: float) -> np.array:
    """
    Filter subformulas by DBE.
    :param subform_arr: 2D array, each row is a subformula array
    :return: boolean array
    """
    dbe_arr = subform_arr[:, 0] + 1 - (subform_arr[:, 1] + subform_arr[:, 4] + subform_arr[:, 3] + subform_arr[:, 2]
                                       + subform_arr[:, 5] + subform_arr[:, 8] + subform_arr[:, 6]) / 2 + \
              (subform_arr[:, 7] + subform_arr[:, 10]) / 2
    dbe_bool_arr = dbe_arr >= cutoff
    return dbe_bool_arr


@njit
def _senior_subform_filter(subform_arr: np.array) -> np.array:
    """
    Filter subformulas by SENIOR rules.
    :param subform_arr: 2D array, each row is a subformula array
    :return: boolean array
    """
    senior_1_1_arr = 6 * subform_arr[:, 11] + 5 * subform_arr[:, 10] + 4 * subform_arr[:, 0] + \
                     3 * subform_arr[:, 7] + 2 * subform_arr[:, 9] + subform_arr[:, 1] + subform_arr[:, 4] + \
                     subform_arr[:, 3] + subform_arr[:, 2] + subform_arr[:, 5] + subform_arr[:, 8] + \
                     subform_arr[:, 6]
    senior_2_arr = np.sum(subform_arr, axis=1)
    senior_bool_arr = (senior_1_1_arr >= 2 * (senior_2_arr - 2))

    return senior_bool_arr


@njit
def _valid_subform_check(subform_arr: np.array, pre_charged_arr: np.array) -> np.array:
    """
    Check whether a subformula (frag and loss) is valid. e.g., 'C2', 'N4', 'P2'; O >= 2*P
    :param subform_arr: 2D array, each row is a subformula array
    :return: boolean array
    """
    # for frag or loss
    frag_atom_sum = np.sum(subform_arr, axis=1)
    loss_form_arr = pre_charged_arr - subform_arr
    loss_atom_sum = np.sum(loss_form_arr, axis=1)
    invalid_bool_arr = (frag_atom_sum == subform_arr[:, 0]) | (frag_atom_sum == subform_arr[:, 7]) | \
                       (frag_atom_sum == subform_arr[:, 10]) | (loss_atom_sum == loss_form_arr[:, 0]) | \
                       (loss_atom_sum == loss_form_arr[:, 7]) | (loss_atom_sum == loss_form_arr[:, 10])

    # O >= 2*P if P > 0
    invalid_o_p_frag_bool_arr = (subform_arr[:, 9] < 2 * subform_arr[:, 10]) & (subform_arr[:, 10] > 0)
    invalid_o_p_loss_bool_arr = (loss_form_arr[:, 9] < 2 * loss_form_arr[:, 10]) & (loss_form_arr[:, 10] > 0)

    invalid_bool_arr = invalid_bool_arr | invalid_o_p_frag_bool_arr | invalid_o_p_loss_bool_arr

    return ~invalid_bool_arr


def _assign_ms2_explanation(mf: MetaFeature, cf: CandidateFormula, pre_charged_arr: np.array,
                            subform_arr: np.array, mass_arr: np.array,
                            ppm: bool, ms2_tol: float) -> CandidateFormula:
    """
    Assign MS2 explanation to a candidate formula.
    :param mf: MetaFeature object
    :param cf: CandidateFormula object
    :param pre_charged_arr: precursor charged array
    :param subform_arr: 2D array, each row is a subformula array
    :param mass_arr: 1D array, mass of each subformula
    :param ppm: whether to use ppm as the unit of tolerance
    :param ms2_tol: mz tolerance for fragment ions / neutral losses
    :return: CandidateFormula object
    """
    candidate_space = None
    ion_mode_int = 1 if mf.adduct.pos_mode else -1
    for i in range(len(mf.ms2_processed.mz_array)):
        # retrieve all indices of mass within tolerance
        this_ms2_tol = ms2_tol if not ppm else ms2_tol * mf.ms2_processed.mz_array[i] * 1e-6
        idx_list = np.where(abs(mf.ms2_processed.mz_array[i] - mass_arr) <= this_ms2_tol)[0]

        if len(idx_list) == 0:
            continue

        # retrieve all subformulas within tolerance
        this_subform_arr = subform_arr[idx_list, :]
        this_mass = mass_arr[idx_list]

        # dbe filter (DBE >= -1)
        bool_arr_1 = _dbe_subform_filter(this_subform_arr, -1.)
        # this_subform_arr = this_subform_arr[bool_arr_1, :]
        # this_mass = this_mass[bool_arr_1]

        # SENIOR rules filter, a soft version
        bool_arr_2 = _senior_subform_filter(this_subform_arr)
        # this_subform_arr = this_subform_arr[bool_arr_2, :]
        # this_mass = this_mass[bool_arr_2]

        # valid subformula check
        bool_arr_3 = _valid_subform_check(this_subform_arr, pre_charged_arr)
        # this_subform_arr = this_subform_arr[bool_arr_3, :]
        # this_mass = this_mass[bool_arr_3]

        # # combine filters
        bool_arr = bool_arr_1 & bool_arr_2 & bool_arr_3
        this_subform_arr = this_subform_arr[bool_arr, :]
        this_mass = this_mass[bool_arr]

        # if no valid subformula, skip
        if this_subform_arr.shape[0] == 0:
            continue

        frag_exp = FragExplanation(mf.ms2_processed.idx_array[i],
                                   Formula(this_subform_arr[0, :], ion_mode_int, this_mass[0]),
                                   Formula(pre_charged_arr - this_subform_arr[0, :], 0))
        # add all subformulas
        if len(this_mass) > 1:
            for j in range(1, len(this_mass)):
                frag_exp.add_frag_nl(Formula(this_subform_arr[j, :], ion_mode_int, this_mass[j]),
                                     Formula(pre_charged_arr - this_subform_arr[j, :], 0))

        if candidate_space is None:
            # create CandidateSpace object
            candidate_space = CandidateSpace(cf.formula.array, pre_charged_arr, [frag_exp])
        else:
            candidate_space.add_frag_exp(frag_exp)

    # if no MS2 explanation, return original candidate formula
    if not candidate_space:
        return cf

    # refine MS2 explanation
    ms2_iso_tol = ms2_tol if not ppm else ms2_tol * mf.mz * 1e-6
    ms2_iso_tol = max(ms2_iso_tol, 0.02)
    candidate_form = candidate_space.refine_explanation(mf, ms2_iso_tol)
    candidate_form.ml_a_prob = cf.ml_a_prob  # copy ml_a_prob
    candidate_form.ms1_isotope_similarity = cf.ms1_isotope_similarity  # copy ms1_isotope_similarity

    return candidate_form


def assign_subformula(ms2_mz: List, precursor_formula: str, adduct: str,
                      ms2_tol: float = 10, ppm: bool = True,
                      dbe_cutoff: float = -1.0) -> Union[List[SubformulaResult], None]:
    """
    Assign subformulas to a given MS2 spectrum with a given precursor formula and adduct. Radical ions are considered.
    :param ms2_mz: MS2 m/z list
    :param precursor_formula: precursor formula string, uncharged
    :param adduct: adduct string, e.g., [M+H]+
    :param ms2_tol: MS2 tolerance
    :param ppm: whether MS2 tolerance is in ppm
    :param dbe_cutoff: float, DBE cutoff
    :return: a list of SubformulaResult objects
    """
    assert len(ms2_mz) > 0, "MS2 m/z list is empty."

    # convert precursor formula to array
    form_arr = read_formula(precursor_formula)
    if form_arr is None:
        return None

    # convert adduct to array
    valid_adduct, pos_mode = check_adduct(adduct)
    if not valid_adduct:
        raise ValueError("Invalid adduct string.")
    ion = Adduct(adduct, pos_mode)
    ion_mode_int = 1 if pos_mode else -1

    # enumerate all subformulas
    pre_charged_arr = form_arr * ion.m + ion.net_formula.array
    subform_arr = enumerate_subformula(pre_charged_arr)

    # mono mass
    mass_arr = _calc_subform_mass(subform_arr, ion.charge)

    # assign ms2 explanation
    out_list = []
    for k, mz in enumerate(ms2_mz):
        # retrieve all indices of mass within tolerance
        this_ms2_tol = ms2_tol if not ppm else ms2_tol * mz * 1e-6
        idx_list = np.where(abs(mz - mass_arr) <= this_ms2_tol)[0]

        if len(idx_list) == 0:
            out_list.append(SubformulaResult(k, []))
            continue

        # retrieve all subformulas within tolerance
        this_subform_arr = subform_arr[idx_list, :]
        this_mass = mass_arr[idx_list]

        # dbe filter (DBE >= dbe_cutoff)
        bool_arr_1 = _dbe_subform_filter(this_subform_arr, dbe_cutoff)

        # SENIOR rules filter, a soft version
        bool_arr_2 = _senior_subform_filter(this_subform_arr)

        # valid subformula check
        bool_arr_3 = _valid_subform_check(this_subform_arr, pre_charged_arr)

        # combine filters
        bool_arr = bool_arr_1 & bool_arr_2 & bool_arr_3
        this_subform_arr = this_subform_arr[bool_arr, :]
        this_mass = this_mass[bool_arr]

        # if no valid subformula, skip
        if this_subform_arr.shape[0] == 0:
            out_list.append(SubformulaResult(k, []))
            continue

        # create SubformulaResult object
        subform_list = []
        for j in range(this_subform_arr.shape[0]):
            form = Formula(this_subform_arr[j, :], ion_mode_int, this_mass[j])
            subform_list.append(FormulaResult(form.__str__(), form.mass, mz))
        out_list.append(SubformulaResult(k, subform_list))

    return out_list
