from typing import Union, List
import numpy as np
from msbuddy.base import Formula, CandidateFormula, MS2Explanation, MetaFeature
from msbuddy.query import check_common_frag, check_common_nl, query_precursor_mass, query_fragnl_mass, convert_na_k
from brainpy import isotopic_variants
from numba import njit


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

        # if multiple explanations, select the most reasonable one
        # 1. check common neutral loss/fragment
        # 2. select the closest to the raw MS2 m/z

        # either common frag or common nl is True
        common_bool = [check_common_frag(frag) or check_common_nl(nl) for frag, nl in zip(self.frag_list, self.nl_list)]
        # if only one common frag/nl, select it
        if sum(common_bool) == 1:
            idx = common_bool.index(True)
            self.optim_frag = self.frag_list[idx]
            self.optim_nl = self.nl_list[idx]
            return
        # if multiple common frag/nl, select the closest one
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


class CandidateSpace:
    """
    CandidateSpace is a class for bottom-up MS/MS interrogation.
    It contains a precursor candidate and a list of FragExplanations.
    """

    def __init__(self, pre_neutral_array: np.array, pre_charged_array: np.array,
                 frag_exp: FragExplanation):
        self.pre_neutral_array = pre_neutral_array
        self.pre_charged_array = pre_charged_array  # used for ms2 global optim.
        self.neutral_mass = float(np.sum(pre_neutral_array * Formula.mass_arr))
        self.frag_exp_list = [frag_exp]  # List[FragExplanation]

    def add_frag_exp(self, frag_exp: FragExplanation):
        self.frag_exp_list.append(frag_exp)

    def __len__(self):
        return len(self.frag_exp_list)

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
            # if the next peak is already explained or the last peak, skip
            if (exp_idx + 1 in explained_idx) or (m + 1 == len(explained_idx)):
                continue
            # idx of next exp peak
            next_exp_idx = explained_idx[m + 1]
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
                new_frag_exp = FragExplanation(exp_idx + 1, new_frag, new_nl)
                new_frag_exp.direct_assign_optim()
                self.frag_exp_list.append(new_frag_exp)

        # sort the frag_exp_list by idx
        self.frag_exp_list = sorted(self.frag_exp_list, key=lambda x: x.idx)

        # convert into a CandidateFormula
        # construct MS2Explanation first
        ms2_raw_exp = MS2Explanation(idx_array=np.array([f.idx for f in self.frag_exp_list]),
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
    int_arr = np.array([iso.intensity for iso in isotope_pattern])

    return int_arr


def calc_isotope_similarity(int_arr_x, int_arr_y,
                            iso_num: int) -> float:
    """
    calculate isotope similarity between two ms1 isotope patterns
    :param int_arr_x: int array of theoretical isotope pattern
    :param int_arr_y: int array of experimental isotope pattern
    :param iso_num: number of isotope peaks to calculate
    :return: isotope similarity, a float between 0 and 1
    """
    min_len = min(len(int_arr_x), iso_num)
    int_arr_x = int_arr_x[:min_len]  # theoretical isotope pattern
    if len(int_arr_y) > min_len:  # experimental isotope pattern
        int_arr_y = int_arr_y[:min_len]
    if len(int_arr_y) < min_len:
        int_arr_y = np.append(int_arr_y, np.zeros(min_len - len(int_arr_y)))

    # normalize
    int_arr_x = int_arr_x.astype(np.float64)
    int_arr_x /= sum(int_arr_x)
    int_arr_y = int_arr_y.astype(np.float64)
    int_arr_y /= sum(int_arr_y)

    # calculate the similarity
    int_diff_arr = abs(int_arr_y - int_arr_x)
    sim_score = 1 - np.sum(int_diff_arr)

    return sim_score


def gen_candidate_formula(meta_feature: MetaFeature, ppm: bool, ms1_tol: float, ms2_tol: float,
                          db_mode: int, element_lower_limit: np.array, element_upper_limit: np.array,
                          max_isotope_cnt: int):
    """
    Generate candidate formulas for a metabolic feature.
    :param meta_feature: MetaFeature object
    :param ppm: whether to use ppm as the unit of tolerance
    :param ms1_tol: mz tolerance for precursor ion
    :param ms2_tol: mz tolerance for fragment ions / neutral losses
    :param db_mode: database mode (int, 0: basic; 1: halogen)
    :param element_lower_limit: lower limit of each element
    :param element_upper_limit: upper limit of each element
    :param max_isotope_cnt: maximum isotope count, used for MS1 isotope pattern matching
    :return: fill in list of candidate formulas (CandidateFormula) in metaFeature
    """

    # if MS2 data missing or non-singly charged species, query precursor mass directly
    if not meta_feature.ms2_processed or abs(meta_feature.adduct.charge) > 1:
        meta_feature.candidate_formula_list = _gen_candidate_formula_from_mz(meta_feature, ppm, ms1_tol,
                                                                             element_lower_limit,
                                                                             element_upper_limit, db_mode)

    else:
        # if MS2 data available, generate candidate space with MS2 data
        ms2_cand_form_list = _gen_candidate_formula_from_ms2(meta_feature, ppm, ms1_tol, ms2_tol,
                                                             element_lower_limit, element_upper_limit, db_mode)
        ms1_cand_form_list = _gen_candidate_formula_from_mz(meta_feature, ppm, ms1_tol,
                                                            element_lower_limit, element_upper_limit, db_mode)
        # merge candidate formulas
        meta_feature.candidate_formula_list = _merge_cand_form_list(ms1_cand_form_list, ms2_cand_form_list)

    # if MS1 isotope data is available and >1 iso peaks, calculate isotope similarity
    if meta_feature.ms1_processed and len(meta_feature.ms1_processed) > 1:
        for candidate_form in meta_feature.candidate_formula_list:
            candidate_form.ms1_isotope_similarity = \
                _calc_ms1_iso_sim(candidate_form, meta_feature, max_isotope_cnt)


# @njit
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


# @njit
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

    senior_1_1 = 6 * form[11] + 5 * form[10] + 4 * form[0] + 3 * form[7] + 2 * form[9] + form[1] + form[4] + form[3] + \
                 form[2] + form[5] + form[8] + form[6]
    senior_1_2 = form[10] + form[7] + form[1] + form[4] + form[3] + form[2] + form[5] + form[8] + form[6]
    # The sum of valences or the total number of atoms having odd valences is even
    if senior_1_1 % 2 != 0 or senior_1_2 % 2 != 0:
        return False

    senior_2 = np.sum(form)
    # The sum of valences is greater than or equal to twice the number of atoms minus 1
    if senior_1_1 < 2 * (senior_2 - 1):
        return False
    return True


# @njit
def _o_p_check(form: np.array) -> bool:
    """
    check whether a formula satisfies the O/P ratio rule
    :param form: 12-dim array
    :return: True if satisfies, False otherwise
    """
    # ["C", "H", "Br", "Cl", "F", "I", "K", "N", "Na", "O", "P", "S"]
    if form[10] == 0:
        return True
    if form[9] / form[10] < 3:
        return False
    return True


# @njit
def _dbe_check(form: np.array) -> bool:
    """
    check whether a formula DBE >= 0
    :param form: 12-dim array
    :return: True if satisfies, False otherwise
    """
    dbe = form[0] + 1 - (form[1] + form[4] + form[3] + form[2] + form[5] + form[8] + form[6]) / 2 + \
          (form[7] + form[10]) / 2
    if dbe < 0:
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
                                   db_mode: int) -> List[CandidateFormula]:
    """
    Generate candidate formulas for a metabolic feature with precursor mz only
    :param meta_feature: MetaFeature object
    :param ppm: whether to use ppm as the unit of tolerance
    :param ms1_tol: mz tolerance for precursor ions
    :param lower_limit: lower limit of each element
    :param upper_limit: upper limit of each element
    :param db_mode: database mode
    :return: list of candidate formulas (CandidateFormula)
    """
    # query precursor mz
    formulas = query_precursor_mass(meta_feature.mz, meta_feature.adduct, ms1_tol, ppm, db_mode)
    # filter out formulas that exceed element limits
    forms = [f for f in formulas if _element_check(f.array, lower_limit, upper_limit)
             and _senior_rules(f.array) and _o_p_check(f.array) and _dbe_check(f.array)]

    # convert neutral formulas into CandidateFormula objects
    return [CandidateFormula(form) for form in forms]


def _gen_candidate_formula_from_ms2(meta_feature: MetaFeature,
                                    ppm: bool, ms1_tol: float, ms2_tol: float,
                                    lower_limit: np.array, upper_limit: np.array,
                                    db_mode: int) -> List[CandidateFormula]:
    """
    Generate candidate formulas for a metabolic feature with MS2 data, then apply element limits
    :param meta_feature: MetaFeature object
    :param ppm: whether to use ppm as the unit of tolerance
    :param ms1_tol: mz tolerance for precursor ions
    :param ms2_tol: mz tolerance for fragment ions / neutral losses
    :param lower_limit: lower limit of each element
    :param upper_limit: upper limit of each element
    :param db_mode: database mode
    :return: list of candidate formulas (CandidateFormula)
    """

    # normalize MS2 intensity
    meta_feature.ms2_processed.normalize_intensity(method='sum')

    # check whether Na and K are contained in the adduct
    na_bool = False
    k_bool = False
    if meta_feature.adduct.net_formula.array[8] > 0:
        na_bool = True
    if meta_feature.adduct.net_formula.array[6] > 0:
        k_bool = True

    # generate list of candidate spaces, CandidateSpace object
    candidate_space_list = []
    ms1_abs_tol = ms1_tol if not ppm else ms1_tol * meta_feature.mz * 1e-6

    for i in range(len(meta_feature.ms2_processed.mz_array)):
        # fragment ion index
        frag_idx = meta_feature.ms2_processed.idx_array[i]
        # fragment ion m/z
        frag_mz = meta_feature.ms2_processed.mz_array[i]
        # neutral loss m/z
        nl_mz = meta_feature.mz - frag_mz

        # query mass in formula database
        if nl_mz < frag_mz:
            # search neutral loss first, for faster search
            nl_form_list = query_fragnl_mass(nl_mz, False, meta_feature.adduct.pos_mode, na_bool, k_bool,
                                             ms2_tol, ppm, db_mode)
            if nl_form_list:
                frag_form_list = query_fragnl_mass(frag_mz, True, meta_feature.adduct.pos_mode,
                                                   na_bool, k_bool, ms2_tol, ppm, db_mode)
            else:
                continue
        else:
            frag_form_list = query_fragnl_mass(frag_mz, True, meta_feature.adduct.pos_mode, na_bool, k_bool,
                                               ms2_tol, ppm, db_mode)
            if frag_form_list:
                nl_form_list = query_fragnl_mass(nl_mz, False, meta_feature.adduct.pos_mode, na_bool, k_bool,
                                                 ms2_tol, ppm, db_mode)
            else:
                continue

        # formula stitching
        # iterate list of Formula objects
        for frag in frag_form_list:
            for nl in nl_form_list:
                # DBE check, sum of DBE should be a non-integer
                if (frag.dbe + nl.dbe) % 1 == 0 or (frag.dbe + nl.dbe) < 0:
                    continue
                # sum mass check
                if abs(frag.mass + nl.mass - meta_feature.mz) > ms1_abs_tol:
                    continue

                # generate precursor formula & check adduct M
                # NOTE: pre_form_arr is in neutral form
                pre_form_arr = _gen_precursor_array(frag.array, nl.array, meta_feature.adduct.net_formula.array,
                                                    meta_feature.adduct.m)
                if pre_form_arr is None:
                    continue

                # check whether the precursor formula is already in the candidate space list
                candidate_exist = False
                for cs in candidate_space_list:
                    if (cs.pre_neutral_array == pre_form_arr).all():
                        candidate_exist = True
                        # check whether this fragment ion has been explained
                        # one fragment ion can be explained by multiple formulas
                        frag_exist = False
                        for f in cs.frag_exp_list:
                            if f.idx == frag_idx:
                                frag_exist = True
                                f.add_frag_nl(frag, nl)
                                break
                        # this fragment ion has not been explained under this precursor formula
                        if not frag_exist:
                            cs.add_frag_exp(FragExplanation(frag_idx, frag, nl))
                        break
                # this precursor formula has not been added to the candidate space list
                if not candidate_exist:
                    candidate_space_list.append(CandidateSpace(pre_form_arr, frag.array + nl.array,
                                                               FragExplanation(frag_idx, frag, nl)))

    # element limit check, SENIOR rules, O/P check, DBE check
    candidate_list = [cs for cs in candidate_space_list
                      if _element_check(cs.pre_neutral_array, lower_limit, upper_limit)
                      and _senior_rules(cs.pre_neutral_array) and _o_p_check(cs.pre_neutral_array)
                      and _dbe_check(cs.pre_neutral_array)]

    # remove candidate space variable to save memory
    del candidate_space_list

    # calculate neutral mass of the precursor ion
    ion_mode_int = 1 if meta_feature.adduct.pos_mode else -1
    t_neutral_mass = (meta_feature.mz - meta_feature.adduct.net_formula.mass -
                      ion_mode_int * 0.0005485799) / meta_feature.adduct.m
    # presort candidate list by explained MS2 peak count (decreasing), then by mz difference (increasing)
    candidate_list.sort(key=lambda x: (-len(x.frag_exp_list), abs(x.neutral_mass - t_neutral_mass)))

    # retain top 2000 candidate spaces
    if len(candidate_list) > 2000:
        candidate_list = candidate_list[:2000]

    # generate CandidateFormula object, refine MS2 explanation
    ms2_iso_tol = ms2_tol if not ppm else ms2_tol * meta_feature.mz * 1e-6
    # common frag/nl + mz diff, consider isotopes
    candidate_formula_list = [cs.refine_explanation(meta_feature, ms2_iso_tol) for cs in candidate_list]

    return candidate_formula_list


def _gen_precursor_array(frag_arr: np.array, nl_array: np.array, adduct_array: np.array, adduct_m: int) -> np.array:
    """
    generate precursor formula array from frag array, nl array and adduct array
    check adduct M
    :param frag_arr: fragment formula array
    :param nl_array: neutral loss formula array
    :param adduct_array: adduct net formula array
    :param adduct_m: adduct M
    :return: precursor formula array
    """
    if adduct_m == 1:
        return frag_arr + nl_array - adduct_array
    else:
        pre_array = (frag_arr + nl_array - adduct_array) / adduct_m
        # all elements should be integers
        if np.all(pre_array % 1 == 0):
            return pre_array
        else:
            return None


def _merge_cand_form_list(ms1_cand_list: List[CandidateFormula],
                          ms2_cand_list: List[CandidateFormula]) -> List[CandidateFormula]:
    """
    Merge MS1 and MS2 candidate formula lists.
    :param ms1_cand_list: candidate formula list from MS1 mz search
    :param ms2_cand_list: candidate formula list from MS2 interrogation
    :return: merged candidate formula list, remove duplicates
    """
    out_list = ms2_cand_list.copy()
    for cf in ms1_cand_list:
        found = False
        for cf2 in ms2_cand_list:
            if _form_array_equal(cf.formula.array, cf2.formula.array):
                found = True
                break
        if not found:
            out_list.append(cf)

    return out_list


@njit
def _form_array_equal(arr1: np.array, arr2: np.array) -> bool:
    """
    check whether two formula arrays are equal
    :param arr1: 12-dim array
    :param arr2: 12-dim array
    :return: True if equal, False otherwise
    """
    return True if np.equal(arr1, arr2).all() else False
