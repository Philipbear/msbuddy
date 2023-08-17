import sys
import numpy as np
from tqdm import tqdm
from file_io import init_db, load_usi, load_mgf
from typing import Tuple, Union, List
from msbuddy.base_class import MetaFeature
from msbuddy.gen_candidate import gen_candidate_formula


class BuddyParamSet:
    """
    Buddy parameter set
    """
    def __init__(self,
                 ppm=True, ms1_tol=5, ms2_tol=10, halogen=False,
                 c_range: Tuple[int, int] = (0, 80),
                 h_range: Tuple[int, int] = (0, 150),
                 n_range: Tuple[int, int] = (0, 20),
                 o_range: Tuple[int, int] = (0, 30),
                 p_range: Tuple[int, int] = (0, 10),
                 s_range: Tuple[int, int] = (0, 15),
                 f_range: Tuple[int, int] = (0, 20),
                 cl_range: Tuple[int, int] = (0, 15),
                 br_range: Tuple[int, int] = (0, 10),
                 i_range: Tuple[int, int] = (0, 10),
                 isotope_bin_mztol: float = 0.02, max_isotope_cnt: int = 4,
                 ms2_denoise: bool = True,
                 rel_int_denoise: bool = True, rel_int_denoise_cutoff: float = 0.01,
                 max_noise_frag_ratio: float = 0.85, max_noise_rsd: float = 0.20,
                 max_frag_reserved: int = 50,
                 use_all_frag: bool = False,
                 ms2_global_optim: bool = False):
        """
        :param ppm: whether ppm is used for m/z tolerance
        :param ms1_tol: MS1 m/z tolerance
        :param ms2_tol: MS2 m/z tolerance
        :param halogen: whether to include halogen atoms; if False, ranges of F, Cl, Br, I will be set to (0, 0)
        :param c_range: C range
        :param h_range: H range
        :param n_range: N range
        :param o_range: O range
        :param p_range: P range
        :param s_range: S range
        :param f_range: F range
        :param cl_range: Cl range
        :param br_range: Br range
        :param i_range: I range
        :param isotope_bin_mztol: m/z tolerance for isotope bin, used for MS1 isotope pattern
        :param max_isotope_cnt: maximum isotope count, used for MS1 isotope pattern
        :param ms2_denoise: whether to denoise MS2 spectrum
        :param rel_int_denoise: whether to use relative intensity for MS2 denoise
        :param rel_int_denoise_cutoff: relative intensity cutoff, used for MS2 denoise
        :param max_noise_frag_ratio: maximum noise fragment ratio, used for MS2 denoise
        :param max_noise_rsd: maximum noise RSD, used for MS2 denoise
        :param max_frag_reserved: max fragment number reserved, used for MS2 data
        :param use_all_frag: whether to use all fragments for annotation; by default, only top N fragments are used, top N is a function of precursor mass
        :param ms2_global_optim: whether to use global optimization for MS2 data (to refine fragment annotation)
        """
        self.ppm = ppm
        self.ms1_tol = ms1_tol
        self.ms2_tol = ms2_tol
        self.halogen = halogen
        self.ele_lower = np.array([c_range[0], h_range[0], br_range[0], cl_range[0], f_range[0], i_range[0],
                                   0, n_range[0], 0, o_range[0], p_range[0], s_range[0]])
        self.ele_upper = np.array([c_range[1], h_range[1], br_range[1], cl_range[1], f_range[1], i_range[1],
                                   0, n_range[1], 0, o_range[1], p_range[1], s_range[1]])
        if not self.halogen:
            self.ele_lower[2:6] = 0
            self.ele_upper[2:6] = 0

        # check valid range
        if np.any(self.ele_lower < 0) or np.any(self.ele_upper < 0):
            raise ValueError("Element range cannot be negative.")
        if np.any(self.ele_lower > self.ele_upper):
            raise ValueError("Element lower bound cannot be larger than upper bound.")

        # check valid param set
        if isotope_bin_mztol <= 0:
            raise ValueError("Isotope bin m/z tolerance must be positive.")
        if max_isotope_cnt <= 0:
            raise ValueError("Maximum isotope count must be positive.")
        if max_noise_frag_ratio <= 0 or max_noise_frag_ratio >= 1:
            raise ValueError("Maximum noise fragment ratio must be in the range of 0 to 1.")
        if max_noise_rsd <= 0:
            raise ValueError("Maximum noise RSD must be positive.")
        if max_frag_reserved <= 0:
            raise ValueError("Top N fragment must be positive.")

        self.isotope_bin_mztol = isotope_bin_mztol
        self.max_isotope_cnt = max_isotope_cnt
        self.ms2_denoise = ms2_denoise
        self.rel_int_denoise = rel_int_denoise
        self.rel_int_denoise_cutoff = rel_int_denoise_cutoff
        self.max_noise_frag_ratio = max_noise_frag_ratio
        self.max_noise_rsd = max_noise_rsd
        self.max_frag_reserved = max_frag_reserved
        self.use_all_frag = use_all_frag
        self.ms2_global_optim = ms2_global_optim


class Buddy:
    """
    Buddy class
    Buddy data is stored in self.data, which is a List[MetaFeature]; MetaFeature is a class defined in
    base_class/MetaFeature.py
    """
    def __init__(self, param_set: Union[BuddyParamSet, None] = None):

        self.db_mode = 0  # 0: no halogen, 1: halogen included
        if param_set is None:
            self.param_set = BuddyParamSet()  # default parameter set
            self.db_loaded = init_db(0)  # database initialization
        else:
            self.param_set = param_set
            # check if halogen is included
            if sum(self.param_set.ele_upper[2:6]) > 0:
                self.db_mode = 1  # halogen included

        self.db_loaded = init_db(self.db_mode)  # database initialization
        self.data = None  # List[MetabolicFeature], metabolic feature list

    def load_usi(self, usi_list: List[str]):
        self.data = load_usi(usi_list)

    def load_mgf(self, file_path):
        self.data = load_mgf(file_path)

    def add_data(self, data: List[MetaFeature]):
        """
        add customized data
        :param data: metabolic feature list
        """
        self.data = data

    def annotate_formula(self) -> int:
        """
        annotate formula for loaded data
        pipeline: data preprocessing -> formula candidate space generation -> ml_a feature generation (ms1) -> ml_a model A
        -> ml_b feature generation (ms2) -> ml_a model B -> formula annotation -> FDR calculation
        :return: number of annotated metabolic features
        """
        if not self.data:
            raise ValueError("No data loaded.")

        ps = self.param_set

        # data preprocessing
        for meta_feature in tqdm(self.data, desc="Data preprocessing", file=sys.stdout, colour="green"):
            meta_feature.data_preprocess(ps.ppm, ps.ms1_tol, ps.ms2_tol,
                                         ps.isotope_bin_mztol, ps.max_isotope_cnt, ps.ms2_denoise, ps.rel_int_denoise,
                                         ps.rel_int_denoise_cutoff, ps.max_noise_frag_ratio, ps.max_noise_rsd,
                                         ps.max_frag_reserved, ps.use_all_frag)

            # generate formula candidate space
            # currently, ms2_global_optim is not used here
            gen_candidate_formula(meta_feature, ps.ppm, ps.ms1_tol, ps.ms2_tol, self.db_mode, False,
                                  ps.ele_lower, ps.ele_upper, ps.max_isotope_cnt)

        # ml_a feature generation + prediction
        #################

        # ml_b feature generation + prediction
        ##################

        # formula annotation + FDR calculation
        ##################

        return 0


# test
if __name__ == '__main__':
    # create parameter set
    buddy_param_set = BuddyParamSet(
        ppm=True, ms1_tol=5, ms2_tol=10,
        c_range=(0, 80), h_range=(0, 150), n_range=(0, 20),
        o_range=(0, 30), p_range=(0, 10), s_range=(0, 15))
    # initiate a Buddy project with the given parameter set
    buddy = Buddy(buddy_param_set)
    # load data
    buddy.load_mgf("../demo.mgf")
    # annotate formula
    buddy.annotate_formula()

    # buddy.load_usi("mzspec:GNPS:TASK-c95481f0c53d42e78a61bf899e9f9adb-spectra/specs_ms.mgf:scan:1943")

    # use default parameter set
    buddy = Buddy()
    buddy.load_mgf("../demo.mgf")
    buddy.annotate_formula()
