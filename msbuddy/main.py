import sys
import numpy as np
from tqdm import tqdm
from file_io import init_db, load_usi, load_mgf
from typing import Tuple, Union, List
from msbuddy.base_class import MetaFeature
from msbuddy.gen_candidate import gen_candidate_formula
from msbuddy.ml import pred_formula_feasibility, pred_formula_prob, calc_fdr
from multiprocessing import Pool, cpu_count


class BuddyParamSet:
    """
    Buddy parameter set
    """

    def __init__(self,
                 ppm=True, ms1_tol=5, ms2_tol=10, halogen=False,
                 multiprocess=True, n_cpu=-1,
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
        :param multiprocess: whether to use multiprocessing for annotation
        :param n_cpu: number of CPUs used for multiprocessing; if -1, all available CPUs will be used
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
        self.db_mode = 0 if not halogen else 1
        self.multiprocess = multiprocess

        cpu_cnt = cpu_count()
        if n_cpu == -1 or n_cpu > cpu_cnt:
            self.n_cpu = cpu_cnt
        else:
            self.n_cpu = n_cpu

        self.ele_lower = np.array([c_range[0], h_range[0], br_range[0], cl_range[0], f_range[0], i_range[0],
                                   0, n_range[0], 0, o_range[0], p_range[0], s_range[0]])
        self.ele_upper = np.array([c_range[1], h_range[1], br_range[1], cl_range[1], f_range[1], i_range[1],
                                   0, n_range[1], 0, o_range[1], p_range[1], s_range[1]])
        if not halogen:
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
    # singleton
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Buddy, cls).__new__(cls)
        return cls._instance

    def __init__(self, param_set: Union[BuddyParamSet, None] = None):

        if param_set is None:
            self.param_set = BuddyParamSet()  # default parameter set
        else:
            self.param_set = param_set

        self.db_loaded = init_db(self.param_set.db_mode)  # database initialization

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

    def annotate_formula(self):
        """
        annotate formula for loaded data
        pipeline: data preprocessing -> formula candidate space generation -> ml_a feature generation (ms1) -> ml_a model A
        -> ml_b feature generation (ms2) -> ml_a model B -> formula annotation -> FDR calculation
        :return: None
        """
        if not self.data:
            raise ValueError("No data loaded.")

        def _preprocess_gen_cand(meta_feature: MetaFeature, ps: BuddyParamSet):
            """
            data preprocessing & formula candidate space generation
            :param meta_feature: MetaFeature object
            :param ps: BuddyParamSet object
            :return: None
            """
            # data preprocessing
            meta_feature.data_preprocess(ps.ppm, ps.ms1_tol, ps.ms2_tol,
                                         ps.isotope_bin_mztol, ps.max_isotope_cnt, ps.ms2_denoise, ps.rel_int_denoise,
                                         ps.rel_int_denoise_cutoff, ps.max_noise_frag_ratio, ps.max_noise_rsd,
                                         ps.max_frag_reserved, ps.use_all_frag)

            # generate formula candidate space; currently, ms2_global_optim is not used here
            gen_candidate_formula(meta_feature, ps.ppm, ps.ms1_tol, ps.ms2_tol, ps.db_mode, False,
                                  ps.ele_lower, ps.ele_upper, ps.max_isotope_cnt)

        if self.param_set.multiprocess:
            # multiprocessing
            with Pool(self.param_set.n_cpu) as pool:
                pool.imap(_preprocess_gen_cand, self.data)
        else:
            # common for loop
            for mf in tqdm(self.data, desc="Data preprocessing & candidate space generation",
                           file=sys.stdout, colour="green"):
                _preprocess_gen_cand(mf, self.param_set)

        # ml_a feature generation + prediction
        pred_formula_feasibility(self.data)

        # ml_b feature generation + prediction
        pred_formula_prob(self.data, self.param_set.ppm, self.param_set.ms1_tol, self.param_set.ms2_tol)

        # FDR calculation
        calc_fdr(self.data)

    def result_summary(self) -> List[dict]:
        """
        summarize results
        :return: a list of dictionary containing result summary
        """
        if not self.data:
            raise ValueError("No data loaded.")

        result_summary_list = []
        for mf in self.data:
            result_summary_list.append(mf.summarize_result())

        return result_summary_list


# test
if __name__ == '__main__':
    # # create parameter set
    # buddy_param_set = BuddyParamSet(
    #     ppm=True, ms1_tol=5, ms2_tol=10, halogen=False, multiprocess=True, n_cpu=-1,
    #     c_range=(0, 80), h_range=(0, 150), n_range=(0, 20),
    #     o_range=(0, 30), p_range=(0, 10), s_range=(0, 15))
    # # initiate a Buddy project with the given parameter set
    # buddy = Buddy(buddy_param_set)
    # # load data
    # # buddy.load_usi(["mzspec:GNPS:TASK-c95481f0c53d42e78a61bf899e9f9adb-spectra/specs_ms.mgf:scan:1943"])
    # buddy.load_mgf("/Users/philip/Documents/test_data/test.mgf")
    # # annotate formula
    # # buddy.annotate_formula()
    # # # result summary
    # result_summary = buddy.result_summary()

    #########################################
    buddy_param_set = BuddyParamSet(multiprocess=False)
    # use default parameter set
    buddy = Buddy(buddy_param_set)
    # buddy.load_mgf("/Users/philip/Documents/test_data/test.mgf")
    buddy.load_usi(["mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00005467952",
                    "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00005716808"])

    # add ms1 data
    from msbuddy.base_class import Spectrum

    # buddy.data[0].ms1_raw = Spectrum(mz_array=np.array([540.369, 541.369]),
    #                                  int_array=np.array([100, 28]))
    # buddy.data[1].ms1_raw = Spectrum(mz_array=np.array([buddy.data[1].mz, buddy.data[1].mz + 1]),
    #                                  int_array=np.array([100, 25]))

    buddy.annotate_formula()
    result_summary_ = buddy.result_summary()
    print(result_summary_)
    print('done')
