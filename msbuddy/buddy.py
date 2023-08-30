import logging
import sys
import numpy as np
from tqdm import tqdm
from typing import Tuple, Union, List
from timeout_decorator import timeout
from msbuddy.base import MetaFeature
from msbuddy.load import init_db, load_usi, load_mgf
from msbuddy.gen_candidate import gen_candidate_formula
from msbuddy.ml import pred_formula_feasibility, pred_formula_prob, calc_fdr
from multiprocessing import Pool, cpu_count

import time

logging.basicConfig(level=logging.INFO)

# global variable containing shared data
global shared_data_dict


class BuddyParamSet:
    """
    Buddy parameter set
    """

    def __init__(self,
                 ppm: bool = True,
                 ms1_tol: float = 5,
                 ms2_tol: float = 10,
                 halogen: bool = False,
                 parallel: bool = False,
                 process_num: int = -1,
                 timeout_secs: float = 300,
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
                 use_all_frag: bool = False):
        """
        :param ppm: whether ppm is used for m/z tolerance
        :param ms1_tol: MS1 m/z tolerance
        :param ms2_tol: MS2 m/z tolerance
        :param halogen: whether to include halogen atoms; if False, ranges of F, Cl, Br, I will be set to (0, 0)
        :param parallel: whether to use parallel processing
        :param process_num: number of processes used for parallel processing
        :param timeout_secs: timeout in seconds
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
        """
        self.ppm = ppm
        self.ms1_tol = ms1_tol
        self.ms2_tol = ms2_tol
        self.db_mode = 0 if not halogen else 1
        self.parallel = parallel
        if process_num > cpu_count() or process_num <= 0:
            self.process_num = cpu_count() - 1
            logging.info(f"Processing core number is set to {self.process_num}.")
        else:
            self.process_num = process_num

        if timeout_secs <= 0:
            logging.warning("Timeout is set to 300 seconds.")
            self.timeout_secs = 300
        self.timeout_secs = timeout_secs

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
            self.isotope_bin_mztol = 0.02
            logging.warning(f"Isotope bin m/z tolerance is set to {self.isotope_bin_mztol}.")
        else:
            self.isotope_bin_mztol = isotope_bin_mztol

        if max_isotope_cnt <= 0:
            self.max_isotope_cnt = 4
            logging.warning(f"Maximum isotope count is set to {self.max_isotope_cnt}.")
        else:
            self.max_isotope_cnt = max_isotope_cnt

        self.ms2_denoise = ms2_denoise
        self.rel_int_denoise = rel_int_denoise

        if rel_int_denoise_cutoff <= 0 or rel_int_denoise_cutoff >= 1:
            self.rel_int_denoise_cutoff = 0.01
            logging.warning(f"Relative intensity denoise cutoff is set to {self.rel_int_denoise_cutoff}.")
        else:
            self.rel_int_denoise_cutoff = rel_int_denoise_cutoff

        if max_noise_frag_ratio <= 0 or max_noise_frag_ratio >= 1:
            self.max_noise_frag_ratio = 0.85
            logging.warning(f"Maximum noise fragment ratio is set to {self.max_noise_frag_ratio}.")
        else:
            self.max_noise_frag_ratio = max_noise_frag_ratio

        if max_noise_rsd <= 0 or max_noise_rsd >= 1:
            self.max_noise_rsd = 0.20
            logging.warning(f"Maximum noise RSD is set to {self.max_noise_rsd}.")
        else:
            self.max_noise_rsd = max_noise_rsd

        if max_frag_reserved <= 0:
            self.max_frag_reserved = 50
            logging.warning(f"Maximum fragment reserved is set to {self.max_frag_reserved}.")
        else:
            self.max_frag_reserved = max_frag_reserved

        self.use_all_frag = use_all_frag


class Buddy:
    """
    Buddy main class
    Buddy data is List[MetaFeature]; MetaFeature is a class defined in base_class/MetaFeature.py
    """
    # singleton pattern
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Buddy, cls).__new__(cls)
        return cls._instance

    def __init__(self, param_set: Union[BuddyParamSet, None] = None):

        if param_set is None:
            self.param_set = BuddyParamSet()  # default parameter set
        else:
            self.param_set = param_set  # customized parameter set

        global shared_data_dict  # Declare it as a global variable
        shared_data_dict = init_db(self.param_set.db_mode)  # database initialization

        self.data = None  # List[MetabolicFeature]

    def load_usi(self, usi: Union[str, List[str]],
                 adduct: Union[None, str, List[str]] = None):
        self.data = load_usi(usi, adduct)

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

        param_set = self.param_set
        modified_mf_ls = []  # modified metabolic feature list, containing annotated results

        @timeout(param_set.timeout_secs)
        def _preprocess_and_gen_cand_nonparallel(meta_feature: MetaFeature, ps: BuddyParamSet) -> MetaFeature:
            """
            a wrapper function for data preprocessing and candidate formula space generation
            :param meta_feature: MetaFeature object
            :param ps: Buddy parameter set
            :return: MetaFeature object
            """
            mf = generate_candidate_formula(meta_feature, ps, shared_data_dict)
            return mf

        start_time = time.time()
        # data preprocessing and candidate space generation
        if param_set.parallel:
            logging.info(f"Parallel processing with {param_set.process_num} processes.")
            with Pool(processes=int(param_set.process_num), initializer=init_pool,
                      initargs=(shared_data_dict,)) as pool:

                async_results = [pool.apply_async(_preprocess_and_gen_cand_parallel, (mf, param_set)) for mf in self.data]

                for i, async_result in enumerate(async_results):
                    try:
                        modified_mf = async_result.get(timeout=param_set.timeout_secs)  # get the result or timeout
                        modified_mf_ls.append(modified_mf)
                    except:
                        mf = self.data[i]
                        logging.warning(f"Timeout for spectrum {mf.identifier}, mz={mf.mz}, rt={mf.rt}, skipped.")
                        modified_mf_ls.append(mf)
                        continue

            del async_results

            # # parallel processing
            # logging.info(f"Parallel processing with {param_set.process_num} processes.")
            # with Pool(processes=int(param_set.process_num), initializer=init_pool,
            #           initargs=(shared_data_dict,)) as pool:
            #     modified_mf_ls = pool.starmap(_preprocess_and_gen_cand_parallel,
            #                                   [(mf, param_set) for mf in self.data])

        else:
            # main loop, with progress bar and timeout
            for mf in tqdm(self.data, desc="Data preprocessing & candidate space generation",
                           file=sys.stdout, colour="green"):
                try:
                    modified_mf = _preprocess_and_gen_cand_nonparallel(mf, param_set)
                    modified_mf_ls.append(modified_mf)
                except:
                    logging.warning(f"Timeout for spectrum {mf.identifier}, mz={mf.mz}, rt={mf.rt}, skipped.")
                    modified_mf_ls.append(mf)
                    continue

        end_time = time.time()
        print(f"Data preprocessing & candidate space generation time: {end_time - start_time} seconds.")

        self.data = modified_mf_ls
        del modified_mf_ls

        # ml_a feature generation + prediction
        pred_formula_feasibility(self.data, shared_data_dict)

        # ml_b feature generation + prediction
        pred_formula_prob(self.data, param_set.ppm, param_set.ms1_tol, param_set.ms2_tol, shared_data_dict)

        # FDR calculation
        calc_fdr(self.data)

    def get_summary(self) -> List[dict]:
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


def init_pool(the_dict):
    """
    initialize pool for parallel processing
    :param the_dict: global dictionary containing shared data
    :return: None
    """
    global shared_data_dict
    shared_data_dict = the_dict


def _preprocess_and_gen_cand_parallel(meta_feature: MetaFeature, ps: BuddyParamSet) -> MetaFeature:
    """
    a wrapper function for data preprocessing and candidate formula space generation
    :param meta_feature: MetaFeature object
    :param ps: Buddy parameter set
    :return: MetaFeature object
    """
    mf = generate_candidate_formula(meta_feature, ps, shared_data_dict)
    return mf


def generate_candidate_formula(mf: MetaFeature, ps: BuddyParamSet, global_dict) -> MetaFeature:
    """
    preprocess data and generate candidate formula space
    :param mf: MetaFeature object
    :param ps: Buddy parameter set
    :param global_dict: global dictionary containing shared data
    :return: MetaFeature object
    """
    # data preprocessing
    mf.data_preprocess(ps.ppm, ps.ms1_tol, ps.ms2_tol,
                       ps.isotope_bin_mztol, ps.max_isotope_cnt, ps.ms2_denoise, ps.rel_int_denoise,
                       ps.rel_int_denoise_cutoff, ps.max_noise_frag_ratio, ps.max_noise_rsd,
                       ps.max_frag_reserved, ps.use_all_frag)
    # generate candidate formula space
    gen_candidate_formula(mf, ps.ppm, ps.ms1_tol, ps.ms2_tol, ps.db_mode, ps.ele_lower, ps.ele_upper,
                          ps.max_isotope_cnt, global_dict)
    return mf


# test
if __name__ == '__main__':
    #########################################
    buddy_param_set = BuddyParamSet(ms1_tol=5, ms2_tol=10, parallel=True, process_num=8,
                                    timeout_secs=2, halogen=True, max_frag_reserved=50)

    buddy = Buddy(buddy_param_set)
    # buddy.load_mgf("/Users/philip/Documents/test_data/mgf/test.mgf")
    buddy.load_mgf('/Users/philip/Documents/projects/collab/martijn_iodine/Iodine_query_refined.mgf')
    # buddy.load_usi(["mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00005467952",
    #                 "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00005716808"])

    # add ms1 data
    from msbuddy.base import Spectrum

    # buddy.data[0].ms1_raw = Spectrum(mz_array=np.array([540.369, 541.369]),
    #                                  int_array=np.array([100, 28]))
    # buddy.data[1].ms1_raw = Spectrum(mz_array=np.array([buddy.data[1].mz, buddy.data[1].mz + 1]),
    #                                  int_array=np.array([100, 25]))

    # test adduct
    # buddy.load_mgf("/Users/philip/Documents/test_data/mgf/na_adduct.mgf")

    # buddy.data = buddy.data[:50]

    buddy.annotate_formula()
    result_summary_ = buddy.get_summary()
    print(result_summary_)
    print('done')
