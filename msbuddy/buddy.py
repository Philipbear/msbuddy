import logging
import pathlib
import sys
from multiprocessing import Pool, cpu_count
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
from timeout_decorator import timeout
from tqdm import tqdm

from msbuddy.base import MetaFeature
from msbuddy.gen_candidate import gen_candidate_formula, assign_subformula_cand_form
from msbuddy.load import init_db, load_usi, load_mgf
from msbuddy.ml import pred_formula_feasibility, pred_formula_prob, pred_form_feasibility_single, calc_fdr
from msbuddy.query import query_neutral_mass
from msbuddy.api import form_arr_to_str

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
                 n_cpu: int = -1,
                 timeout_secs: float = 300,
                 batch_size: int = 1000,
                 top_n_candidate: int = 500,
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
                 max_noise_frag_ratio: float = 0.90, max_noise_rsd: float = 0.20,
                 max_frag_reserved: int = 50,
                 use_all_frag: bool = False):
        """
        :param ppm: whether ppm is used for m/z tolerance
        :param ms1_tol: MS1 m/z tolerance
        :param ms2_tol: MS2 m/z tolerance
        :param halogen: whether to include halogen atoms; if False, ranges of F, Cl, Br, I will be set to (0, 0)
        :param parallel: whether to use parallel processing
        :param n_cpu: number of CPU cores used for parallel processing; if -1, all available cores will be used
        :param timeout_secs: timeout in seconds
        :param batch_size: batch size for formula annotation; a larger batch size takes more memory
        :param top_n_candidate: top N formula candidates reserved prior to subformula assignment
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
        if n_cpu > cpu_count() or n_cpu <= 0:
            self.n_cpu = cpu_count()
            if self.parallel:
                logging.info(f"Processing core number is set to {self.n_cpu}.")
        else:
            self.n_cpu = n_cpu

        if timeout_secs <= 0:
            logging.warning("Timeout is set to 300 seconds.")
            self.timeout_secs = 300
        self.timeout_secs = timeout_secs
        if self.parallel:
            self.timeout_secs += 20  # add 20 seconds for db initialization

        if batch_size <= 1:
            self.batch_size = 1000
            logging.warning(f"Batch size is set to {self.batch_size}.")
        else:
            self.batch_size = int(batch_size)

        if top_n_candidate <= 1:
            self.top_n_candidate = 500
            logging.warning(f"Top N candidate is set to {self.top_n_candidate}.")
        else:
            self.top_n_candidate = int(top_n_candidate)

        self.ele_lower = np.array([c_range[0], h_range[0], br_range[0], cl_range[0], f_range[0], i_range[0],
                                   0, n_range[0], 0, o_range[0], p_range[0], s_range[0]], dtype=np.int16)
        self.ele_upper = np.array([c_range[1], h_range[1], br_range[1], cl_range[1], f_range[1], i_range[1],
                                   0, n_range[1], 0, o_range[1], p_range[1], s_range[1]], dtype=np.int16)
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
            self.max_noise_frag_ratio = 0.90
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

        logging.info("msbuddy: molecular formula annotation in MS-based metabolomics. maintained by Shipei Xing.")
        logging.info("Initializing...")
        if param_set is None:
            self.param_set = BuddyParamSet()  # default parameter set
        else:
            self.param_set = param_set  # customized parameter set

        global shared_data_dict  # Declare it as a global variable
        shared_data_dict = init_db(self.param_set.db_mode)  # database initialization

        self.data = None  # List[MetabolicFeature]

    def update_param_set(self, new_param_set: BuddyParamSet):
        self.param_set = new_param_set
        global shared_data_dict  # Declare it as a global variable
        shared_data_dict = init_db(self.param_set.db_mode)  # database initialization

    def load_usi(self, usi_list: Union[str, List[str]],
                 adduct_list: Union[None, str, List[str]] = None):
        self.data = load_usi(usi_list, adduct_list)

    def load_mgf(self, file_path):
        self.data = load_mgf(file_path)

    def add_data(self, data: List[MetaFeature]):
        """
        add customized data
        :param data: metabolic feature list
        """
        self.data = data

    def clear_data(self):
        """
        clear loaded data
        :return: None
        """
        self.data = None

    def preprocess_and_generate_candidate_formula(self, batch_start_idx: int = 0, batch_end_idx: int = None):
        """
        preprocess data and generate candidate formula space
        :param batch_start_idx: start index of batch
        :param batch_end_idx: end index of batch
        :return: None. Update self.data
        """

        @timeout(self.param_set.timeout_secs)
        def _preprocess_and_gen_cand_nonparallel(meta_feature: MetaFeature, ps: BuddyParamSet) -> MetaFeature:
            """
            a wrapper function for data preprocessing and candidate formula space generation
            :param meta_feature: MetaFeature object
            :param ps: Buddy parameter set
            :return: MetaFeature object
            """
            mf = _generate_candidate_formula(meta_feature, ps, shared_data_dict)
            return mf

        batch_data = self.data[batch_start_idx:batch_end_idx]
        modified_mf_ls = []  # modified metabolic feature list, containing annotated results

        # data preprocessing and candidate space generation
        if self.param_set.parallel:
            with Pool(processes=int(self.param_set.n_cpu), initializer=init_pool,
                      initargs=(shared_data_dict,)) as pool:
                async_results = [pool.apply_async(_preprocess_and_gen_cand_parallel,
                                                  (mf, self.param_set)) for mf in batch_data]
                # Initialize tqdm progress bar

                pbar = tqdm(total=len(batch_data), colour="green", desc="Candidate space generation",
                            file=sys.stdout)
                for i, async_result in enumerate(async_results):
                    pbar.update(1)  # Update tqdm progress bar
                    try:
                        modified_mf = async_result.get(timeout=self.param_set.timeout_secs)
                        modified_mf_ls.append(modified_mf)
                    except:
                        mf = batch_data[i]
                        logging.warning(f"Timeout for spectrum {mf.identifier}, mz={mf.mz}, rt={mf.rt}, skipped.")
                        modified_mf_ls.append(mf)
            pbar.close()  # Close tqdm progress bar
            del async_results
        else:
            # normal loop, timeout implemented using timeout_decorator
            for mf in tqdm(batch_data, file=sys.stdout, colour="green", desc="Candidate space generation"):
                modified_mf = _preprocess_and_gen_cand_nonparallel(mf, self.param_set)
                try:
                    modified_mf = _preprocess_and_gen_cand_nonparallel(mf, self.param_set)
                    modified_mf_ls.append(modified_mf)
                except:
                    logging.warning(f"Timeout for spectrum {mf.identifier}, mz={mf.mz}, rt={mf.rt}, skipped.")
                    modified_mf_ls.append(mf)

        # update data
        self.data[batch_start_idx:batch_end_idx] = modified_mf_ls
        del modified_mf_ls

    def assign_subformula_annotation(self, batch_start_idx: int = 0, batch_end_idx: int = None):
        """
        assign subformula annotation for loaded data, no timeout implemented
        :param batch_start_idx: start index of batch
        :param batch_end_idx: end index of batch
        :return: None. Update self.data
        """
        batch_data = self.data[batch_start_idx:batch_end_idx]
        modified_mf_ls = []  # modified metabolic feature list

        if self.param_set.parallel:
            with Pool(processes=int(self.param_set.n_cpu)) as pool:
                async_results = [pool.apply_async(_gen_subformula,
                                                  (mf, self.param_set)) for mf in batch_data]

                pbar = tqdm(total=len(batch_data), colour="green", desc="Subformula assignment: ", file=sys.stdout)
                for i, async_result in enumerate(async_results):
                    pbar.update(1)  # Update tqdm progress bar
                    modified_mf = async_result.get()
                    modified_mf_ls.append(modified_mf)
            pbar.close()  # Close tqdm progress bar
            del async_results
        else:
            # normal loop
            for mf in tqdm(batch_data, desc="Subformula assignment: ", file=sys.stdout, colour="green"):
                modified_mf = _gen_subformula(mf, self.param_set)
                modified_mf_ls.append(modified_mf)

        # update data
        self.data[batch_start_idx:batch_end_idx] = modified_mf_ls
        del modified_mf_ls

    def annotate_formula(self):
        """
        annotate formula for loaded data
        pipeline: data preprocessing -> formula candidate space generation -> ml model A
        -> subformula annotation -> ml model B -> FDR calculation
        :return: None. Update self.data
        """
        n_batch = self.__annotate_formula_prepare()

        # loop over batches
        for n in range(n_batch):
            self.__annotate_formula_main_batch(n, n_batch)

        logging.info("Job finished.")

    def annotate_formula_cmd(self, output_path: pathlib.Path, write_details: bool = False):
        """
        annotate formula for loaded data, command line version
        write out summary results to a csv file, clear computed data after annotation to save memory
        :param output_path: output path
        :param write_details: whether to write out detailed results
        :return: None
        """
        n_batch = self.__annotate_formula_prepare()
        output_path.mkdir(parents=True, exist_ok=True)

        # summary results DataFrame
        result_summary_df = pd.DataFrame(columns=['identifier', 'mz', 'rt', 'adduct', 'formula_rank_1', 'estimated_fdr',
                                                  'formula_rank_2', 'formula_rank_3'])
        # loop over batches
        for n in range(n_batch):
            start_idx, end_idx = self.__annotate_formula_main_batch(n, n_batch)
            result_summary_df = self.__write_batch_results(output_path, write_details, start_idx, end_idx,
                                                           result_summary_df)
            # clear computed data to save memory, convert to None of the same size
            self.data[start_idx:end_idx] = [None] * (end_idx - start_idx)

        result_summary_df.to_csv(output_path / 'buddy_result_summary.tsv', sep="\t", index=False)
        logging.info("Job finished.")

    def __annotate_formula_prepare(self) -> int:
        """
        prepare for formula annotation
        :return: batch number
        """
        # select MetaFeatures with precursor 1 < mass < 1500
        self.data = [mf for mf in self.data if 1 < mf.mz < 1500]

        if not self.data:
            raise ValueError("No data loaded.")
        logging.info(f"Total {len(self.data)} spectra loaded.")

        if self.param_set.parallel:
            # parallel processing
            logging.info(f"Parallel processing with {self.param_set.n_cpu} processes.")

        # batches
        n_batch = int(np.ceil(len(self.data) / self.param_set.batch_size))
        logging.info(f"Total {n_batch} batches.")

        return n_batch

    def __annotate_formula_main_batch(self, n: int, n_batch: int) -> Tuple[int, int]:
        """
        annotate formula for batch data N
        :param n: batch number
        :param n_batch: total batch number
        :return: start index and end index of batch
        """
        tqdm.write(f"Batch {n + 1}/{n_batch}:")
        # get batch data
        start_idx, end_idx = _get_batch(self.data, self.param_set.batch_size, n)

        # data preprocessing and candidate space generation
        self.preprocess_and_generate_candidate_formula(start_idx, end_idx)

        # ml_a feature generation + prediction, retain top candidates
        tqdm.write("Formula feasibility assessment...")
        pred_formula_feasibility(self.data, start_idx, end_idx, self.param_set.top_n_candidate,
                                 self.param_set.db_mode, shared_data_dict)

        # assign subformula annotation
        self.assign_subformula_annotation(start_idx, end_idx)

        # ml_b feature generation + prediction
        tqdm.write("Formula probability prediction...")
        pred_formula_prob(self.data, start_idx, end_idx, self.param_set, shared_data_dict)

        # FDR calculation
        calc_fdr(self.data, start_idx, end_idx)

        return start_idx, end_idx

    def __write_batch_results(self, output_path: pathlib.Path, write_details: bool,
                              start_idx: int, end_idx: int, result_df: pd.DataFrame) -> pd.DataFrame:
        """
        write out batch results
        :param output_path: output path
        :param write_details: whether to write out detailed results
        :param start_idx: start index of batch
        :param end_idx: end index of batch
        :param result_df: result summary DataFrame
        :return: updated summary results DataFrame
        """
        batch_data = self.data[start_idx:end_idx]

        # update summary results DataFrame
        for mf in batch_data:
            individual_result = mf.summarize_result()
            result_df = result_df.append({
                'identifier': mf.identifier,
                'mz': mf.mz,
                'rt': mf.rt if mf.rt else 'None',
                'adduct': mf.adduct.string,
                'formula_rank_1': individual_result['formula_rank_1'],
                'estimated_fdr': individual_result['estimated_fdr'],
                'formula_rank_2': individual_result['formula_rank_2'],
                'formula_rank_3': individual_result['formula_rank_3']
            }, ignore_index=True)

        # write out detailed results
        if write_details:
            for mf in batch_data:
                # make a directory for each mf
                # replace '/' with '_' in the identifier, remove special characters
                _id = str(mf.identifier).replace('/', '_').replace(':', '_').replace(' ', '_').strip()
                folder_name = _id + '_mz' + str(round(mf.mz, 4)) + '_rt'
                folder_name += str(round(mf.rt, 2)) if mf.rt else 'None'
                mf_path = pathlib.Path(output_path / folder_name)
                mf_path.mkdir(parents=True, exist_ok=True)

                # write the tsv file containing all the candidate formulas
                all_candidates_df = pd.DataFrame(columns=['rank', 'formula', 'formula_feasibility',
                                                          'ms1_isotope_similarity', 'explained_ms2_peak',
                                                          'total_valid_ms2_peak', 'estimated_prob',
                                                          'normalized_estimated_prob', 'estimated_fdr'])
                for m, cf in enumerate(mf.candidate_formula_list):
                    # string for explained ms2 peak
                    if mf.ms2_processed:
                        if cf.ms2_raw_explanation:
                            exp_ms2_peak = len(cf.ms2_raw_explanation)
                        else:
                            exp_ms2_peak = '0'
                    else:
                        exp_ms2_peak = 'None'
                    all_candidates_df = all_candidates_df.append({
                        'rank': m,
                        'formula': cf.formula.__str__(),
                        'formula_feasibility': cf.ml_a_prob,
                        'ms1_isotope_similarity': cf.ms1_isotope_similarity if cf.ms1_isotope_similarity else 'None',
                        'explained_ms2_peak': exp_ms2_peak,
                        'total_valid_ms2_peak': len(mf.ms2_processed) if mf.ms2_processed else 'None',
                        'estimated_prob': cf.estimated_prob,
                        'normalized_estimated_prob': cf.normed_estimated_prob,
                        'estimated_fdr': cf.estimated_fdr
                    }, ignore_index=True)
                all_candidates_df.to_csv(mf_path / 'formula_results.tsv', sep="\t", index=False)

                # write the tsv file containing preprocessed spectrum
                if mf.ms1_processed:
                    ms1_df = pd.DataFrame(columns=['raw_idx', 'mz', 'intensity'])
                    for m in range(len(mf.ms1_processed)):
                        ms1_df = ms1_df.append({
                            'raw_idx': mf.ms1_processed.idx_array[m],
                            'mz': mf.ms1_processed.mz_array[m],
                            'intensity': mf.ms1_processed.int_array[m]
                        }, ignore_index=True)
                    ms1_df.to_csv(mf_path / 'ms1_preprocessed.tsv', sep="\t", index=False)
                if mf.ms2_processed:
                    ms2_df = pd.DataFrame(columns=['raw_idx', 'mz', 'intensity'])
                    for m in range(len(mf.ms2_processed)):
                        ms2_df = ms2_df.append({
                            'raw_idx': mf.ms2_processed.idx_array[m],
                            'mz': mf.ms2_processed.mz_array[m],
                            'intensity': mf.ms2_processed.int_array[m]
                        }, ignore_index=True)
                    ms2_df.to_csv(mf_path / 'ms2_preprocessed.tsv', sep="\t", index=False)

        return result_df

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

    def mass_to_formula(self, mass: float, mz_tol: float, ppm: bool) -> List[str]:
        """
        convert mass to formula, return list of formula strings
        :param mass: target mass, should be <1500
        :param mz_tol: mz tolerance
        :param ppm: whether mz_tol is in ppm
        :return: list of formula strings
        """
        formulas = query_neutral_mass(mass, mz_tol, ppm, shared_data_dict)
        return [form_arr_to_str(f.array) for f in formulas]

    def predict_formula_feasibility(self, formula: Union[str, np.array]) -> Union[float, None]:
        """
        predict formula feasibility score for a single formula
        :param formula: formula string or array
        :return: feasibility score (float) or None
        """
        return pred_form_feasibility_single(formula, shared_data_dict)


def _get_batch(data: List[MetaFeature], batch_size: int, n: int):
    """
    get batch data
    :param data: data list
    :param batch_size: batch size
    :param n: batch number
    :return: batch data
    """
    start_idx = n * batch_size
    end_idx = min((n + 1) * batch_size, len(data))
    return start_idx, end_idx


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
    mf = _generate_candidate_formula(meta_feature, ps, shared_data_dict)
    return mf


def _gen_subformula(mf: MetaFeature, ps: BuddyParamSet) -> MetaFeature:
    """
    a wrapper function for subformula generation
    :param mf: MetaFeature object
    :param ps: Buddy parameter set
    :return: MetaFeature object
    """
    if not mf.ms2_processed:
        return mf

    if not mf.candidate_formula_list:
        return mf

    mf = assign_subformula_cand_form(mf, ps.ppm, ps.ms2_tol)
    return mf


def _generate_candidate_formula(mf: MetaFeature, ps: BuddyParamSet, global_dict) -> MetaFeature:
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
    import time
    #########################################
    buddy_param_set = BuddyParamSet(ms1_tol=5, ms2_tol=10, parallel=False, n_cpu=4, batch_size=1000,
                                    timeout_secs=300, halogen=True, max_frag_reserved=50,
                                    i_range=(0, 20))

    buddy = Buddy(buddy_param_set)
    # buddy.load_mgf("/Users/shipei/Documents/test_data/mgf/test.mgf")
    buddy.load_mgf('/Users/shipei/Documents/projects/collab/martijn_iodine/Iodine_query_refined.mgf')
    # buddy.load_usi(["mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00005467952",
    #                 "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00005716808"])
    #
    # # add ms1 data
    # from msbuddy.base import Spectrum
    #
    # buddy.data[0].ms1_raw = Spectrum(mz_array=np.array([540.369, 541.369]),
    #                                  int_array=np.array([100, 28]))
    # buddy.data[1].ms1_raw = Spectrum(mz_array=np.array([buddy.data[1].mz, buddy.data[1].mz + 1]),
    #                                  int_array=np.array([100, 25]))

    # test adduct
    # buddy.load_mgf("/Users/philip/Documents/test_data/mgf/na_adduct.mgf")

    buddy.data = buddy.data[:10]

    buddy.annotate_formula_cmd(pathlib.Path('/Users/shipei/Documents/test_data/buddy_result'), write_details=True)

    # start_time = time.time()
    # buddy.annotate_formula()
    # result_summary_ = buddy.get_summary()
    # # print(result_summary_)
    # print(f"Total time: {time.time() - start_time} seconds.")
    print('done')
