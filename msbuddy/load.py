import logging
from json import loads as loads
from pathlib import Path
from typing import List, Union

import numpy as np
from gdown import download as download
from joblib import load as j_load
from requests import get

from msbuddy.base import MetaFeature, Spectrum

logging.basicConfig(level=logging.INFO)


def check_and_download(url: str, path) -> bool:
    """
    check if the file exists, if not, download from url
    :param url: url to download
    :param path: path to save
    :return: True if success
    """
    if not path.exists() or path.stat().st_size < 10 ** 4:
        download(url, str(path))
        return True


def init_db(db_mode: int) -> dict:
    """
    init databases used in the project
    :param db_mode: 0: basic; 1: halogen
    :return: True if success
    """
    root_path = Path(__file__).parent

    logging.info('Initializing databases...')

    global_dict = dict()
    # load database & models into memory
    global_dict['common_loss_db'] = j_load(root_path / 'data' / 'common_loss.joblib')
    global_dict['common_frag_db'] = j_load(root_path / 'data' / 'common_frag.joblib')
    global_dict['model_a'] = j_load(root_path / 'data' / 'model_a.joblib')
    global_dict['model_a_mean_arr'] = j_load(root_path / 'data' / 'mean_arr.joblib')
    global_dict['model_a_std_arr'] = j_load(root_path / 'data' / 'std_arr.joblib')
    # global_dict['model_b_noms1_ms2'] = j_load(root_path / 'data' / 'model_b_noms1_ms2_2.joblib')

    # check existence of basic_db_mass.joblib, basic_db_formula.joblib
    check_and_download('https://drive.google.com/uc?id=1obPMk9lcfkUpRkeGSkM1s4C9Bzatm1li',
                       root_path / 'data' / 'basic_db_mass.joblib')
    check_and_download('https://drive.google.com/uc?id=155AEYIv5XFBIc7Adpnfi-vH3s47QkbJf',
                       root_path / 'data' / 'basic_db_formula.joblib')

    global_dict['basic_db_mass'] = j_load(root_path / 'data' / 'basic_db_mass.joblib')
    global_dict['basic_db_formula'] = j_load(root_path / 'data' / 'basic_db_formula.joblib')
    global_dict['basic_db_idx'] = j_load(root_path / 'data' / 'basic_db_idx.joblib')

    if db_mode >= 1:
        # check existence of halogen_db_mass.joblib, halogen_db_formula.joblib
        check_and_download('https://drive.google.com/uc?id=1SMhezxtXtjQNj2N8odYWSEufOO_6N1o5',
                           root_path / 'data' / 'halogen_db_mass.joblib')
        check_and_download('https://drive.google.com/uc?id=18G8_qzTXWHDIw9Z9PwvMtjKWOi6FtwDU',
                           root_path / 'data' / 'halogen_db_formula.joblib')

        global_dict['halogen_db_mass'] = j_load(root_path / 'data' / 'halogen_db_mass.joblib')
        global_dict['halogen_db_formula'] = j_load(root_path / 'data' / 'halogen_db_formula.joblib')
        global_dict['halogen_db_idx'] = j_load(root_path / 'data' / 'halogen_db_idx.joblib')

    return global_dict


def load_mgf(file_path) -> List[MetaFeature]:
    """
    read mgf file
    :param file_path: path to mgf file
    :return: list of MetaFeature
    """
    with open(file_path, 'r') as file:
        # create meta_feature_list
        meta_feature_list = []
        cnt = 0
        for line in file:
            # empty line
            _line = line.strip()  # remove leading and trailing whitespace
            if not _line:
                continue
            elif line.startswith('BEGIN IONS'):
                # initialize a new spectrum entry
                mz_arr = np.array([])
                int_arr = np.array([])
                precursor_mz = None
                identifier = None
                charge = None
                pos_mode = None
                ms2_spec = True
                rt = None
                adduct_str = None
            elif line.startswith('END IONS'):
                # create a new MetaFeature
                if precursor_mz is None:
                    raise ValueError('No precursor mz found.')
                if identifier is None:
                    identifier = cnt
                if charge is None:
                    charge = 1 if pos_mode else -1
                elif charge == 0:
                    charge = 1 if pos_mode else -1
                else:
                    charge = abs(charge) if pos_mode else -abs(charge)

                # create MetaFeature object if the same identifier does not exist
                mf_idx = None
                for idx, mf in enumerate(meta_feature_list):
                    if mf.identifier == identifier:
                        mf_idx = idx
                        break

                # if the same identifier exists, add to the existing MetaFeature
                if mf_idx is not None:
                    if ms2_spec:
                        meta_feature_list[mf_idx].ms2_raw = Spectrum(mz_arr, int_arr)
                    else:
                        meta_feature_list[mf_idx].ms1_raw = Spectrum(mz_arr, int_arr)
                    continue
                # if the same identifier does not exist, create a new MetaFeature
                else:
                    mf = MetaFeature(mz=precursor_mz,
                                     charge=charge,
                                     rt=rt,
                                     adduct=adduct_str,
                                     ms2=Spectrum(mz_arr, int_arr),
                                     identifier=identifier)
                    meta_feature_list.append(mf)
                cnt += 1
                continue
            else:
                # if line contains '=', it is a key-value pair
                if '=' in _line:
                    # split by first '=', in case of multiple '=' in the line
                    key, value = _line.split('=', 1)
                    # if key (into all upper case) is 'PEPMASS', it is precursor mz
                    if key.upper() == 'PEPMASS':
                        precursor_mz = float(value)
                    # if key is 'CHARGE' and charge is not set, it is charge
                    elif key.upper() == 'CHARGE':
                        if '+' in value:
                            pos_mode = True
                            value = value.replace('+', '')
                            charge = int(value)
                        elif '-' in value:
                            pos_mode = False
                            value = value.replace('-', '')
                            charge = -int(value)
                    # if key is 'ION', it is adduct type
                    elif key.upper() == 'ION':
                        adduct_str = value
                    # if key is 'IONMODE', it is ion mode
                    elif key.upper() == 'IONMODE':
                        if value.upper() == 'POSITIVE':
                            pos_mode = True
                        elif value.upper() == 'NEGATIVE':
                            pos_mode = False
                    # if key is 'MSLEVEL', it is ms level
                    elif key.upper() == 'MSLEVEL':
                        if value == '1':
                            ms2_spec = False
                    # if key is 'RTINSECONDS', it is rt
                    elif key.upper() == 'RTINSECONDS' and value != '':
                        rt = float(value)
                    elif key.upper() == 'RTINMINUTES' and value != '':
                        rt = float(value) * 60
                else:
                    # if no '=', it is a spectrum pair, split by '\t' or ' '
                    this_mz, this_int = _line.split()
                    mz_arr = np.append(mz_arr, float(this_mz))
                    int_arr = np.append(int_arr, float(this_int))

    return meta_feature_list


def _load_usi(usi: str, adduct: Union[str, None] = None) -> MetaFeature:
    """
    Read from a USI string and return a MetaFeature object.
    The GNPS API is used to get the spectrum from the USI.
    Citation: Universal MS/MS Visualization and Retrieval with the Metabolomics Spectrum Resolver Web Service.
    Wout Bittremieux et al. doi: 10.1101/2020.05.09.085000
    :param usi: USI string
    :param adduct: adduct string
    :return: MetaFeature object
    """
    # get spectrum from USI
    url = 'https://metabolomics-usi.gnps2.org/json/?usi1=' + usi
    response = get(url)
    json_data = loads(response.text)

    # check if the USI is valid
    if 'error' in json_data:
        raise ValueError

    # valid: dict_keys(['n_peaks', 'peaks', 'precursor_charge', 'precursor_mz', 'splash'])
    # ion mode
    charge = json_data['precursor_charge']
    if charge == 0 and adduct:
        pos_mode = str(adduct)[-1] != '-'  # use adduct if charge is 0
        charge = 1 if pos_mode else -1

    ms2_mz = np.array(json_data['peaks'])[:, 0]
    ms2_int = np.array(json_data['peaks'])[:, 1]

    data = MetaFeature(mz=json_data['precursor_mz'],
                       charge=charge,
                       adduct=adduct,
                       ms2=Spectrum(ms2_mz, ms2_int),
                       identifier=usi)
    return data


def load_usi(usi_list: Union[str, List[str]],
             adduct_list: Union[None, str, List[str]] = None) -> List[MetaFeature]:
    """
    Read from a sequence of USI strings and return a list of MetaFeature objects.
    Invalid USI strings are discarded.
    The GNPS API is used to get the spectrum from the USI.
    Citation: Universal MS/MS Visualization and Retrieval with the Metabolomics Spectrum Resolver Web Service.
    Wout Bittremieux et al. doi: 10.1101/2020.05.09.086066.
    See https://ccms-ucsd.github.io/GNPSDocumentation/api/#experimental-or-library-spectrum-by-usi for details.
     ---------------------------------------------------------
    :param usi_list: List of USI string or a single USI string
    :param adduct_list: adduct string, e.g. [M+H]+
    :return: List of MetaFeature objects
    """

    data_list = []

    # if usi_list is a single string, convert it to a list
    if isinstance(usi_list, str):
        usi_list = [usi_list]
        if adduct_list is not None:
            adduct_list = [adduct_list]

    if adduct_list is None:
        adduct_list = [None] * len(usi_list)
    elif len(adduct_list) != len(usi_list):
        logging.warning('adduct_list and usi_list must have the same length. Default adducts are used.')

    # retrieve indices of unique USIs from the list
    seen = {}
    unique_indices = []
    for idx, item in enumerate(usi_list):
        if item not in seen:
            seen[item] = True
            unique_indices.append(idx)
    usi_list_unique = [usi_list[idx] for idx in unique_indices]
    adduct_list_unique = [adduct_list[idx] for idx in unique_indices]

    # load data
    for usi, adduct in zip(usi_list_unique, adduct_list_unique):
        try:
            data_list.append(_load_usi(usi, adduct))
        except:
            logging.warning('Invalid USI: ' + usi)
            continue
    return data_list


# test
if __name__ == '__main__':
    usi_data = load_usi('mzspec:GNPS:null:accession:CCMSLIB00000001500')
