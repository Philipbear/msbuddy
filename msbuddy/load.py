# ==============================================================================
# Copyright (C) 2023 Shipei Xing <s1xing@health.ucsd.edu>
#
# Licensed under the Apache License 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://github.com/Philipbear/msbuddy/blob/main/LICENSE
# ==============================================================================
"""
File: load.py
Author: Shipei Xing
Email: s1xing@health.ucsd.edu
GitHub: Philipbear
Description: load databases and data files
"""

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

current_version = 'v0.2.4'

def check_download_joblibload(url: str, path):
    """
    check if the file exists, if not, download from url, and load
    :param url: url to download
    :param path: path to save
    :return: loaded object
    """
    if not path.exists() or path.stat().st_size < 10 ** 3:
        download(url, str(path))

    return j_load(path)


def init_db() -> dict:
    """
    init databases used in the project
    :return: global_dict
    """
    # get root path
    root_path = Path(__file__).parent

    # create data folder if not exists
    data_path = root_path / 'data'
    data_path.mkdir(parents=True, exist_ok=True)

    global_dict = dict()

    # load common_loss_db, common_frag_db
    db_name = 'common_db_' + current_version + '.joblib'
    global_dict['common_loss_db'], global_dict['common_frag_db'] = (
        check_download_joblibload(
            'https://drive.google.com/uc?id=1mxUeHNYC_XEbDzKAeax2m5Qc-CpqPPzL',
            data_path / db_name))

    # load ml_a
    db_name = 'ml_a_' + current_version + '.joblib'
    global_dict['model_a'], global_dict['model_a_mean_arr'], global_dict['model_a_std_arr'] = (
        check_download_joblibload(
            'https://drive.google.com/uc?id=19-htf-iifTUpAMOSB9DhFs0XkqqW1Gxm',
            data_path / db_name))

    # load ml_b
    db_name = 'ml_b_' + current_version + '.joblib'
    global_dict['model_b_ms1_ms2'], global_dict['model_b_noms1_ms2'], global_dict['model_b_ms1_noms2'], global_dict[
        'model_b_noms1_noms2'], global_dict['model_b_mean_arr'], global_dict['model_b_std_arr'] = (
        check_download_joblibload(
            'https://drive.google.com/uc?id=17fuTj7oZdFu6TRQDUaFvWNrm1UFt1JZG',
            data_path / db_name))

    # formula_db
    db_name = 'formula_db_' + current_version + '.joblib'
    basic_db, halogen_db = (
        check_download_joblibload(
            'https://drive.google.com/uc?id=17UjBzBe5LBzT_c8lEE-6PVBJO6xP3WSv',
            data_path / db_name))

    global_dict['basic_db_mass'], global_dict['basic_db_formula'], global_dict['basic_db_idx'] = basic_db
    global_dict['halogen_db_mass'], global_dict['halogen_db_formula'], global_dict['halogen_db_idx'] = halogen_db

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

                # # if no peaks found, skip
                # if mz_arr.size == 0:
                #     continue

                # create MetaFeature object if the same identifier does not exist
                mf_idx = None
                for idx, mf in enumerate(meta_feature_list):
                    if mf.identifier == identifier:
                        mf_idx = idx
                        break

                # if the same identifier exists, add to the existing MetaFeature
                if mf_idx is not None:
                    if ms2_spec and meta_feature_list[mf_idx].ms2_raw is None:
                        meta_feature_list[mf_idx].ms2_raw = Spectrum(mz_arr, int_arr) if mz_arr.size > 0 else None
                    elif ms2_spec is False and meta_feature_list[mf_idx].ms1_raw is None:
                        meta_feature_list[mf_idx].ms1_raw = Spectrum(mz_arr, int_arr) if mz_arr.size > 0 else None
                    continue
                # if the same identifier does not exist, create a new MetaFeature
                else:
                    mf = MetaFeature(mz=precursor_mz,
                                     charge=charge,
                                     rt=rt,
                                     adduct=adduct_str,
                                     ms2=Spectrum(mz_arr, int_arr) if mz_arr.size > 0 else None,
                                     identifier=identifier)
                    meta_feature_list.append(mf)
                    cnt += 1
                continue
            else:
                # if line contains '=', it is a key-value pair
                if '=' in _line:
                    # split by first '=', in case of multiple '=' in the line
                    key, value = _line.split('=', 1)
                    key, value = key.strip(), value.strip()
                    # if key (into all upper case) is 'PEPMASS', it is precursor mz
                    if key.upper() in ['PEPMASS', 'PRECURSOR_MZ']:
                        precursor_mz = float(value)
                    # if key is 'TITLE', it is identifier
                    elif key.upper() in ['TITLE', 'SPECTRUMID', 'SPECTRUM_ID']:
                        identifier = value.strip()
                    # if key is 'CHARGE' and charge is not set, it is charge
                    elif key.upper() == 'CHARGE':
                        if '-' in value:
                            pos_mode = False
                            value = value.replace('-', '')
                            charge = -int(value)
                        else:
                            pos_mode = True
                            value = value.replace('+', '')
                            charge = int(value)
                    # if key is 'ION', it is adduct type
                    elif key.upper() == 'ION':
                        adduct_str = value
                    # if key is 'IONMODE', it is ion mode
                    elif key.upper() == 'IONMODE':
                        if value.upper() in ['POSITIVE', 'POS', 'P']:
                            pos_mode = True
                        elif value.upper() in ['NEGATIVE', 'NEG', 'N']:
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

    # get adduct
    if adduct == '':
        adduct = None

    # valid: dict_keys(['n_peaks', 'peaks', 'precursor_charge', 'precursor_mz', 'splash'])
    # ion mode
    charge = json_data['precursor_charge']
    if charge == 0 and adduct is not None:
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
            adduct_list = [adduct_str.strip() for adduct_str in adduct_list]

    if adduct_list is None:
        adduct_list = [None] * len(usi_list)
    elif len(adduct_list) != len(usi_list):
        logging.warning('adduct_list and usi_list must have the same length. Default adducts are used.')

    usi_list = [usi.strip() for usi in usi_list]

    # retrieve indices of unique USIs from the list
    seen = {}
    unique_indices = []
    for idx, item in enumerate(usi_list):
        if item not in seen:
            seen[item] = True
            unique_indices.append(idx)
        else:
            logging.warning('Duplicate USI: ' + item + '. Only the first occurrence is used.')
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
# if __name__ == '__main__':
    #################
    # init_db()

    # import joblib
    # compile all these databases

    # basic_db_mass = j_load('../db_prep/basic_db_mass.joblib')
    # basic_db_formula = j_load('../db_prep/basic_db_formula.joblib')
    # basic_db_idx = j_load('../db_prep/basic_db_idx.joblib')
    # halogen_db_mass = j_load('../db_prep/halogen_db_mass.joblib')
    # halogen_db_formula = j_load('../db_prep/halogen_db_formula.joblib')
    # halogen_db_idx = j_load('../db_prep/halogen_db_idx.joblib')
    #
    # basic_db = [basic_db_mass, basic_db_formula, basic_db_idx]
    # halogen_db = [halogen_db_mass, halogen_db_formula, halogen_db_idx]
    # formula_db = [basic_db, halogen_db]
    #
    # joblib.dump(formula_db, "data/formula_db.joblib")

    # common_db
    # common_loss_db = j_load('data/common_loss.joblib')
    # common_frag_db = j_load('data/common_frag.joblib')
    #
    # common_db = [common_loss_db, common_frag_db]
    # joblib.dump(common_db, "data/common_db.joblib")

    # # model_a
    # model_a = j_load('data/model_a.joblib')
    # model_a_mean_arr = j_load('data/ml_a_mean_arr.joblib')
    # model_a_std_arr = j_load('data/ml_a_std_arr.joblib')
    #
    # ml_a = [model_a, model_a_mean_arr, model_a_std_arr]
    # joblib.dump(ml_a, "data/ml_a.joblib")

    # # model_b
    # model_b_ms1_ms2 = j_load('data/ml_b_ms1_ms2.joblib')
    # model_b_noms1_ms2 = j_load('data/ml_b_noms1_ms2.joblib')
    # model_b_ms1_noms2 = j_load('data/ml_b_ms1_noms2.joblib')
    # model_b_noms1_noms2 = j_load('data/ml_b_noms1_noms2.joblib')
    # model_b_mean_arr = j_load('data/ml_b_mean_arr.joblib')
    # model_b_std_arr = j_load('data/ml_b_std_arr.joblib')
    #
    # ml_b = [model_b_ms1_ms2, model_b_noms1_ms2, model_b_ms1_noms2, model_b_noms1_noms2,
    #         model_b_mean_arr, model_b_std_arr]
    # joblib.dump(ml_b, "data/ml_b.joblib")
