# ==============================================================================
# Copyright (C) 2024-2026 Shipei Xing <sxing@fudan.edu.cn>
#
# Licensed under the Apache License 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://github.com/Philipbear/msbuddy/blob/main/LICENSE
# ==============================================================================
"""
File: load.py
Author: Shipei Xing
Email: sxing@fudan.edu.cn
GitHub: Philipbear
Description: load databases and data files
"""

import errno
import logging
import os
from json import loads as loads
from pathlib import Path
from typing import List, Union

import numpy as np
from gdown import download as download
from joblib import load as j_load
from requests import get

from msbuddy.base import MetaFeature, Spectrum, _validate_adduct_elements
from msbuddy.utils import UnsupportedElementError

logging.basicConfig(level=logging.INFO)

current_db_version = 'v0.2.4'
current_model_version = 'v0.3.0'

DATA_FILES = {
    'common_db': {
        'filename': 'common_db_' + current_db_version + '.joblib',
        'url': (
            'https://github.com/Philipbear/msbuddy/releases/download/'
            'msbuddy_data_v0.2.4/common_db_v0.2.4.joblib'),
    },
    'formula_db': {
        'filename': 'formula_db_' + current_db_version + '.joblib',
        'url': (
            'https://github.com/Philipbear/msbuddy/releases/download/'
            'msbuddy_data_v0.2.4/formula_db_v0.2.4.joblib'),
    },
    'ml_model': {
        'filename': 'ml_' + current_model_version + '.joblib',
        'url': (
            'https://github.com/Philipbear/msbuddy/releases/download/'
            'msbuddy_data_v0.3.0/ml_v0.3.0.joblib'),
    },
}


def resolve_data_dir(data_dir=None) -> Path:
    """Resolve the data directory while preserving the original default."""
    if data_dir is not None:
        return Path(data_dir).expanduser()
    return Path(__file__).parent / 'data'


def _resolve_data_paths(data_dir=None, data_files=None) -> dict:
    """Resolve default and user-provided data file names or paths."""
    data_path = resolve_data_dir(data_dir)
    overrides = {} if data_files is None else dict(data_files)

    unknown = sorted(set(overrides) - set(DATA_FILES))
    if unknown:
        valid = ', '.join(sorted(DATA_FILES))
        raise ValueError(
            f"Unknown data_files key(s): {', '.join(unknown)}. "
            f"Valid keys are: {valid}.")

    paths = {}
    for name, info in DATA_FILES.items():
        value = overrides.get(name, info['filename'])
        if value is None:
            raise ValueError(f"data_files['{name}'] cannot be None.")

        path = Path(value).expanduser()
        if not path.is_absolute():
            path = data_path / path
        paths[name] = path

    return paths


def get_data_files(data_dir=None, data_files=None) -> dict:
    """
    Return the paths and download URLs for the files used by msbuddy.

    Values in ``data_files`` may be file names relative to ``data_dir`` or
    absolute paths. Supported keys are ``common_db``, ``formula_db`` and
    ``ml_model``. This function does not create directories or download files.
    """
    paths = _resolve_data_paths(data_dir, data_files)
    return {
        name: {
            'path': path,
            'url': DATA_FILES[name]['url'],
            'exists': path.is_file(),
        }
        for name, path in paths.items()
    }


def _permission_error(path: Path) -> PermissionError:
    return PermissionError(
        f"msbuddy could not read or write its data file:\n  {path}\n\n"
        "The default data directory is inside the installed msbuddy package "
        "for backwards compatibility. A system installation may be owned by "
        "root. Choose a writable directory using one of these options:\n\n"
        "Python:\n"
        "  from msbuddy import Msbuddy, MsbuddyConfig, download_data\n"
        "  download_data('~/msbuddy-data')\n"
        "  config = MsbuddyConfig(data_dir='~/msbuddy-data')\n"
        "  engine = Msbuddy(config)\n\n"
        "Command line:\n"
        "  msbuddy --download-data --data-dir ~/msbuddy-data\n"
        "  msbuddy --mgf input.mgf --data-dir ~/msbuddy-data")


def _missing_custom_file_error(path: Path) -> FileNotFoundError:
    return FileNotFoundError(
        f"The configured msbuddy data file is missing or incomplete:\n"
        f"  {path}\n\n"
        "Provide a compatible file at that path, remove its data_files "
        "override to use the standard filename and automatic download, or "
        "call download_data(...) to download the standard release asset to "
        "the configured path.")


def _file_is_ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= 10 ** 3
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
            raise _permission_error(path) from exc
        raise


def _download_if_missing(url: str, path: Path) -> Path:
    if _file_is_ready(path):
        return path

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        download(url, str(path))
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
            raise _permission_error(path) from exc
        raise

    if not _file_is_ready(path):
        if path.parent.exists() and not os.access(path.parent, os.W_OK):
            raise _permission_error(path)
        raise RuntimeError(f'Failed to download msbuddy data file: {path}')

    return path


def download_data(data_dir=None, data_files=None, include_model=True) -> dict:
    """
    Download the standard msbuddy files to the selected local paths.

    Custom file names or paths can be supplied through ``data_files``. If a
    custom file already exists, it is used as-is. If it is missing, the
    standard msbuddy release file is downloaded to that path.
    """
    files = get_data_files(data_dir, data_files)
    selected = ['common_db', 'formula_db']
    if include_model:
        selected.append('ml_model')

    for name in selected:
        _download_if_missing(files[name]['url'], files[name]['path'])

    return {name: files[name]['path'] for name in selected}


def check_download_joblibload(url: str, path, download_missing=True):
    """
    check if the file exists, if not, optionally download from url, and load
    :param url: url to download
    :param path: path to save
    :param download_missing: whether a missing file may be downloaded
    :return: loaded object
    """
    path = Path(path)
    if download_missing:
        path = _download_if_missing(url, path)
    elif not _file_is_ready(path):
        raise _missing_custom_file_error(path)

    try:
        return j_load(path)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
            raise _permission_error(path) from exc
        raise


def init_db(data_dir=None, data_files=None) -> dict:
    """
    init databases used in the project
    :return: global_dict
    """
    files = _resolve_data_paths(data_dir, data_files)
    custom_files = set(data_files or {})
    global_dict = dict()

    # load common_loss_db, common_frag_db
    global_dict['common_loss_db'], global_dict['common_frag_db'] = (
        check_download_joblibload(
            DATA_FILES['common_db']['url'], files['common_db'],
            download_missing='common_db' not in custom_files))

    # formula_db
    basic_db, halogen_db = (
        check_download_joblibload(
            DATA_FILES['formula_db']['url'], files['formula_db'],
            download_missing='formula_db' not in custom_files))

    global_dict['basic_db_mass'], global_dict['basic_db_formula'], global_dict['basic_db_idx'] = basic_db
    global_dict['halogen_db_mass'], global_dict['halogen_db_formula'], global_dict['halogen_db_idx'] = halogen_db

    return global_dict


def init_ml_models(global_dict, data_dir=None, data_files=None) -> dict:
    """
    init ml models if not exists
    :return: global_dict
    """

    # check if ml models exist
    if 'model_ms1_ms2' in global_dict:
        return global_dict

    files = _resolve_data_paths(data_dir, data_files)
    custom_files = set(data_files or {})

    # load ml
    (global_dict['model_ms1_ms2'], global_dict['model_noms1_ms2'], global_dict['model_ms1_noms2'], global_dict[
        'model_noms1_noms2'], global_dict['platt_a_0'], global_dict['platt_b_0'], global_dict['platt_a_1'],
     global_dict['platt_b_1'], global_dict['platt_a_2'], global_dict['platt_b_2'], global_dict['platt_a_3'],
     global_dict['platt_b_3']) = (
        check_download_joblibload(
            DATA_FILES['ml_model']['url'], files['ml_model'],
            download_missing='ml_model' not in custom_files))

    return global_dict


def load_mgf(file_path) -> List[MetaFeature]:
    """
    read mgf file
    :param file_path: path to mgf file
    :return: list of MetaFeature
    """
    with open(file_path, 'r') as file:
        # Keep insertion order in the result while finding matching MS1/MS2
        # entries without scanning all previously loaded features.
        meta_feature_list = []
        features_by_identifier = {}
        cnt = 0
        for line in file:
            # empty line
            _line = line.strip()  # remove leading and trailing whitespace
            if not _line or _line[0] in '#;!/':
                continue
            elif _line.startswith('BEGIN IONS'):
                # initialize a new spectrum entry
                mz_arr = []
                int_arr = []
                precursor_mz = None
                identifier = None
                charge = None
                pos_mode = None
                ms2_spec = True
                rt = None
                adduct_str = None
            elif _line.startswith('END IONS'):
                # create a new MetaFeature
                if precursor_mz is None:
                    raise ValueError('No precursor mz found.')
                if identifier is None:
                    identifier = cnt
                if pos_mode is None and adduct_str:
                    if adduct_str.endswith('+'):
                        pos_mode = True
                    elif adduct_str.endswith('-'):
                        pos_mode = False
                if pos_mode is None:
                    pos_mode = True
                    logging.warning('No ionization mode found for spectrum %s; assuming positive mode.', identifier)
                if charge is None:
                    charge = 1 if pos_mode else -1
                elif charge == 0:
                    charge = 1 if pos_mode else -1
                else:
                    charge = abs(charge) if pos_mode else -abs(charge)

                # if the same identifier exists, add to the existing MetaFeature
                existing = features_by_identifier.get(identifier)
                if existing is not None:
                    if ms2_spec and existing.ms2_raw is None:
                        existing.ms2_raw = Spectrum(mz_arr, int_arr) if mz_arr else None
                    elif ms2_spec is False and existing.ms1_raw is None:
                        existing.ms1_raw = Spectrum(mz_arr, int_arr) if mz_arr else None
                    continue
                # if the same identifier does not exist, create a new MetaFeature
                else:
                    if ms2_spec:
                        mf = MetaFeature(mz=precursor_mz,
                                         charge=charge,
                                         rt=rt,
                                         adduct=adduct_str,
                                         ms2=Spectrum(mz_arr, int_arr) if mz_arr else None,
                                         identifier=identifier)
                    else:
                        mf = MetaFeature(mz=precursor_mz,
                                         charge=charge,
                                         rt=rt,
                                         adduct=adduct_str,
                                         ms1=Spectrum(mz_arr, int_arr) if mz_arr else None,
                                         identifier=identifier)
                    meta_feature_list.append(mf)
                    features_by_identifier[identifier] = mf
                    cnt += 1
                continue
            else:
                # if line contains '=', it is a key-value pair
                if '=' in _line:
                    # split by first '=', in case of multiple '=' in the line
                    key, value = _line.split('=', 1)
                    key, value = key.strip(), value.strip()
                    # if key (into all upper case) is 'PEPMASS', it is precursor mz
                    if key.upper() in ['PEPMASS', 'PRECURSOR_MZ', 'PRECURSORMZ']:
                        # MGF PEPMASS may contain an optional precursor intensity.
                        precursor_mz = float(value.split()[0])
                    # identifier
                    elif key.upper() in ['TITLE', 'FEATURE_ID', 'SPECTRUMID', 'SPECTRUM_ID']:
                        identifier = value.strip()
                    # if key is 'CHARGE' and charge is not set, it is charge
                    elif key.upper() == 'CHARGE':
                        if '-' in value:
                            pos_mode = False
                            value = value.replace('-', '')
                            charge = -int(value)
                        elif '+' in value:
                            pos_mode = True
                            value = value.replace('+', '')
                            charge = int(value)
                        else:
                            charge = int(value)
                    # if key is 'ION', it is adduct type
                    elif key.upper() in ['ION', 'IONTYPE', 'ION_TYPE', 'ADDUCT', 'ADDUCTTYPE', 'ADDUCT_TYPE']:
                        _validate_adduct_elements(value)
                        adduct_str = value
                    # if key is 'IONMODE', it is ion mode
                    elif key.upper() in ['IONMODE', 'ION_MODE']:
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
                    # Some MGF files include a third peak charge or annotation.
                    parts = _line.split()
                    if len(parts) < 2:
                        raise ValueError(f'Invalid MGF peak line: {_line}')
                    this_mz, this_int = parts[:2]
                    mz_arr.append(float(this_mz))
                    int_arr.append(float(this_int))

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
    url = 'https://api.metabolomics-usi.gnps2.org/json/?usi1=' + usi
    response = get(url, timeout=10)
    json_data = loads(response.text)

    # check if the USI is valid
    if 'error' in json_data:
        raise ValueError

    # get adduct
    if adduct == '':
        adduct = None

    # valid: dict_keys(['n_peaks', 'peaks', 'precursor_charge', 'precursor_mz', 'splash'])
    # ion mode
    charge = json_data.get('precursor_charge')
    if charge is None or charge == 0:
        if adduct is not None:
            pos_mode = str(adduct)[-1] != '-'  # use adduct if charge is 0
            charge = 1 if pos_mode else -1
        else:
            charge = 1  # default to +1
            logging.warning('Charge is 0 and adduct is not provided. Default charge +1 is used.')

    peaks = json_data.get('peaks')
    ms2 = None
    if peaks is not None and len(peaks) > 0:
        peaks = np.asarray(peaks, dtype=float)
        if peaks.ndim != 2 or peaks.shape[1] < 2:
            raise ValueError('USI peaks must contain m/z and intensity pairs.')
        ms2 = Spectrum(peaks[:, 0], peaks[:, 1])

    precursor_mz = float(json_data['precursor_mz'])
    if not np.isfinite(precursor_mz) or precursor_mz <= 0:
        raise ValueError('USI precursor m/z must be a positive finite number.')

    data = MetaFeature(mz=precursor_mz,
                       charge=charge,
                       adduct=adduct,
                       ms2=ms2,
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

    # Normalize inputs before aligning adducts and removing duplicate USIs.
    if isinstance(usi_list, str):
        usi_list = [usi_list]
    else:
        usi_list = list(usi_list)

    if adduct_list is None:
        adduct_list = [None] * len(usi_list)
    elif isinstance(adduct_list, str):
        _validate_adduct_elements(adduct_list)
        adduct_list = [adduct_list.strip()] * len(usi_list)
    else:
        adduct_list = [item.strip() if isinstance(item, str) else item for item in adduct_list]
        for adduct in adduct_list:
            _validate_adduct_elements(adduct)
        if len(adduct_list) != len(usi_list):
            logging.warning('adduct_list and usi_list must have the same length. Default adducts are used.')
            adduct_list = [None] * len(usi_list)

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
        except UnsupportedElementError:
            raise
        except Exception as exc:
            logging.warning('Invalid USI: %s (%s)', usi, exc)
            continue
    return data_list


if __name__ == '__main__':
    
    mf = _load_usi('mzspec:TINYMASS:01KG1DBJTAEHQGZRPNZPHF8EYN:scan:1')
    print(mf)
    
    # init_db()
    #############
    # compile all these databases
    # import joblib
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

    # # model
    # model_ms1_ms2 = j_load('data/model_ms1_ms2.joblib')
    # model_noms1_ms2 = j_load('data/model_ms2.joblib')
    # model_ms1_noms2 = j_load('data/model_ms1.joblib')
    # model_noms1_noms2 = j_load('data/model.joblib')
    #
    # ml = [model_ms1_ms2, model_noms1_ms2, model_ms1_noms2, model_noms1_noms2, 1.898745, -2.012396, 1.930475,
    # -2.617705, 1.428206, -1.237494, 1.610414, -2.045858]
    # joblib.dump(ml, "data/ml_v0.3.0.joblib")
