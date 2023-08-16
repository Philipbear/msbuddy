import sys
import numpy as np
import requests
import json
from typing import List, Union
from gdown import download as gdown_download
from pathlib import Path
from joblib import load as joblib_load
from msbuddy.utils import set_dependency
from pyteomics.mgf import read as mgf_read
from msbuddy.base_class import MetaFeature, Spectrum


def check_and_download(url: str, path) -> bool:
    """
    check if the file exists, if not, download from url
    :param url: url to download
    :param path: path to save
    :return: True if success
    """
    if not path.exists() or path.stat().st_size < 10 ** 4:
        gdown_download(url, str(path))
        return True


def init_db(db_mode: int) -> bool:
    """
    init databases used in the project
    :param db_mode: 0: basic; 1: halogen
    :return: True if success
    """
    root_path = Path(__file__).parent

    sys.stdout.write("Loading ML models...\n")
    # load database & models into memory
    set_dependency(common_loss_db=joblib_load(root_path / 'data' / 'common_loss.joblib'),
                   common_frag_db=joblib_load(root_path / 'data' / 'common_frag.joblib'),
                   model_a=joblib_load(root_path / 'data' / 'formula_model.joblib'),
                   model_a_mean_arr=joblib_load(root_path / 'data' / 'mean_arr.joblib'),
                   model_a_std_arr=joblib_load(root_path / 'data' / 'std_arr.joblib'))

    # check existence of basic_db_mass.joblib, basic_db_formula.joblib
    check_and_download('https://drive.google.com/uc?id=1obPMk9lcfkUpRkeGSkM1s4C9Bzatm1li',
                       root_path / 'data' / 'basic_db_mass.joblib')
    check_and_download('https://drive.google.com/uc?id=155AEYIv5XFBIc7Adpnfi-vH3s47QkbJf',
                       root_path / 'data' / 'basic_db_formula.joblib')

    sys.stdout.write("Loading basic_db...\n")
    set_dependency(basic_db_mass=joblib_load(root_path / 'data' / 'basic_db_mass.joblib'),
                   basic_db_formula=joblib_load(root_path / 'data' / 'basic_db_formula.joblib'),
                   basic_db_idx=joblib_load(root_path / 'data' / 'basic_db_idx.joblib'))

    if db_mode >= 1:
        # check existence of halogen_db_mass.joblib, halogen_db_formula.joblib
        check_and_download('https://drive.google.com/uc?id=1SMhezxtXtjQNj2N8odYWSEufOO_6N1o5',
                           root_path / 'data' / 'halogen_db_mass.joblib')
        check_and_download('https://drive.google.com/uc?id=18G8_qzTXWHDIw9Z9PwvMtjKWOi6FtwDU',
                           root_path / 'data' / 'halogen_db_formula.joblib')

        sys.stdout.write("Loading halogen_db...\n")
        set_dependency(halogen_db_mass=joblib_load(root_path / 'data' / 'halogen_db_mass.joblib'),
                       halogen_db_formula=joblib_load(root_path / 'data' / 'halogen_db_formula.joblib'),
                       halogen_db_idx=joblib_load(root_path / 'data' / 'halogen_db_idx.joblib'))
    return True


def load_mgf(file) -> List[MetaFeature]:
    """
    load mgf file. read mgf file using pyteomics.mgf.read
    :param file: mgf file path
    :return: list of MetaFeature
    """
    _mgf = mgf_read(file)

    # create meta_feature_list
    meta_feature_list = []
    for spec in _mgf:

        params = spec['params']

        # identifier
        identifier = params['title']

        # adduct
        # if 'ion' in params: adduct = 'ion'; else: None
        adduct = None
        if 'ion' in params.keys():
            adduct = params['ion']

        # rt
        rt = None
        if 'rtinseconds' in params.keys():
            rt = params['rtinseconds']
        elif 'rtinminutes' in params.keys():
            rt = params['rtinminutes'] * 60

        # spectrum
        mz_arr = np.array(spec['m/z array'])
        int_arr = np.array(spec['intensity array'])
        ms2_spectrum = Spectrum(mz_arr, int_arr)

        # create meta_feature
        meta_feature = MetaFeature(mz=params['pepmass'][0],
                                   charge=params['charge'][0],
                                   rt=rt,
                                   adduct=adduct,
                                   ms2=ms2_spectrum,
                                   identifier=identifier)

        # add meta_feature to meta_feature_list
        meta_feature_list.append(meta_feature)

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
    response = requests.get(url)
    json_data = json.loads(response.text)

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


def load_usi(usi_list: List[str],
             adduct_list: Union[List[str], None] = None) -> List[MetaFeature]:
    """
    Read from a sequence of USI strings and return a list of MetaFeature objects.
    Invalid USI strings are discarded.
    The GNPS API is used to get the spectrum from the USI.
    Citation: Universal MS/MS Visualization and Retrieval with the Metabolomics Spectrum Resolver Web Service.
    Wout Bittremieux et al. doi: 10.1101/2020.05.09.086066.
    See https://ccms-ucsd.github.io/GNPSDocumentation/api/#experimental-or-library-spectrum-by-usi for details.
     ---------------------------------------------------------
    :param usi_list: List of USI string
    :param adduct_list: adduct string, e.g. [M+H]+
    :return: List of MetaFeature objects
    """

    if adduct_list is None:
        adduct_list = [None] * len(usi_list)
    elif len(adduct_list) != len(usi_list):
        raise ValueError('The length of adduct_list must be the same as the length of usi_list.')

    data_list = []
    for usi, adduct in zip(usi_list, adduct_list):
        try:
            data_list.append(_load_usi(usi, adduct))
        except:
            print('Invalid USI: ' + usi)

    return data_list


# test
if __name__ == '__main__':
    init_db(1)

    from msbuddy.utils import dependencies

    # print(dependencies['common_loss_db'])
    print("basic db: " + str(len(dependencies['basic_db_mass'])))
    print("halogen db: " + str(len(dependencies['halogen_db_mass'])))

    # usi_1 = 'mzspec:GNPS:TASK-c95481f0c53d42e78a61bf899e9f9adb-spectra/specs_ms.mgf:scan:1943'
    # # usi_str = 'mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555'
    # usi_2 = 'mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00005436077'
    # usi_3 = 'mzspec:MSV000078547:120228_nbut_3610_it_it_take2:scan:389'
    #
    # _data = load_usi([usi_1, usi_2, usi_3])
    # print(_data[1])
