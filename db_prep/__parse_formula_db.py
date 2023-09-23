import pandas as pd
from rdkit.Chem import PandasTools
import glob
from pathlib import Path
from argparse import ArgumentParser


# load all sdf files in the folder
def load_pubchem_sdf(sdf_file_path: Path, output_path: Path):
    for sdf in glob.glob(str(sdf_file_path) + '/*.sdf'):
        csv_str = str(sdf).split('Compound_')[1].split('.')[0] + '_formulaCount.csv'
        # if csv file exists, skip
        if (output_path / csv_str).exists():
            continue

        file = PandasTools.LoadSDF(sdf, smilesName='SMILES', molColName='Molecule', includeFingerprints=False)
        result_dup = file[['PUBCHEM_MOLECULAR_FORMULA']]
        occur = result_dup.groupby(['PUBCHEM_MOLECULAR_FORMULA']).size()
        occur_df = pd.DataFrame({'PUBCHEM_MOLECULAR_FORMULA': occur.index, 'PubChem': occur.values})
        result_no_dup = result_dup.drop_duplicates(subset=['PUBCHEM_MOLECULAR_FORMULA'])
        result = pd.merge(result_no_dup, occur_df, on='PUBCHEM_MOLECULAR_FORMULA')
        result = result.rename(columns={'PUBCHEM_MOLECULAR_FORMULA': 'Formula_str'})

        result.to_csv(output_path / csv_str, index=False)


def merge_pubchem_csv(csv_file_path: Path, output_path: Path):
    list_csv = list()
    for csv in glob.glob(str(csv_file_path) + '/*formulaCount.csv'):
        file = pd.read_csv(csv)
        list_csv.append(file)
    merged = pd.concat(list_csv).groupby(['Formula_str']).sum().reset_index()
    merged.to_csv(output_path / 'merged_pubchem.csv', index=False)


def load_coconut_sdf(input_path: Path, output_path: Path):
    coconut = glob.glob(str(input_path) + '/COCONUT_DB.sdf')
    file = PandasTools.LoadSDF(coconut[0], smilesName='SMILES', molColName='Molecule', includeFingerprints=False)
    result_dup = file[['molecular_formula']]
    occur = result_dup.groupby(['molecular_formula']).size()
    occur_df = pd.DataFrame({'molecular_formula': occur.index, 'Coconut': occur.values})
    result_no_dup = result_dup.drop_duplicates(subset=['molecular_formula'])
    result = pd.merge(result_no_dup, occur_df, on='molecular_formula')
    result = result.rename(columns={'molecular_formula': 'Formula_str'})
    result.to_csv(output_path / 'coconut_formula.csv', index=False)


def load_npatlas_xlsx(input_path: Path, output_path: Path):
    npatlas = glob.glob(str(input_path) + '/NPAtlas_download.xlsx')
    file = pd.read_excel(npatlas[0])
    result_dup = file[['compound_molecular_formula']]
    occur = result_dup.groupby(['compound_molecular_formula']).size()
    occur_df = pd.DataFrame({'compound_molecular_formula': occur.index, 'npatlas': occur.values})
    result_no_dup = result_dup.drop_duplicates(subset=['compound_molecular_formula'])
    result = pd.merge(result_no_dup, occur_df, on='compound_molecular_formula')
    result = result.rename(columns={'compound_molecular_formula': 'Formula_str'})
    result.to_csv(output_path / 'npatlas_formula.csv', index=False)

def get_args():
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('--sdf', type=str, help='sdf file path')
    parser.add_argument('--csv', type=str, help='csv file path')
    parser.add_argument('--output', type=str, help='output file path')
    return parser.parse_args()


if __name__ == '__main__':
    # start_time = time.time()
    #
    # # load_pubchem_sdf(Path("/Users/tshyun/Desktop/2022.2.10/COOP/workterm1/pyBUDDY/db"),
    # #                  Path("/Users/tshyun/Desktop/2022.2.10/COOP/workterm1/pyBUDDY/db"))
    # merge_pubchem_csv(Path("/Users/tshyun/Desktop/2022.2.10/COOP/workterm1/pyBUDDY/db"),
    #                   Path("/Users/tshyun/Desktop/2022.2.10/COOP/workterm1/pyBUDDY/db"))
    # # 32.6382905983h for total, est 8s for this file
    # # est 95000M total file size, this one is 6.5M
    # print("--- %s seconds ---" % (time.time() - start_time))

    args = get_args()
    # load_coconut_sdf(Path("/Users/tshyun/Desktop/2022.2.10/COOP/workterm1/pyBUDDY/db"),
    #                  Path("/Users/tshyun/Desktop/2022.2.10/COOP/workterm1/pyBUDDY/db"))
    load_npatlas_xlsx(Path("/Users/tshyun/Desktop/2022.2.10/COOP/workterm1/pyBUDDY/db"),
                     Path("/Users/tshyun/Desktop/2022.2.10/COOP/workterm1/pyBUDDY/db"))

    # load_pubchem_sdf(Path(args.sdf), Path(args.output))
    # merge_pubchem_csv(Path(args.csv), Path(args.output))
