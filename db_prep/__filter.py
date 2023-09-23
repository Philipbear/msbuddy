import glob

import pandas as pd
import chemparse
import numpy as np
import time

from tqdm import tqdm
import argparse
import pathlib

chemical_elements = ['C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'Si', 'B', 'Se', 'Na', 'K']
mass_array = np.array([12.000000, 1.007825, 14.003074, 15.994915, 30.973762, 31.972071, 18.998403, 34.968853, 78.918336,
                       126.904473, 27.976927, 10.012937, 73.922476, 22.989769, 38.963707])


def dict_to_chem_formula(input: dict) -> str:
    new_formula = ''
    for key in input:
        if key != '+' and key != '-':
            if int(input[key]) == 0:
                break
            elif int(input[key]) == 1:
                new_formula = new_formula + key
            else:
                new_formula = new_formula + key + str(int(input[key]))
    return new_formula


def charged_formula_coconut(input: str):
    df = pd.read_csv(input)
    index = 0
    no_chemical_drop_index_list = []
    charged_drop_index_list = []
    new_formula_df = pd.DataFrame(columns=['Formula_str', 'Coconut'])
    for formula, times in df.itertuples(index=False):
        formula_dict = chemparse.parse_formula(formula)
        keys = list(formula_dict.keys())
        if not keys:
            new_formula = ''
            start_index = formula.find('[')
            end_index = formula.find(']')
            clean_formula = formula[start_index + 1: end_index]
            clean_formula_dict = chemparse.parse_formula(clean_formula)
            if formula[len(formula) - 1] == '+':
                if formula[len(formula) - 2] == ']':
                    loss_h = 1
                else:
                    loss_h = int(formula[len(formula) - 2])
                if int(clean_formula_dict['H']) >= loss_h:
                    clean_formula_dict['H'] = int(clean_formula_dict['H']) - loss_h
                    for key in clean_formula_dict:
                        if int(clean_formula_dict[key]) == 0:
                            break
                        new_formula = new_formula + key + str(int(clean_formula_dict[key]))
                    new_formula_df.loc[len(new_formula_df)] = [new_formula, times]
            if formula[len(formula) - 1] == '-':
                if formula[len(formula) - 2] == ']':
                    add_h = 1
                else:
                    add_h = int(formula[len(formula) - 2])
                if 'H' in clean_formula_dict.keys():
                    clean_formula_dict['H'] = int(clean_formula_dict['H']) + add_h
                    new_formula = dict_to_chem_formula(clean_formula_dict)
                else:
                    new_formula = new_formula + 'C' + str(int(clean_formula_dict['C']))
                    if add_h == 1:
                        new_formula = new_formula + 'H'
                    else:
                        new_formula = new_formula + 'H' + str(int(add_h))
                    for key in clean_formula_dict:
                        if key != 'C':
                            if int(clean_formula_dict[key]) == 1:
                                new_formula = new_formula + key
                            else:
                                new_formula = new_formula + key + str(int(clean_formula_dict[key]))
                new_formula_df.loc[len(new_formula_df)] = [new_formula, times]
            charged_drop_index_list.append(index)
        for key in keys:
            if chemical_elements.count(key) == 0:
                no_chemical_drop_index_list.append(index)
                break
        index = index + 1
    total_drop = charged_drop_index_list + no_chemical_drop_index_list
    df_modified = df.drop(total_drop, axis=0)
    df_modified = pd.concat([df_modified, new_formula_df])
    df_modified.to_csv('coconut_formula_modified.csv', index=False)


def charged_formula_pub_and_npa(input: str):
    df = pd.read_csv(input)
    index = 0
    no_chemical_drop_index_list = []
    charged_drop_index_list = []
    if 'npatlas' in input:
        new_formula_df = pd.DataFrame(columns=['Formula_str', 'npatlas'])
    else:
        new_formula_df = pd.DataFrame(columns=['Formula_str', 'PubChem'])
    for formula in df['Formula_str']:
        formula_dict = chemparse.parse_formula(formula)
        keys = list(formula_dict.keys())
        for key in keys:
            if chemical_elements.count(key) == 0:
                no_chemical_drop_index_list.append(index)
                break
        index = index + 1
    df = df.drop(no_chemical_drop_index_list, axis=0)
    for formula, times in df.itertuples(index=False):
        formula_dict = chemparse.parse_formula(formula)
        keys = list(formula_dict.keys())
        new_formula = ''
        if '+' in keys:
            loss_h = formula_dict['+']
            if 'H' in formula_dict.keys():
                if int(formula_dict['H']) >= loss_h:
                    formula_dict['H'] = int(formula_dict['H']) - loss_h
                    new_formula = dict_to_chem_formula(formula_dict)
                    new_formula_df.loc[len(new_formula_df)] = [new_formula, times]
            charged_drop_index_list.append(index)
        if '-' in keys:
            add_h = formula_dict['-']
            if 'H' in formula_dict.keys():
                formula_dict['H'] = int(formula_dict['H']) + add_h
                new_formula = dict_to_chem_formula(formula_dict)
            else:
                print(formula_dict)
                new_formula = new_formula + 'C' + str(int(formula_dict['C']))
                if add_h == 1:
                    new_formula = new_formula + 'H'
                else:
                    new_formula = new_formula + 'H' + str(int(add_h))
                for key in formula_dict:
                    if key != 'C':
                        if int(formula_dict[key]) == 1:
                            new_formula = new_formula + key
                        else:
                            new_formula = new_formula + key + str(int(formula_dict[key]))
            new_formula_df.loc[len(new_formula_df)] = [new_formula, times]
            charged_drop_index_list.append(index)
    df_modified = df.drop(charged_drop_index_list, axis=0)
    df_modified = pd.concat([df_modified, new_formula_df])
    if 'npatlas' in input:
        df_modified.to_csv('npatlas_formula_modified.csv', index=False)
    else:
        df_modified.to_csv('pubchem_formula_modified.csv', index=False)


def merge():
    df_pubchem = pd.read_csv('pubchem_formula_modified.csv')
    df_coconut = pd.read_csv('coconut_formula_modified.csv')
    df_npatlas = pd.read_csv('npatlas_formula_modified.csv')
    df_merged = df_pubchem.merge(df_npatlas, how='outer')
    df_merged = df_merged.merge(df_coconut, how='outer')
    df_merged = df_merged.fillna(0)
    df_merged.to_csv('merged_pub_coco_npa.csv', index=False)


def test_file():
    df = pd.read_csv('merged_pub_coco_npa.csv')
    df.head(10000).to_csv('test_merged.csv', index=False)


def split_csv():
    # for i, chunk in enumerate(pd.read_csv('merged_pub_coco_npa.csv', chunksize=40000)):
    #     chunk.to_csv('chunk{}.csv'.format(i), index=False)
    for i, chunk in enumerate(pd.read_csv('update.csv', chunksize=40000)):
        chunk.to_csv('updatesplit{}.csv'.format(i), index=False)


def parse_mass_dbe():
    # df = pd.read_csv(pathlib.Path(folder_path) / 'merged_pub_coco_npa.csv')
    number = 0
    for file in glob.glob(str('/Users/tshyun/Desktop/2022.2.10/COOP/workterm1/pyBUDDY/db/split_ori') + '/*.csv'):
        # df = pd.read_csv('test_merged.csv')
        df = pd.read_csv(file)
        mass_element_df = pd.DataFrame(
            columns=['Formula_str', 'mass', 'C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'Si', 'B', 'Se', 'Na',
                     'K', 'PubChem', 'npatlas', 'Coconut'])
        # df.columns = ['Formula_str','mass', 'C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'Si', 'B', 'Se', 'Na', 'K',
        #               'PubChem','Npatlas','Coconut']
        count = 0
        for formula, PubChem, npatlas, Coconut in tqdm(df.itertuples(index=False)):
            formula_dict = chemparse.parse_formula(formula)
            # from Formula.read_formula function
            array = np.zeros(15, dtype=int)
            for i, element in enumerate(chemical_elements):
                if element in formula_dict.keys():
                    array[i] = formula_dict[element]
            dbe = array[0] + array[10] + 1 - (
                    array[1] + array[6] + array[7] + array[8] + array[9] + array[13] + array[14]) / 2 + (
                          array[2] + array[4] + array[11]) / 2
            mass = np.sum(array * mass_array)
            if not (dbe >= -5 and mass <= 2000):
                pass
            else:
                mass_element_df.loc[len(mass_element_df)] = [formula, mass, array[0], array[1], array[2], array[3],
                                                             array[4], array[5], array[6], array[7], array[8], array[9],
                                                             array[10], array[11], array[12], array[13], array[14],
                                                             PubChem, npatlas, Coconut]
            count = count + 1
        # output_path = pathlib.Path(folder_path) / 'merged_with_formula.csv'
        mass_element_df.to_csv(
            '/Users/tshyun/Desktop/2022.2.10/COOP/workterm1/pyBUDDY/db/split_ori/chunkoutput{}.csv'.format(number),
            index=False)
        number = number + 1
    # df_modified = df.drop(drop_index_list, axis=0)
    # print(len(df_modified))
    # print(df_modified.head(10))
    # print(df.head(10))


def concat_csv():
    output = pd.read_csv('chunkoutput0.csv')
    count = 0
    for file in glob.glob(str('/Users/tshyun/Desktop/2022.2.10/COOP/workterm1/pyBUDDY/db/split_output') + '/*.csv'):
        df = pd.read_csv(file)
        output = pd.concat([output, df], axis=0)
        print(count)
        count = count + 1
    output.to_csv('output.csv', index=False)


def arg_parser():
    parser = argparse.ArgumentParser(description='Merge database')
    parser.add_argument('--path', type=str, default='data')
    return parser.parse_args()


def update_old():
    update = pd.read_csv("output.csv")
    print(update.shape)
    update = update.rename(
        columns={'Formula_str': 'formula_str', 'C': 'c', 'H': 'h', 'N': 'n', 'O': 'o', 'P': 'p', 'S': 's',
                 'F': 'f', 'Cl': 'cl', 'Br': 'br', 'I': 'i', 'Si': 'si', 'B': 'b', 'Se': 'se', 'Na': 'na',
                 'K': 'k'})
    old = pd.read_csv("formulaDB_20210622.csv")
    print(old.shape)
    new = old.merge(update, how='outer',
                    left_on=["formula_str", "c", "h", "b", "br", "cl", "f", "i", "k", "n", "na", "o", "p", "s", "se",
                             "si"],
                    right_on=["formula_str", "c", "h", "b", "br", "cl", "f", "i", "k", "n", "na", "o", "p", "s", "se",
                              "si"])
    print(new.shape)
    new.to_csv('update.csv', index=False)


def modify():
    df = pd.read_csv('update.csv')
    df.mass_x = df.c * 12.000000 + df.h * 1.007825 + df.n * 14.003074 + df.o * 15.994915 + df.p * 30.973762 + df.s * 31.972071 + df.f * 18.998403 + df.cl * 34.968853 + df.br * 78.918336 + df.i * 126.904473 + df.si * 27.976927 + df.b * 10.012937 + df.se * 73.922476 + df.na * 22.989769 + df.k * 38.963707
    df = df.round({'mass_x': 6})
    df.PubChem_x = df[['PubChem_x', 'PubChem_y']].max(axis=1)
    df.COCONUT = df[['COCONUT', 'Coconut']].max(axis=1)
    df = df[['formula_str', 'mass_x', 'c', 'h', 'b', 'br', 'cl', 'f', 'i', 'k', 'n', 'na', 'o', 'p', 's', 'se', 'si', 'PubChem_x', 'ANPDB', 'BLEXP', 'BMDB', 'ChEBI', 'COCONUT', 'DrugBank', 'DSSTOX', 'ECMDB', 'FooDB', 'HMDB', 'HSDB', 'KEGG', 'LMSD', 'MaConDa', 'MarkerDB', 'MCDB', 'NORMAN', 'NPASS', 'npatlas', 'Plantcyc', 'SMPDB', 'STOFF_IDENT', 'T3DB', 'TTD', 'UNPD', 'YMDB', 'mass_y', 'PubChem_y', 'Coconut']]
    df = df.rename(columns={'mass_x': 'mass', 'PubChem_x': 'PubChem', 'npatlas': 'NPAtlas'})
    df = df.drop(columns=['mass_y', 'PubChem_y', 'Coconut'])
    df = df.replace(np.nan, 0.0)
    df.to_csv('formulaDB_20230316.csv', index=False)


if __name__ == '__main__':
    # start_time = time.time()
    # # charged_formula_pub_and_npa('npatlas_formula.csv')
    # # charged_formula_pub_and_npa('merged_pubchem.csv')
    # # charged_formula_coconut('coconut_formula.csv')
    # # merge()
    # # test_file()
    # parse_mass_dbe()
    # print("--- %s seconds ---" % (time.time() - start_time))

    # args = arg_parser()
    # # start_time = time.time()
    # parse_mass_dbe(args.path)
    modify()
    # print("--- %s seconds ---" % (t
