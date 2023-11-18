import joblib
import pandas as pd
import numpy as np
from numba import njit


# for numba
alphabet_np = np.array(
    [ord(char) for word in ["C", "H", "Br", "Cl", "F", "I", "K", "N", "Na", "O", "P", "S"] for char in word],
    dtype=np.int16)
word_lengths = np.array([len(word) for word in ["C", "H", "Br", "Cl", "F", "I", "K", "N", "Na", "O", "P", "S"]],
                        dtype=np.int16)


# @njit
def _form_arr_to_str(form_arr: np.array):
    """
    Inner func: convert formula array to string. (Numba version)
    :param form_arr: formula array
    :return: formula_list
    """
    formula_list = []
    idx = 0
    for i in range(len(word_lengths)):
        if form_arr[i]:
            for _ in range(word_lengths[i]):
                formula_list.append(alphabet_np[idx])
                idx += 1
            if form_arr[i] > 1:
                for digit in str(form_arr[i]):
                    formula_list.append(ord(digit))
        else:
            idx += word_lengths[i]
    return formula_list

def _ascii_to_str(ascii_arr) -> str:
    """
    Convert ASCII integer array to string
    :param ascii_arr: ASCII array
    :return: string
    """
    return ''.join(chr(i) for i in ascii_arr)

def form_arr_to_str(form_arr) -> str:
    """
    Convert formula array to string. (Numba version)
    :param form_arr: formula np.array
    :return: formula string
    """
    return _ascii_to_str(_form_arr_to_str(form_arr))

def load_frag_table():
    """
    load common fragment table from file, no carbon
    :return: np.array, shape=(n, 12), dtype=int
    """
    tb = pd.read_csv("E:/pyBUDDY/database/CommonFragTable.csv")

    # reorder the element columns into the same order as in the database
    out_arr = np.zeros((tb.shape[0], 12), dtype=int)
    for i in range(tb.shape[0]):
        out_arr[i, :] = np.array([0, tb.loc[i, 'H'], tb.loc[i, 'Br'], tb.loc[i, 'Cl'], tb.loc[i, 'F'], tb.loc[i, 'I'],
                                  0, tb.loc[i, 'N'], 0, tb.loc[i, 'O'], tb.loc[i, 'P'], tb.loc[i, 'S']])

    # reorder entries by columns of H, then by Br, Cl, F, I, N, O, P, S, in ascending order
    indices = np.lexsort((out_arr[:, 11], out_arr[:, 10], out_arr[:, 9], out_arr[:, 8], out_arr[:, 7], out_arr[:, 6],
                          out_arr[:, 5], out_arr[:, 4], out_arr[:, 3], out_arr[:, 2], out_arr[:, 1], out_arr[:, 0]))
    arr = out_arr[indices, :]

    # save to file
    joblib.dump(arr, "../data/common_frag.joblib")


def load_loss_table():
    """
    load common loss table from file, radical + non-radical
    :return: np.array, shape=(n, 12), dtype=int
    """
    tb = pd.read_csv("E:/pyBUDDY/database/CommonLossTable.csv")

    # reorder the element columns into the same order as in the database
    out_arr = np.zeros((tb.shape[0], 12), dtype=int)
    for i in range(tb.shape[0]):
        out_arr[i, :] = np.array(
            [tb.loc[i, 'C'], tb.loc[i, 'H'], tb.loc[i, 'Br'], tb.loc[i, 'Cl'], tb.loc[i, 'F'], tb.loc[i, 'I'],
             0, tb.loc[i, 'N'], 0, tb.loc[i, 'O'], tb.loc[i, 'P'], tb.loc[i, 'S']])

    # order by C, H, Br, Cl, F, I, N, O, P, S
    indices = np.lexsort((out_arr[:, 11], out_arr[:, 10], out_arr[:, 9], out_arr[:, 8], out_arr[:, 7],
                          out_arr[:, 6], out_arr[:, 5], out_arr[:, 4], out_arr[:, 3], out_arr[:, 2], out_arr[:, 1],
                          out_arr[:, 0]))
    arr = out_arr[indices, :]

    # save to file
    joblib.dump(arr, "../data/common_loss.joblib")


def formula_db_idx_arr():
    """
    create a database index array for formula db.
    this idx array is used for fast searching of formula db
    :return: np.array, dtype=int
    """
    db = pd.read_csv("formulaDB_20230316.csv")
    print("db size before dereplication: ", db.shape[0])


    # rename "PubChem" to "pubchem"
    db.rename(columns={'PubChem': 'pubchem'}, inplace=True)
    # create a new column "OtherDB" for non-PubChem databases, fill in the total count of non-PubChem databases
    db['other_db'] = db[['ANPDB', 'BLEXP', 'BMDB', 'ChEBI', 'COCONUT', 'DrugBank', 'DSSTOX', 'ECMDB',
                         'FooDB', 'HMDB', 'HSDB', 'KEGG', 'LMSD', 'MaConDa', 'MarkerDB', 'MCDB', 'NORMAN', 'NPASS', 'NPAtlas',
                         'Plantcyc', 'SMPDB', 'STOFF_IDENT', 'T3DB', 'TTD', 'UNPD', 'YMDB']].sum(axis=1)
    db['freq'] = db['pubchem'] + db['other_db']

    # drop rows with se > 0 or si > 0 or b > 0
    db = db.loc[(db['se'] == 0) & (db['si'] == 0) & (db['b'] == 0), :]
    db.index = range(db.shape[0])

    # refill formula_str column
    db['form_str'] = db.apply(lambda x: form_arr_to_str(np.array(x[['c', 'h', 'br', 'cl', 'f', 'i', 'k', 'n', 'na', 'o', 'p', 's']].values)), axis=1)


    # dereplicate
    db = db.drop_duplicates(subset=['form_str'], inplace=False)
    print("db size after dereplication: ", db.shape[0])

    # reindex
    db.index = range(db.shape[0])
    # sort by mass
    db.sort_values(by=['mass'], inplace=True)
    db.index = range(db.shape[0])

    # c, h, b, br, cl, f, i, k, n, na, o, p, s, se, si
    # basic db: db with sum of b, k, na, se, si, cl, f, i, br = 0
    basic_db = db.loc[
               (db['b'] + db['k'] + db['na'] + db['se'] + db['si'] + db['cl'] + db['f'] + db['i'] + db['br']) == 0, :]
    # halogen db: db with sum of b, k, na, se, si = 0, and sum of cl, f, i, br > 0
    bool_arr1 = (db['b'] + db['k'] + db['na'] + db['se'] + db['si']) == 0
    bool_arr2 = (db['cl'] + db['f'] + db['i'] + db['br']) > 0
    halogen_db = db.loc[bool_arr1 & bool_arr2, :]
    # # all db: db with sum of b, k, na, se, si > 0
    # all_db = db.loc[(db['b'] + db['k'] + db['na'] + db['se'] + db['si']) > 0, :]

    print("basic db size: ", basic_db.shape[0])
    print("halogen db size: ", halogen_db.shape[0])
    # print("all db size: ", all_db.shape[0])

    basic_db_model = []
    # combine c, h, b, br, cl, f, i, k, n, na, o, p, s, se, si into formula_arr, np.array
    basic_db['formula_arr'] = basic_db[['c', 'h', 'br', 'cl', 'f', 'i', 'k', 'n', 'na', 'o', 'p', 's']].values.tolist()
    basic_db_mass = basic_db['mass'].values
    basic_db_formula = np.stack(basic_db['formula_arr'].values)
    basic_db_freq_array = basic_db['freq'].values
    # log 10 of freq
    basic_db_logfreq = np.log10(basic_db_freq_array)
    # joblib.dump(basic_db_mass_array, "basic_db_mass.joblib")
    # joblib.dump(basic_db_formula_array, "basic_db_formula.joblib")


    print("halogen db")
    halogen_db_model = []
    # combine c, h, b, br, cl, f, i, k, n, na, o, p, s, se, si into formula_arr, np.array
    halogen_db['formula_arr'] = halogen_db[['c', 'h', 'br', 'cl', 'f', 'i', 'k', 'n', 'na', 'o', 'p', 's']].values.tolist()
    halogen_db_mass = halogen_db['mass'].values
    halogen_db_formula = np.stack(halogen_db['formula_arr'].values)
    halogen_db_freq_array = halogen_db['freq'].values
    # log 10 of freq
    halogen_db_logfreq = np.log10(halogen_db_freq_array)
    # joblib.dump(halogen_db_mass_array, "halogen_db_mass.joblib")
    # joblib.dump(halogen_db_formula_array, "halogen_db_formula.joblib")


    print("idx array")
    # basic db
    # mass array
    mass_arr = np.array(basic_db['mass'])
    # create idx array, only for mass < 1500
    basic_db_idx = np.zeros(15000, dtype=int)
    # for each index i, find the index of the first formula with mass >= i/10
    for i in range(len(basic_db_idx)):
        # find the first index of mass >= i/10
        basic_db_idx[i] = np.where(mass_arr >= i / 10)[0][0]
    # joblib.dump(basic_db_idx, "basic_db_idx.joblib")

    # halogen db
    # mass array
    mass_arr = np.array(halogen_db['mass'])
    # create idx array, only for mass < 1500
    halogen_db_idx = np.zeros(15000, dtype=int)
    # for each index i, find the index of the first formula with mass >= i/10
    for i in range(len(halogen_db_idx)):
        # find the first index of mass >= i/10
        halogen_db_idx[i] = np.where(mass_arr >= i / 10)[0][0]
    # joblib.dump(halogen_db_idx, "halogen_db_idx.joblib")

    basic_db = [basic_db_mass, basic_db_formula, basic_db_logfreq, basic_db_idx]
    halogen_db = [halogen_db_mass, halogen_db_formula, halogen_db_logfreq, halogen_db_idx]
    formula_db = [basic_db, halogen_db]
    joblib.dump(formula_db, "formula_db_v0.3.0.joblib")


# test
if __name__ == '__main__':
    # load_frag_table()
    # load_loss_table()
    # formula_db_idx_arr()

    # test
    import numpy as np
    basic_db, halogen_db = joblib.load("formula_db_v0.3.0.joblib")

    data_np = np.array(basic_db[2])

    # Calculating statistics using numpy
    mean_np = np.mean(data_np)
    median_np = np.median(data_np)
    quantiles_np = np.quantile(data_np, [0.25, 0.5, 0.75])  # 25th, 50th (median), 75th percentiles

    print('done')
