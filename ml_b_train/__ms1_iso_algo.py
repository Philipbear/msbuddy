import pandas as pd
import numpy as np
from msbuddy.base_class import Formula
from brainpy import isotopic_variants
from tqdm import tqdm


def sim_ms1_iso_pattern(form_arr):
    """
    simulate MS1 isotope pattern
    :param form_arr: numpy array of formula
    :return: theoretical & simulated isotope pattern
    """

    # calculate theoretical isotope pattern
    # mapping to a dictionary
    arr_dict = {}
    for i, element in enumerate(Formula.alphabet):
        arr_dict[element] = form_arr[i]

    # calculate isotope pattern
    isotope_pattern = isotopic_variants(arr_dict, npeaks=4)
    int_arr = np.array([iso.intensity for iso in isotope_pattern])

    # simulation
    sim_int_arr = int_arr.copy()
    a1, a2, a3 = 2, 2, 2
    b1, b2, b3 = -1, -1, -1

    # M + 1
    while a1 * b1 < -1:
        a1 = abs(np.random.normal(0, 0.11))
        b1 = np.random.choice([-1, 1])
    sim_int_arr[1] = sim_int_arr[1] * (1 + a1 * b1)

    # M + 2
    if len(int_arr) >= 3:
        while a2 * b2 < -1:
            a2 = abs(np.random.normal(0, 0.16))
            # random.choice([-1, 1]), 0.7 probability to be 1
            b2 = np.random.choice([-b1, b1], p=[0.3, 0.7])
        sim_int_arr[2] = sim_int_arr[2] * (1 + a2 * b2)

    # M + 3
    if len(int_arr) >= 4:
        while a3 * b3 < -1:
            a3 = abs(np.random.normal(0, 0.19))
            b3 = np.random.choice([-b2, b2], p=[0.3, 0.7])
        sim_int_arr[3] = sim_int_arr[3] * (1 + a3 * b3)

    return int_arr, sim_int_arr


def dp_iso_sim(int_arr_x, int_arr_y):
    """
    dot product similarity of two isotope patterns
    :param int_arr_x: the first isotope pattern
    :param int_arr_y: the second isotope pattern
    :return: the similarity
    """
    min_len = min(len(int_arr_x), len(int_arr_y))
    int_arr_x = int_arr_x[:min_len]
    int_arr_y = int_arr_y[:min_len]
    # calculate the similarity
    sim = np.dot(int_arr_x, int_arr_y) / (np.linalg.norm(int_arr_x) * np.linalg.norm(int_arr_y))
    return sim


def cor_iso_sim(int_arr_x, int_arr_y):
    """
    correlation similarity of two isotope patterns
    :param int_arr_x: the first isotope pattern
    :param int_arr_y: the second isotope pattern
    :return: the similarity
    """
    min_len = min(len(int_arr_x), len(int_arr_y))
    int_arr_x = int_arr_x[:min_len]
    int_arr_y = int_arr_y[:min_len]
    # calculate the similarity
    sim = np.corrcoef(int_arr_x, int_arr_y)[0, 1]
    return sim


def ori_iso_sim(int_arr_x, int_arr_y, iso_num=4):
    """
    original similarity of two isotope patterns from BUDDY
    :param int_arr_x: the theoretical isotope pattern
    :param int_arr_y: the experimental isotope pattern
    :param iso_num: the number of isotopes
    :return: the similarity
    """
    min_len = min(len(int_arr_x), iso_num)
    int_arr_x = int_arr_x[:min_len]  # theoretical isotope pattern
    if len(int_arr_y) > min_len:  # experimental isotope pattern
        int_arr_y = int_arr_y[:min_len]
    if len(int_arr_y) < min_len:
        int_arr_y = np.append(int_arr_y, np.zeros(min_len - len(int_arr_y)))

    # normalize
    int_arr_x /= int_arr_x[0]
    int_arr_y /= int_arr_y[0]

    # calculate the similarity
    int_diff_arr = abs(int_arr_y - int_arr_x)
    # clip the difference
    int_diff_arr = np.clip(int_diff_arr, 0, 1)
    sim = np.prod(1 - int_diff_arr)

    return sim


def iso_sim_2(int_arr_x, int_arr_y, iso_num=4):
    """
    original similarity of two isotope patterns from BUDDY
    :param int_arr_x: the theoretical isotope pattern
    :param int_arr_y: the experimental isotope pattern
    :param iso_num: the number of isotopes
    :return: the similarity
    """
    min_len = min(len(int_arr_x), iso_num)
    int_arr_x = int_arr_x[:min_len]  # theoretical isotope pattern
    if len(int_arr_y) > min_len:  # experimental isotope pattern
        int_arr_y = int_arr_y[:min_len]
    if len(int_arr_y) < min_len:
        int_arr_y = np.append(int_arr_y, np.zeros(min_len - len(int_arr_y)))

    # normalize
    int_arr_x /= sum(int_arr_x)
    int_arr_y /= sum(int_arr_y)

    # calculate the similarity
    int_diff_arr = abs(int_arr_y - int_arr_x)
    sim = 1 - np.sum(int_diff_arr)

    return sim


def select_ms1_iso_algo():
    """
    select the best algorithm for MS1 isotope pattern
    :return: the best algorithm
    """
    db = pd.read_csv('../ml_a/formula_data.csv')

    # add two columns to store iso sim results
    # db['iso_sim_dp'] = None
    # db['iso_sim_cor'] = None
    db['iso_sim_ori'] = None
    db['iso_sim_2'] = None
    # db['iso_sim_dp_wrong'] = None
    # db['iso_sim_cor_wrong'] = None
    db['iso_sim_ori_wrong'] = None
    db['iso_sim_2_wrong'] = None

    ele_arr = np.array([db['c'][0], db['h'][0], db['br'][0], db['cl'][0], db['f'][0], db['i'][0],
                        0, db['n'][0], 0, db['o'][0], db['p'][0], db['s'][0]])
    theo_int_arr_pre, sim_int_arr_pre = sim_ms1_iso_pattern(ele_arr)

    print('start calculating iso sim...')
    for i in tqdm(range(len(db))):
        # ["C", "H", "Br", "Cl", "F", "I", "K", "N", "Na", "O", "P", "S"]
        ele_arr = np.array([db['c'][i], db['h'][i], db['br'][i], db['cl'][i], db['f'][i], db['i'][i],
                            0, db['n'][i], 0, db['o'][i], db['p'][i], db['s'][i]])
        theo_int_arr, sim_int_arr = sim_ms1_iso_pattern(ele_arr)
        # db['iso_sim_dp'][i] = dp_iso_sim(theo_int_arr, sim_int_arr)
        # db['iso_sim_cor'][i] = cor_iso_sim(theo_int_arr, sim_int_arr)
        db['iso_sim_ori'][i] = ori_iso_sim(theo_int_arr, sim_int_arr)
        db['iso_sim_2'][i] = iso_sim_2(theo_int_arr, sim_int_arr)
        # db['iso_sim_dp_wrong'][i] = dp_iso_sim(theo_int_arr_pre, sim_int_arr)
        # db['iso_sim_cor_wrong'][i] = cor_iso_sim(theo_int_arr_pre, sim_int_arr)
        db['iso_sim_ori_wrong'][i] = ori_iso_sim(theo_int_arr_pre, sim_int_arr)
        db['iso_sim_2_wrong'][i] = iso_sim_2(theo_int_arr_pre, sim_int_arr)
        theo_int_arr_pre = theo_int_arr.copy()

    # save the results
    db.to_csv('../ml_b/formulaDB_20230316_iso_sim.csv', index=False)


# test
if __name__ == '__main__':
    # import time
    # start = time.time()
    # for i in range(100):
    #     theo, sim = sim_ms1_iso_pattern(np.array([50, 63, 0, 4, 1, 0, 0, 6, 0, 3, 0, 0]))
    # end = time.time()
    # print(end - start)
    # print(theo)
    # print(sim)

    select_ms1_iso_algo()

    # after running the above code, iso_sim_2 is selected for MS1 isotope pattern similarity calculation
