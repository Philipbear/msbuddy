from typing import List
from numba import njit
import numpy as np
from chemparse import parse_formula

# define alphabet
alphabet = ["C", "H", "Br", "Cl", "F", "I", "K", "N", "Na", "O", "P", "S"]


# functions related to formula
def read_formula(form: str) -> np.array:
    """
    read neutral formula string and return a 12-dim array, return None if invalid
    :param form: formula string in neutral form, e.g., C6H5NO2
    :return: 12-dim array
    """

    # parse formula string
    try:
        parsed = parse_formula(form)
    except:
        return None

    # check whether contains elements not in alphabet
    for element in parsed.keys():
        if element not in alphabet:
            return None

    # convert to 12-dim array
    array = np.zeros(12, dtype=int)
    for i, element in enumerate(alphabet):
        if element in parsed.keys():
            array[i] = parsed[element]
    return array


def enumerate_subform_arr(form_arr: List[int]) -> np.ndarray:
    """
    enumerate all possible sub-formula arrays of a given formula array
    :param form_arr: a list-like object
    :return: a 2-dim numpy array, with each row being a sub-formula array
    """

    form_arr = np.array(form_arr)

    # enumerate all possible sub-formula arrays
    return enumerate_subformula(form_arr)


@njit
def enumerate_subformula(pre_charged_arr: np.array) -> np.array:
    """
    Enumerate all subformulas of a candidate formula. (Numba version)
    :param pre_charged_arr: precursor charged array
    :return: 2D array, each row is a subformula array
    """
    n = len(pre_charged_arr)
    total_subform_cnt = np.prod(pre_charged_arr + 1)

    subform_arr = np.zeros((total_subform_cnt, n), dtype=np.int64)
    tempSize = 1

    for i in range(n):
        count = pre_charged_arr[i]
        repeatSize = tempSize
        tempSize *= (count + 1)

        pattern = np.arange(count + 1)

        repeated_pattern = np.empty(repeatSize * len(pattern), dtype=np.int64)
        for j in range(len(pattern)):
            repeated_pattern[j * repeatSize: (j + 1) * repeatSize] = pattern[j]

        full_repeats = total_subform_cnt // len(repeated_pattern)

        for j in range(full_repeats):
            start_idx = j * len(repeated_pattern)
            end_idx = (j + 1) * len(repeated_pattern)
            subform_arr[start_idx:end_idx, i] = repeated_pattern

    return subform_arr


# for numba
alphabet_np = np.array(
    [ord(char) for word in ["C", "H", "Br", "Cl", "F", "I", "K", "N", "Na", "O", "P", "S"] for char in word],
    dtype=np.int32)
word_lengths = np.array([len(word) for word in ["C", "H", "Br", "Cl", "F", "I", "K", "N", "Na", "O", "P", "S"]],
                        dtype=np.int32)


@njit
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


def ascii_to_str(ascii_arr) -> str:
    """
    Convert ASCII integer array to string
    :param ascii_arr: ASCII array
    :return: string
    """
    return ''.join(chr(i) for i in ascii_arr)


def form_arr_to_str(form_arr) -> str:
    """
    Convert formula array to string. (Numba version)
    :param form_arr: formula array
    :return: formula string
    """
    return ascii_to_str(_form_arr_to_str(form_arr))
