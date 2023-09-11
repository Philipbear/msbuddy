from typing import List

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


def form_arr_to_str(form_arr: List[int]) -> str:
    """
    convert 12-dim formula array to Hill string
    :param form_arr: 12-dim array
    :return: formula string
    """

    def decode(element: str, cnt: int) -> str:
        if not cnt:
            return ''
        if cnt == 1:
            return element
        return element + str(cnt)

    formula_str = ''.join([decode(s, c) for s, c in zip(alphabet, form_arr)])
    return formula_str


