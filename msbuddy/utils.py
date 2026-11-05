# ==============================================================================
# Copyright (C) 2024-2026 Shipei Xing <sxing@fudan.edu.cn>
#
# Licensed under the Apache License 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://github.com/Philipbear/msbuddy/blob/main/LICENSE
# ==============================================================================
"""
File: utils.py
Author: Shipei Xing
Email: sxing@fudan.edu.cn
GitHub: Philipbear
Description: main util functions for msbuddy
"""

from collections.abc import Mapping
from itertools import product
from numbers import Integral
import re
from typing import Iterator, List

import numpy as np
from numba import njit

# define alphabet
alphabet = ["C", "H", "Br", "Cl", "F", "I", "K", "N", "Na", "O", "P", "S"]
SUPPORTED_ELEMENTS = tuple(alphabet)
_ELEMENT_SET = frozenset(SUPPORTED_ELEMENTS)
_MAX_ATOM_COUNT = np.iinfo(np.int16).max
_FORMULA_TOKEN = re.compile(r'[A-Z][a-z]?|[0-9]+|[().·]')


class UnsupportedElementError(ValueError):
    """An element is not supported by msbuddy's formula database."""


def _validate_elements(elements):
    for element in elements:
        if element not in _ELEMENT_SET:
            raise UnsupportedElementError(
                f'Unsupported element {element!r}. Supported elements: {", ".join(SUPPORTED_ELEMENTS)}.')


def _validate_composition(composition):
    """Return a normalized element-count mapping after validating its contents."""
    if not isinstance(composition, Mapping):
        raise TypeError('Formula composition must be a mapping of element symbols to atom counts.')
    _validate_elements(composition)
    for element, count in composition.items():
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral):
            raise ValueError(f'Atom count for {element} must be a nonnegative integer.')
        if count < 0 or count > _MAX_ATOM_COUNT:
            raise ValueError(f'Atom count for {element} must be between 0 and {_MAX_ATOM_COUNT}.')
    return {element: int(composition[element]) for element in SUPPORTED_ELEMENTS
            if element in composition and composition[element] != 0}


def _composition_to_array(composition):
    """Convert a validated public composition to the engine's fixed-order array."""
    composition = _validate_composition(composition)
    return np.array([composition.get(element, 0) for element in SUPPORTED_ELEMENTS], dtype=np.int16)


def _array_to_composition(array):
    return {element: int(count) for element, count in zip(SUPPORTED_ELEMENTS, array) if count != 0}


class FormulaResult:
    """
    FormulaResult class, for mass_to_formula function
    """

    def __init__(self, formula: str, mass: float, t_mass: float):
        """
        Constructor
        :param formula: formula string
        :param mass: exact mass of the formula
        :param t_mass: target mass
        """
        self.formula = formula
        self.mass = mass
        self.target_mass = t_mass
        self.mass_error = mass - t_mass
        self.mass_error_ppm = self.mass_error / t_mass * 1e6


class SubformulaResult:
    """
    SubformulaResult class, for assign_subformula function
    """

    def __init__(self, idx: int, subform_list: List[FormulaResult]):
        """
        Constructor
        :param idx: index of the fragment ion
        :param subform_list: list of FormulaResult objects
        """
        self.idx = idx
        self.subform_list = subform_list

    @property
    def peak_index(self):
        """Index of the input fragment peak."""
        return self.idx

    @property
    def formulas(self):
        """Formula matches for this fragment peak."""
        return self.subform_list


def read_formula(formula: str) -> dict:
    """Parse a formula into element counts, raising ValueError for invalid input.

    Parenthesized groups, leading multipliers, and dot-separated components are
    supported; a dot separates components rather than denoting a decimal count.
    """
    if not isinstance(formula, str):
        raise TypeError('Formula must be a string.')
    formula = formula.strip()
    if not formula:
        raise ValueError('Formula cannot be empty.')
    tokens = []
    position = 0
    for match in _FORMULA_TOKEN.finditer(formula):
        if match.start() != position:
            raise ValueError(f'Invalid formula syntax at position {position}: {formula!r}.')
        tokens.append(match.group())
        position = match.end()
    if position != len(formula):
        raise ValueError(f'Invalid formula syntax at position {position}: {formula!r}.')

    index = 0

    def number():
        nonlocal index
        if index < len(tokens) and tokens[index].isdigit():
            value = int(tokens[index])
            index += 1
            if value <= 0:
                raise ValueError('Formula multipliers and atom counts must be positive.')
            return value
        return 1

    def sequence():
        nonlocal index
        counts = {}
        consumed = False
        while index < len(tokens) and tokens[index] not in (')', '.', '·'):
            token = tokens[index]
            index += 1
            if token == '(':
                group = sequence()
                if index >= len(tokens) or tokens[index] != ')':
                    raise ValueError(f'Unmatched parenthesis in formula: {formula!r}.')
                index += 1
            elif token[0].isupper():
                _validate_elements((token,))
                group = {token: 1}
            else:
                raise ValueError(f'Invalid formula syntax: {formula!r}.')
            multiplier = number()
            for element, count in group.items():
                counts[element] = counts.get(element, 0) + count * multiplier
            consumed = True
        if not consumed:
            raise ValueError(f'Formula contains an empty component: {formula!r}.')
        return counts

    counts = {}
    while index < len(tokens):
        multiplier = number()
        component = sequence()
        for element, count in component.items():
            counts[element] = counts.get(element, 0) + count * multiplier
        if index < len(tokens):
            if tokens[index] not in ('.', '·'):
                raise ValueError(f'Invalid formula syntax: {formula!r}.')
            index += 1
            if index == len(tokens):
                raise ValueError(f'Formula contains an empty component: {formula!r}.')
    return _validate_composition(counts)


def formula_to_string(composition: Mapping) -> str:
    """Format an element-count mapping as a molecular formula string."""
    counts = _validate_composition(composition)
    return ''.join(element + (str(count) if count != 1 else '')
                   for element, count in counts.items())


def enumerate_subformulas(composition: Mapping) -> Iterator[dict]:
    """Iterate over all subformula compositions, including the empty one."""
    counts = _validate_composition(composition)
    items = list(counts.items())
    return ({element: value for (element, _), value in zip(items, reversed(values)) if value}
            for values in product(*(range(count + 1) for _, count in reversed(items))))


def enumerate_subform_arr(form_arr: List[int]) -> np.ndarray:
    """
    enumerate all possible sub-formula arrays of a given formula array
    :param form_arr: a list-like object
    :return: a 2-dim numpy array, with each row being a sub-formula array
    """

    form_arr = np.array(form_arr, dtype=np.int16)

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

    subform_arr = np.zeros((total_subform_cnt, n), dtype=np.int16)
    tempSize = 1

    for i in range(n):
        count = pre_charged_arr[i]
        repeatSize = tempSize
        tempSize *= (count + 1)

        pattern = np.arange(count + 1)

        repeated_pattern = np.empty(repeatSize * len(pattern), dtype=np.int16)
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
    dtype=np.int16)
word_lengths = np.array([len(word) for word in ["C", "H", "Br", "Cl", "F", "I", "K", "N", "Na", "O", "P", "S"]],
                        dtype=np.int16)


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


def read_formula_str(x: str) -> dict:
    """Parse a formula string; equivalent to :func:`read_formula`."""
    return read_formula(x)


def add_formula_str(x: str, y: str) -> dict:
    """Add the atom counts from two validated formula strings."""
    combined = read_formula(x)
    for element, count in read_formula(y).items():
        combined[element] = combined.get(element, 0) + count
    return _validate_composition(combined)
