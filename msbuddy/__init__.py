"""Public API for molecular formula annotation and related utilities.

The scientific stack is loaded when an API object is first requested, so
inspecting package metadata does not initialize the entire annotation pipeline.
"""

from importlib import import_module as _import_module

_PUBLIC_API = {
    'assign_subformula': ('.cand', 'assign_subformula'),
    'Formula': ('.base', 'Formula'),
    'Spectrum': ('.base', 'Spectrum'),
    'Adduct': ('.base', 'Adduct'),
    'MetaFeature': ('.base', 'MetaFeature'),
    'Msbuddy': ('.main', 'Msbuddy'),
    'MsbuddyConfig': ('.main', 'MsbuddyConfig'),
    'download_data': ('.load', 'download_data'),
    'get_data_files': ('.load', 'get_data_files'),
    'read_formula': ('.utils', 'read_formula'),
    'formula_to_string': ('.utils', 'formula_to_string'),
    'enumerate_subformulas': ('.utils', 'enumerate_subformulas'),
    'SUPPORTED_ELEMENTS': ('.utils', 'SUPPORTED_ELEMENTS'),
    'UnsupportedElementError': ('.utils', 'UnsupportedElementError'),
}

# These submodules were also exposed by the original eager imports.
_PUBLIC_MODULES = ('utils', 'base', 'query', 'ml', 'cand', 'export', 'load', 'main')
__all__ = [*_PUBLIC_MODULES, *_PUBLIC_API]


def __getattr__(name):
    if name in _PUBLIC_API:
        module_name, attribute = _PUBLIC_API[name]
        value = getattr(_import_module(module_name, __name__), attribute)
    elif name in _PUBLIC_MODULES:
        value = _import_module('.' + name, __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

__version__ = "1.1.1"
__author__ = "Shipei Xing"
__email__ = "sxing@fudan.edu.cn"
__author_email__ = "sxing@fudan.edu.cn"

__copyright__ = "Copyright 2024-2026 Shipei Xing"
__license_expression__ = "Apache 2.0"

__url__ = "https://github.com/Philipbear/msbuddy"
__docs__ = "https://msbuddy.readthedocs.io/en/latest"

__title__ = "msbuddy"
