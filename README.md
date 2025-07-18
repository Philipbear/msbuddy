# msbuddy
[![Maintainer](https://img.shields.io/badge/maintainer-Shipei_Xing-blue)](https://scholar.google.ca/citations?user=en0zumcAAAAJ&hl=en)
[![PyPI](https://img.shields.io/pypi/v/msbuddy?color=green)](https://pypi.org/project/msbuddy/)
[![docs](https://readthedocs.org/projects/msbuddy/badge/?version=latest)](https://msbuddy.readthedocs.io/en/latest/?badge=latest)
[![PyPI Downloads](https://static.pepy.tech/badge/msbuddy)](https://pepy.tech/projects/msbuddy)
[![Generic badge](https://img.shields.io/badge/msbuddy-MS_tool-<COLOR>.svg)](https://github.com/Philipbear/msbuddy)

<p align="center">
  <img src="https://github.com/Philipbear/msbuddy/blob/main/logo/logo.svg" alt="Sample Image" height="100"/>
</p>


`msbuddy` is developed for molecular formula analysis in MS-based small molecule (<1500 Da) studies.

`msbuddy` is able to provide formula annotations for queries:
  * with or without MS1 isotopic pattern 
  * with or without MS/MS spectra
  * in either positive or negative ionization mode

The minimum requirement for a msbuddy query is a single _m/z_ value and its ionization polarity.

**Official documentation**  is available at https://msbuddy.readthedocs.io/en/latest/.



## Installation
`msbuddy` is available on PyPI, you can install the latest version via `pip`:
```commandline
pip install msbuddy
```
To perform formula annotation, you also have to install the [lightgbm](https://github.com/microsoft/LightGBM/tree/master/python-package) package.
```commandline
pip install lightgbm
```
``
Note: Python version >= 3.9 and <3.13 is required.
``

## Python usage 

### Quick start ([docs](https://msbuddy.readthedocs.io/en/latest/quickstart.html))
As a quick start, we here load a mgf file and annotate the molecular formula for each query.
All the configurations are specified in the [`MsbuddyConfig`](https://msbuddy.readthedocs.io/en/latest/pyapi.html#msbuddy.MsbuddyConfig) class.
Parallel computing is supported.

```python
from msbuddy import Msbuddy, MsbuddyConfig

# instantiate a MsbuddyConfig object
msb_config = MsbuddyConfig(ms_instr='orbitrap', # supported: "qtof", "orbitrap", "fticr" or None
                                                # custom MS1 and MS2 tolerance will be used if None
                           ppm=True,  # use ppm for mass tolerance
                           ms1_tol=5,  # MS1 tolerance in ppm or Da
                           ms2_tol=10,  # MS2 tolerance in ppm or Da
                           halogen=False)

# instantiate a Msbuddy object
msb_engine = Msbuddy(msb_config)

# load data, here we use a mgf file as an example
msb_engine.load_mgf('input_file.mgf')

# annotate molecular formula
msb_engine.annotate_formula()

# retrieve the annotation result summary
result = msb_engine.get_summary()
```
See [demo mgf file](https://github.com/Philipbear/msbuddy/tree/main/demo).
One of the following fields is recommended to be included, which will be used as the query identifier: `TITLE`, `FEATURE_ID`, `SPECTRUMID`, or `SPECTRUM_ID`.


MS/MS spectra can be loaded via their [USIs](https://www.biorxiv.org/content/10.1101/2020.05.09.086066v2) if they are publicly available:
```python
# you can load multiple USIs at once
msb_engine.load_usi(['mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036',
                     'mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740037'])
```

Here is a test case on [Google Colab](https://colab.research.google.com/drive/1oLIapoylGbh37ivhcAN-X4tK6l5kIIpf?usp=sharing).

### Other API functions
- parse a formula string (e.g., C6H12O6, 2H2O, C5H7NO2.HCl) ([read_formula_str](https://msbuddy.readthedocs.io/en/latest/pyapi.html#read_formula_str))
- assign subformulas for an MS/MS with a given precursor formula ([assign_subformula](https://msbuddy.readthedocs.io/en/latest/pyapi.html#assign_subformula))
- enumerate all possible subformulas of a given precursor formula ([enumerate_subform_arr](https://msbuddy.readthedocs.io/en/latest/pyapi.html#enumerate_subform_arr))
- generate molecular formulas from a neutral mass ([mass_to_formula](https://msbuddy.readthedocs.io/en/latest/pyapi.html#mass_to_formula))
- generate molecular formulas from a charged _m/z_ ([mz_to_formula](https://msbuddy.readthedocs.io/en/latest/pyapi.html#mz_to_formula))

See more in the [documentation](https://msbuddy.readthedocs.io/en/latest/pyapi.html).

## Command-line tool ([docs](https://msbuddy.readthedocs.io/en/latest/cmdapi.html))

**msbuddy** can also be used as a command-line tool.
Result summary will be generated in a tsv file.
More detailed annotation results can be written with the `-details` option (or `-d` for short).

Here is an example command for a mgf file from Orbitrap:
```bash
msbuddy -mgf <mgf_file> -output <output_folder> -ms orbitrap -details
```
for a single USI string (from FT-ICR, for example):
```bash
msbuddy -usi <usi_string> -output <output_folder> -ms fticr
```
or a csv file ([template files](https://github.com/Philipbear/msbuddy/tree/main/demo)) containing multiple USI strings (QTOF in this example):
```bash
msbuddy -csv <csv_file> -output <output_folder> -ms qtof -details
```

For mgf file or csv file, you can omit the `-output` option, and results will be written in the same folder as the input file.


Here is an example of processing a mgf file from Orbitrap (`-ms orbitrap`) in parallel (`-p`) using 12 cores (`-n_cpu 12`). Detailed results are written (`-d`).
Halogen atoms (FClBrI) are also considered (`-hal`).
```bash
msbuddy -mgf <mgf_file> -ms orbitrap -p -n_cpu 12 -d -hal
```

Run the following command to see the full list of options:
```bash
msbuddy --help
```

## Links
[**msbuddy documentation**](https://msbuddy.readthedocs.io/en/latest/)

[PyPI release](https://pypi.org/project/msbuddy/)

[Change log](https://github.com/Philipbear/msbuddy/blob/main/changelog.md)

[GitHub repository](https://github.com/Philipbear/msbuddy)


##  Citation
> S. Xing et al. BUDDY: molecular formula discovery via bottom-up MS/MS interrogation. **Nature Methods** 2023. [DOI: 10.1038/s41592-023-01850-x](https://doi.org/10.1038/s41592-023-01850-x)


## License
This work is licensed under the Apache License 2.0.

## Contact
Please contact me via **philipxsp@hotmail.com**.
