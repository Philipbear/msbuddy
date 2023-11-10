# msbuddy
![Maintainer](https://img.shields.io/badge/maintainer-Shipei_Xing-blue)
[![PyPI](https://img.shields.io/pypi/v/msbuddy?color=green)](https://pypi.org/project/msbuddy/)
[![docs](https://readthedocs.org/projects/msbuddy/badge/?version=latest)](https://msbuddy.readthedocs.io/en/latest/?badge=latest)
[![Generic badge](https://img.shields.io/badge/msbuddy-mass_spec_tools-<COLOR>.svg)](https://github.com/Philipbear/msbuddy)

<p align="center">
  <img src="https://github.com/Philipbear/msbuddy/blob/main/logo/logo.svg" alt="Sample Image" height="100"/>
</p>


`msbuddy` is developed for molecular formula analysis in MS-based small molecule studies.

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

To install a specific version, see PyPI [release history](https://pypi.org/project/msbuddy/#history).

## Python usage 

### Quick start ([docs](https://msbuddy.readthedocs.io/en/latest/quickstart.html))
As a quick start, we here load a mgf file and annotate the molecular formula for each query.
All the configurations are specified in the [`MsbuddyConfig`](https://msbuddy.readthedocs.io/en/latest/pyapi.html#msbuddy.MsbuddyConfig) class.
**Parallel computing** is [supported](https://msbuddy.readthedocs.io/en/latest/quickstart.html).

```python
from msbuddy import Msbuddy, MsbuddyConfig

# instantiate a MsbuddyConfig object
msb_config = MsbuddyConfig(# highly recommended to specify
                           ms_instr='orbitrap',  # supported: "qtof", "orbitrap" and "fticr"
                           # whether to consider halogen atoms FClBrI
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


MS/MS spectra can be loaded via their [USIs](https://www.biorxiv.org/content/10.1101/2020.05.09.086066v2) if they are publicly available:
```python
# you can load multiple USIs at once
msb_engine.load_usi(['mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036',
                     'mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740037'])
```

### Other API functions
- assign subformulas for an MS/MS with a given precursor formula ([assign_subformula](https://msbuddy.readthedocs.io/en/latest/pyapi.html#assign_subformula))
- enumerate all possible subformulas of a given precursor formula ([enumerate_subform_arr](https://msbuddy.readthedocs.io/en/latest/pyapi.html#enumerate_subform_arr))
- generate molecular formulas from a neutral mass ([mass_to_formula](https://msbuddy.readthedocs.io/en/latest/pyapi.html#mass_to_formula))
- generate molecular formulas from a charged _m/z_ ([mz_to_formula](https://msbuddy.readthedocs.io/en/latest/pyapi.html#mz_to_formula))
- predict formula feasibility using a deep learning model ([predict_formula_feasibility](https://msbuddy.readthedocs.io/en/latest/pyapi.html#predict_formula_feasibility))

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
Please contact the msbuddy developer & maintainer Shipei Xing via **s1xing@health.ucsd.edu** or **philipxsp@hotmail.com**.
