# msbuddy
![Maintainer](https://img.shields.io/badge/maintainer-Shipei_Xing-blue)
[![PyPI](https://img.shields.io/pypi/v/msbuddy?color=green)](https://pypi.org/project/msbuddy/)
[![docs](https://readthedocs.org/projects/msbuddy/badge/?version=latest)](https://msbuddy.readthedocs.io/en/latest/?badge=latest)
[![Generic badge](https://img.shields.io/badge/msbuddy-mass_spec_tools-<COLOR>.svg)](https://github.com/Philipbear/msbuddy)

.
## msbuddy is in final training process and will be released soon.
.


`msbuddy` is developed for molecular formula annotation in mass spectrometry-based small molecule analysis.
* `msbuddy` is able to provide formula annotations for:
  * queries with or without MS1 isotopic pattern 
  * queries with or without MS/MS spectra
  * both positive and negative ionization modes
The minimum requirement is a single _m/z_ value and its ionization polarity.
* `msbuddy` enables:
  * candidate space shrinkage via bottom-up MS/MS interrogation
  * accurate formula annotation via machine learning
  * false discovery rate estimation

**The entire documentation is available at [msbuddy Documentation](https://msbuddy.readthedocs.io/en/latest/).**

## Python API

### Installation
`msbuddy` is available on PyPI, you can install it via `pip`:

```commandline
pip install msbuddy
```

### Quick Start ([link](https://msbuddy.readthedocs.io/en/latest/quickstart.html))

As a quick start, we here load a mgf file and annotate the molecular formula for each MS/MS spectrum.
All the parameter settings are specified in the [`BuddyParamSet`](https://msbuddy.readthedocs.io/en/latest/pyapi.html#msbuddy.BuddyParamSet) class.
**Parallel computing** is also supported.

```python
from msbuddy import Buddy, BuddyParamSet

# instantiate a BuddyParamSet object
param = BuddyParamSet(ms_instr='orbitrap', # supported: "qtof", "orbitrap" and "fticr"
                                           # highly recommended to specify
                      halogen=False, timeout_secs=300,
                      parallel=True, n_cpu=12)

# instantiate a Buddy object
buddy = Buddy(param)

# load data, here we use a mgf file as an example
buddy.load_mgf('input_file.mgf')

# annotate molecular formula
buddy.annotate_formula()

# retrieve the annotation result summary
results = buddy.get_summary()
```

MS/MS spectra can also be loaded via their [USIs](https://www.biorxiv.org/content/10.1101/2020.05.09.086066v2):
```python
buddy.load_usi('mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036')

# you can also load a list of USIs at once
buddy.load_usi(['mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036',
                'mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740037'])
```

## Command-line API ([link](https://msbuddy.readthedocs.io/en/latest/cmdapi.html))

**msbuddy** can also be used as a command-line tool.
In the result summary, top 5 formula candidates will be reported for each query.
The annotation details can be output with the `-details` option (or `-d` for short).
Here is an example command for a mgf file:
```bash
msbuddy -mgf <mgf_file> -output <output_folder> -details
```
or for a csv file containing USI strings (one USI per line):
```bash
msbuddy -csv <csv_file> -output <output_folder>
```
Run the following command to see the full list of options:
```bash
msbuddy --help
```

## Links
[**msbuddy Documentation**](https://msbuddy.readthedocs.io/en/latest/)

[PyPI release](https://pypi.org/project/msbuddy/)

[GitHub repository](https://github.com/Philipbear/msbuddy)

## Citation
S. Xing et al. BUDDY: molecular formula discovery via bottom-up MS/MS interrogation. **Nature Methods** 2023. [DOI: 10.1038/s41592-023-01850-x](https://doi.org/10.1038/s41592-023-01850-x)

## License
This work is licensed under the Apache License 2.0.

## Contact
To contribute to `msbuddy`, please feel free to [file an issue](https://github.com/Philipbear/msbuddy/issues), or submit a pull request with improvements.

You are always welcome to directly contact the msbuddy developer & maintainer Shipei Xing (s1xing@health.ucsd.edu).
