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
  * metabolic features with or without MS1 isotopic pattern 
  * metabolic features with or without MS/MS spectra
  * both positive and negative ionization modes
* `msbuddy` enables:
  * candidate space shrinkage via bottom-up MS/MS interrogation
  * accurate formula annotation via machine learning
  * false discovery rate estimation

## Python API

**The entire documentation is available at [msbuddy Documentation](https://msbuddy.readthedocs.io/en/latest/).**

### Installation
`msbuddy` is available on PyPI, you can install it via `pip`:

```commandline
pip install msbuddy
```

### Quick Start

As a quick start, we here load a mgf file and annotate the molecular formula for each MS/MS spectrum:
```python
from msbuddy import Buddy

# instantiate a Buddy object
buddy = Buddy()

# load data, here we use a mgf file as an example
buddy.load_mgf('input_file.mgf')

# annotate molecular formula
buddy.annotate_formula()

# retrieve the annotation result summary
result = buddy.get_summary()
```

To specify the parameter settings, you can use the [`BuddyParamSet`](https://msbuddy.readthedocs.io/en/latest/pyapi.html#msbuddy.BuddyParamSet) object:
```python
from msbuddy import Buddy, BuddyParamSet

# instantiate a BuddyParamSet object
param = BuddyParamSet(ppm = True, ms1_tol = 5, ms2_tol = 10,
                      halogen = False, timeout_secs = 300)
                      
# instantiate a Buddy object with the specified parameter settings
buddy = Buddy(param)
```


MS/MS spectra can also be loaded via their [USIs](https://www.biorxiv.org/content/10.1101/2020.05.09.086066v2):
```python
buddy.load_usi('mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036')

# you can also load a list of USIs at once
buddy.load_usi(['mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036',
                'mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740037'])
```

## Command-line API

**msbuddy** can also be used as a command-line tool:
```bash
msbuddy --mgf <mgf file> --output <output folder>
```
or for a csv file containing USI strings (one USI per line):
```bash
msbuddy --csv <csv file> --output <output folder>
```
Run the following command to see the full list of options:
```bash
msbuddy --help
```

## Documentation
Link to [**msbuddy Documentation**](https://msbuddy.readthedocs.io/en/latest/)

## Citation
[S. Xing et al. BUDDY: molecular formula discovery via bottom-up MS/MS interrogation.](https://doi.org/10.1038/s41592-023-01850-x) **Nature Methods** 2023. DOI: 10.1038/s41592-023-01850-x

## License
This work is licensed under the MIT license.

## Contact
To contribute to `msbuddy`, please feel free to [file an issue](https://github.com/Philipbear/msbuddy/issues), or submit a pull request with improvements.

You are also welcome to directly contact the maintainer Shipei Xing (philipxsp@hotmail.com).
