# msbuddy
[![Generic badge](https://img.shields.io/badge/msbuddy-ver_0.1.1-<COLOR>.svg)](https://github.com/Philipbear/msbuddy)
![Maintainer](https://img.shields.io/badge/maintainer-Shipei_Xing-blue)


`msbuddy` is developed for molecular formula annotation in mass spectrometry-based small molecule analysis.

The entire documentation is available at [msbuddy Documentation](https://msbuddy.readthedocs.io/en/latest/).


## Python API

### Installation
`msbuddy` is available on PyPI, you can install it via `pip`:

```
pip install msbuddy
```

### Quick Start

As a quick start, you can use the following code to annotate molecular formula from a mgf file:
```
from msbuddy import Buddy

# initialize a Buddy object
buddy = Buddy()

# load data, here we use a mgf file as an example
buddy.load_mgf('input_file.mgf')

# annotate molecular formula
buddy.annotate_formula()

# retrieve the annotation result summary
result = buddy.get_summary()
```

To specify the parameter settings, you can:
```
from msbuddy import Buddy, BuddyParamSet

# initialize a BuddyParamSet object
param = BuddyParamSet(ppm = True, ms1_tol = 5, ms2_tol = 10,
                      halogen = False, timeout_secs = 300)
                      
# initialize a Buddy object with the parameter settings
buddy = Buddy(param)
```
The entire list of `BuddyParamSet` is described in the [API documentation](https://msbuddy.readthedocs.io/en/latest/api.html#msbuddy.BuddyParamSet).


MS/MS spectra can also be loaded via their [USIs](https://www.biorxiv.org/content/10.1101/2020.05.09.086066v2):
```
buddy.load_usi('mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000579622')

# you can also load a list of USIs at once
buddy.load_usi(['mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000579622',
                'mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000579623'])
```


## Command-line Interface

```
msbuddy --input input_file.mgf --output output_folder
```

```
msbuddy --help
```


## Citation
[S. Xing et al. BUDDY: molecular formula discovery via bottom-up MS/MS interrogation. **Nature Methods** 2023. DOI: 10.1038/s41592-023-01850-x](https://doi.org/10.1038/s41592-023-01850-x)

## License
This work is licensed under the MIT license.

## Dependency
This project's environment is maintained by conda, [install it first](https://docs.conda.io/en/main/miniconda.html),
and type in the following to create the environment:

`conda env create -f environment.yml -n msbuddy`

Alternatively, you can use the following command to update the environment based on the specifications in the YML file:

`conda env update -f environment.yml -n msbuddy`