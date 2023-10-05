# msbuddy
![Maintainer](https://img.shields.io/badge/maintainer-Shipei_Xing-blue)
[![PyPI](https://img.shields.io/pypi/v/msbuddy?color=green)](https://pypi.org/project/msbuddy/)
[![docs](https://readthedocs.org/projects/msbuddy/badge/?version=latest)](https://msbuddy.readthedocs.io/en/latest/?badge=latest)
[![Generic badge](https://img.shields.io/badge/msbuddy-mass_spec_tools-<COLOR>.svg)](https://github.com/Philipbear/msbuddy)

`msbuddy` is developed for molecular formula annotation in mass spectrometry-based small molecule analysis.
* `msbuddy` is able to provide formula annotations for:
  * queries with or without MS1 isotopic pattern 
  * queries with or without MS/MS spectra
  * both positive and negative ionization modes

The minimum requirement for a msbuddy query is a single _m/z_ value and its ionization polarity.

* `msbuddy` enables:
  * candidate space shrinkage via bottom-up MS/MS interrogation
  * accurate formula annotation via deep learning
  * false discovery rate estimation

**The official documentation is available at [msbuddy Documentation](https://msbuddy.readthedocs.io/en/latest/).**

## Installation ([link](https://msbuddy.readthedocs.io/en/latest/install.html))
`msbuddy` is available on PyPI, you can install it via `pip`:
```commandline
pip install msbuddy
```


## Python API ([link](https://msbuddy.readthedocs.io/en/latest/quickstart.html))

As a quick start, we here load a mgf file and annotate the molecular formula for each MS/MS spectrum.
All the configurations are specified in the [`MsbuddyConfig`](https://msbuddy.readthedocs.io/en/latest/pyapi.html#msbuddy.MsbuddyConfig) class.
**Parallel computing** is also supported.

```python
from msbuddy import Msbuddy, MsbuddyConfig

# instantiate a MsbuddyConfig object
msb_config = MsbuddyConfig(ms_instr='orbitrap', # supported: "qtof", "orbitrap" and "fticr"
                                                # highly recommended to specify
                           halogen=False, # whether to consider halogen atoms FClBrI
                           parallel=True, n_cpu=12)

# instantiate a Msbuddy object
msb_engine = Msbuddy(msb_config)

# load data, here we use a mgf file as an example
msb_engine.load_mgf('input_file.mgf')

# annotate molecular formula
msb_engine.annotate_formula()

# retrieve the annotation result summary
result = msb_engine.get_summary()
```

MS/MS spectra can also be loaded via their [USIs](https://www.biorxiv.org/content/10.1101/2020.05.09.086066v2):
```python
# you can load multiple USIs at once
msb_engine.load_usi(['mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036',
                     'mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740037'])
```

## Command-line API ([link](https://msbuddy.readthedocs.io/en/latest/cmdapi.html))

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
or a csv file ([templates files](https://github.com/Philipbear/msbuddy/tree/main/demo)) containing multiple USI strings (QTOF in this example):
```bash
msbuddy -csv <csv_file> -output <output_folder> -ms qtof -details
```

For mgf file or csv file, you can omit the `-output` option, and results will be written in the same folder as the input file.


Here is an example of processing a mgf file from Orbitrap (`-ms orbitrap`) in parallel (`-p`) using 12 cores (`-n_cpu`). Detailed results are written (`-d`).
Halogen atoms (FClBrI) are also considered (`-hal`).
```bash
msbuddy -mgf <mgf_file> -ms orbitrap -p -n_cpu 12 -d -hal
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

You are welcome to directly contact the msbuddy developer & maintainer Shipei Xing (s1xing@health.ucsd.edu).
