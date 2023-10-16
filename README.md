# msbuddy
![Maintainer](https://img.shields.io/badge/maintainer-Shipei_Xing-blue)
[![PyPI](https://img.shields.io/pypi/v/msbuddy?color=green)](https://pypi.org/project/msbuddy/)
[![docs](https://readthedocs.org/projects/msbuddy/badge/?version=latest)](https://msbuddy.readthedocs.io/en/latest/?badge=latest)
[![Generic badge](https://img.shields.io/badge/msbuddy-mass_spec_tools-<COLOR>.svg)](https://github.com/Philipbear/msbuddy)

`msbuddy` is developed for molecular formula annotation in mass spectrometry-based small molecule analysis.
`msbuddy` is able to provide formula annotations for queries:
  * with or without MS1 isotopic pattern 
  * with or without MS/MS spectra
  * in either positive or negative ionization mode

The minimum requirement for a msbuddy query is a single _m/z_ value and its ionization polarity.

**Official documentation**  is available at https://msbuddy.readthedocs.io/en/latest/.

**msbuddy ver 0.2.0** is now released, with refined models and much improved performance. See [change log](https://github.com/Philipbear/msbuddy/blob/main/changelog.md) for details.

##  Citation
> S. Xing et al. BUDDY: molecular formula discovery via bottom-up MS/MS interrogation. **Nature Methods** 2023. [DOI: 10.1038/s41592-023-01850-x](https://doi.org/10.1038/s41592-023-01850-x)



## Installation
`msbuddy` is available on PyPI, you can install the latest version via `pip`:
```commandline
pip install msbuddy
```

To install a specific version, see PyPI [release history](https://pypi.org/project/msbuddy/#history).

## Python usage ([docs](https://msbuddy.readthedocs.io/en/latest/quickstart.html))

As a quick start, we here load a mgf file and annotate the molecular formula for each query.
All the configurations are specified in the [`MsbuddyConfig`](https://msbuddy.readthedocs.io/en/latest/pyapi.html#msbuddy.MsbuddyConfig) class.
**Parallel computing** is supported.

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

MS/MS spectra can be loaded via their [USIs](https://www.biorxiv.org/content/10.1101/2020.05.09.086066v2) if they are publicly available:
```python
# you can load multiple USIs at once
msb_engine.load_usi(['mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036',
                     'mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740037'])
```
**msbuddy** also provides python APIs for generating molecular formulas from [a neutral mass](https://msbuddy.readthedocs.io/en/latest/pyapi.html#mass_to_formula) or [a charged _m/z_ value](https://msbuddy.readthedocs.io/en/latest/pyapi.html#mz_to_formula).

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
or a csv file ([templates files](https://github.com/Philipbear/msbuddy/tree/main/demo)) containing multiple USI strings (QTOF in this example):
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
[**msbuddy Documentation**](https://msbuddy.readthedocs.io/en/latest/)

[PyPI release](https://pypi.org/project/msbuddy/)

[GitHub repository](https://github.com/Philipbear/msbuddy)

## License
This work is licensed under the Apache License 2.0.

## Contact
You are welcome to directly contact the msbuddy developer & maintainer Shipei Xing via **s1xing@health.ucsd.edu** or **philipxsp@hotmail.com**.

To contribute to `msbuddy`, please feel free to [file an issue](https://github.com/Philipbear/msbuddy/issues), or submit a pull request with improvements.
