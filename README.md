# msbuddy
[![Generic badge](https://img.shields.io/badge/msbuddy-ver_0.0.1-<COLOR>.svg)](https://github.com/Philipbear/msbuddy)
![Maintainer](https://img.shields.io/badge/maintainer-Shipei_Xing-blue)

`msbuddy` is a tool developed for de novo molecular formula annotation in mass spectrometry-based small molecule analysis.

## Python API

### Installation
`msbuddy` is available on PyPI, you can install it via `pip`:

```
pip install msbuddy
```

### Usage


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