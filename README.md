# msbuddy

[![Maintainer](https://img.shields.io/badge/maintainer-Shipei_Xing-blue)](https://scholar.google.ca/citations?user=en0zumcAAAAJ&hl=en)
[![PyPI](https://img.shields.io/pypi/v/msbuddy?color=green)](https://pypi.org/project/msbuddy/)
[![PyPI Downloads](https://static.pepy.tech/badge/msbuddy)](https://pepy.tech/projects/msbuddy)

<p align="center">
  <img src="https://github.com/Philipbear/msbuddy/blob/main/logo/logo.svg" alt="msbuddy logo" height="100"/>
</p>

`msbuddy` annotates molecular formulas for small molecules (below 1500 Da) from mass spectrometry data. It accepts precursor m/z alone, MS1 isotope patterns, MS/MS spectra, or combinations of these inputs in positive or negative ionization mode. A precursor m/z and polarity are the minimum input.

[Documentation](https://msbuddy.readthedocs.io/en/latest/) · [PyPI](https://pypi.org/project/msbuddy/) · [Version updates and validation](https://msbuddy.readthedocs.io/en/latest/updates.html)

## Install

```bash
pip install msbuddy
```

The standard model and formula databases download on first use (about 27 MB in version 1.4). Inference runs on CPU; PyTorch and CUDA are not required.

## Python quick start

```python
from msbuddy import Msbuddy, MsbuddyConfig

engine = Msbuddy(MsbuddyConfig(ms_instr="orbitrap"))
engine.load_mgf("input_file.mgf")
engine.annotate_formula()
results = engine.get_summary()
```

Use `ms_instr="qtof"`, `"timstof"`, or `"fticr"` as appropriate. Acquisition metadata are optional. See the [quick start](https://msbuddy.readthedocs.io/en/latest/quickstart.html) and [configuration guide](https://msbuddy.readthedocs.io/en/latest/config.html) for other inputs and settings.

Halogens are searched by default. Set `MsbuddyConfig(halogen=False)` only when prior knowledge rules out F, Cl, Br, and I. Reported confidence values are model-based estimates; [validation and limitations](https://msbuddy.readthedocs.io/en/latest/updates.html) explain their use.

MS/MS spectra can also be loaded by [USI](https://msbuddy.readthedocs.io/en/latest/datain.html). A [demo MGF](demo/input_file.mgf) and [API reference](https://msbuddy.readthedocs.io/en/latest/pyapi.html) are available. The API also supports direct formula mass searches and fragment subformula annotation.

## Command line

```bash
msbuddy -mgf input_file.mgf -output results -ms orbitrap -details
```

Run `msbuddy --help` for options. The [command-line guide](https://msbuddy.readthedocs.io/en/latest/cmdapi.html) covers USIs, batch input, parallel processing, and exports.

## Upgrading and contributing

See the [migration guide](https://msbuddy.readthedocs.io/en/latest/migration.html) when upgrading older scripts. Release details are in the [version updates](https://msbuddy.readthedocs.io/en/latest/updates.html) and [changelog](changelog.md).

The annotation pipeline is in `msbuddy/` and tests are in `tests/`. Run `python -m unittest discover -s tests -v` after installing development dependencies. See the [development guide](https://msbuddy.readthedocs.io/en/latest/development.html).

## Citation

S. Xing et al. BUDDY: molecular formula discovery via bottom-up MS/MS interrogation. *Nature Methods* (2023). [DOI: 10.1038/s41592-023-01850-x](https://doi.org/10.1038/s41592-023-01850-x)

## License and contact

Apache License 2.0. Contact: sxing@fudan.edu.cn.
