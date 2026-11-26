# msbuddy

[![Maintainer](https://img.shields.io/badge/maintainer-Shipei_Xing-blue)](https://scholar.google.ca/citations?user=en0zumcAAAAJ&hl=en)
[![PyPI](https://img.shields.io/pypi/v/msbuddy?color=green)](https://pypi.org/project/msbuddy/)
[![PyPI Downloads](https://static.pepy.tech/badge/msbuddy)](https://pepy.tech/projects/msbuddy)

<p align="center">
  <img src="https://github.com/Philipbear/msbuddy/blob/main/logo/logo.svg" alt="msbuddy logo" height="100"/>
</p>

`msbuddy` annotates molecular formulas for small molecules from mass spectrometry data. It accepts precursor *m/z* alone, MS1 isotope patterns, MS/MS spectra, or combinations of these inputs in positive or negative ionization mode. A precursor *m/z* and polarity are the minimum input.

[Documentation](https://msbuddy.readthedocs.io/en/latest/) · [PyPI](https://pypi.org/project/msbuddy/) · [Version updates and validation](https://msbuddy.readthedocs.io/en/latest/updates.html)

## Install

```bash
pip install msbuddy
```

The model and formula databases (~40 MB total) download automatically on first use. Runs on CPU only; PyTorch and GPU/CUDA are not required. Python 3.8 or newer is supported.

## What's new in 2.0

- **Higher annotation accuracy:** Improved formula ranking across positive and negative modes, diverse adducts/dimers, and halogenated compounds (F, Cl, Br, and I are searched by default).
- **Fast, lightweight CPU runtime:** Scales cleanly across CPU cores with low memory usage—no GPU required.
- **Optional automatic adduct selection (`adduct="auto"`):** Uses your provided adduct by default (or `[M+H]+` / `[M-H]-` when omitted), with opt-in multi-adduct disambiguation when the adduct form is unknown.
- **Batch-level confidence:** Reports per-feature posterior error probability (`posterior_error_probability`) and dataset-wide `batch_q_value` via `estimate_batch_fdr()`.

## Python quick start

### 1. Annotate an MGF file

```python
from msbuddy import Msbuddy, MsbuddyConfig

engine = Msbuddy(MsbuddyConfig(ms_instr="orbitrap"))  # we also support "tof" and "fticr"
engine.load_mgf("input_file.mgf")
engine.annotate_formula()
results = engine.get_summary()
```

Halogens (F, Cl, Br, and I) are searched by default; set `MsbuddyConfig(halogen=False)` only when prior knowledge rules out halogens. See the [demo MGF](demo/input_file.mgf), [quick start](https://msbuddy.readthedocs.io/en/latest/quickstart.html), and [configuration guide](https://msbuddy.readthedocs.io/en/latest/config.html) for more options.

### 2. Annotate from USIs

```python
from msbuddy import Msbuddy, MsbuddyConfig

engine = Msbuddy(MsbuddyConfig(ms_instr="orbitrap"))
engine.load_usi([
    "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036",
    "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740037",
])
engine.annotate_formula()
results = engine.get_summary()
```

### 3. Annotate a list of spectra via Python API

```python
from msbuddy import MetaFeature, Msbuddy, MsbuddyConfig, Spectrum

features = [
    MetaFeature(
        identifier="feature_1",
        mz=181.0707,
        charge=1,
        adduct="[M+H]+",  # optional; defaults to [M+H]+ / [M-H]-, or use "auto"
        ms1=Spectrum([181.0707, 182.0741], [100.0, 6.7]),  # optional MS1 isotope pattern
        ms2=Spectrum([109.0284, 127.0390, 145.0495, 163.0601], [30.0, 60.0, 100.0, 80.0]),
    ),
    MetaFeature(
        identifier="feature_2",
        mz=179.0561,
        charge=-1,
        ms2=Spectrum([89.0244, 101.0244, 119.0350, 161.0455], [100.0, 40.0, 70.0, 30.0]),
    ),
]

engine = Msbuddy(MsbuddyConfig(ms_instr="orbitrap"))
engine.add_data(features)
engine.annotate_formula()
results = engine.get_summary()
```

See the [API reference](https://msbuddy.readthedocs.io/en/latest/pyapi.html) for direct mass-to-formula search (`mass_to_formula`, `mz_to_formula`) and fragment subformula assignment (`assign_subformula`).

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
