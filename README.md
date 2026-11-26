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

The standard public-library model (`ml_v2.0.0.joblib`, 21.4 MB) and lossless formula databases (`v0.4.0`, 18.4 MB) download automatically on first use (~39.8 MB total in version 2.0). Inference runs on CPU via ONNX Runtime and LightGBM; PyTorch and CUDA are not required. Python 3.8 or newer is supported.

## What's new in 2.0

- **Unified four-stage CPU engine (`format_version=5`):** Combines an Exact-Mass Spectral Transformer with isobaric-contrastive alignment (`ONNX`), two-pass compiled `nogil` Numba candidate generation + batched `FormulaEncoderV3` shortlisting, 48-d compiled MS/MS physics (with internal MS1–MS2 self-recalibration) + 12-d analytical MS1 isotope scoring, and a FiLM-conditioned `PureSetRerankerV3` (`ONNX`) + `LightGBM` LambdaRank hybrid with calibrated `[ABSENT]` posteriors.
- **Fast shared-memory parallelism:** Streams `formula_db_v0.4.0.msbdb` directly into contiguous `int16` arrays (~520 MB peak RSS) and executes `parallel=True` across a shared-memory thread pool without duplicating database memory.
- **Opt-in multi-adduct disambiguation (`adduct="auto"`):** Default known-adduct behavior (`[M+H]+`/`[M-H]-` when omitted) is unchanged. Passing `adduct="auto"` on singly charged queries ranks across common polarity-specific adducts and dimers with Bayesian prevalence log-priors against one shared `[ABSENT]` anchor and writes the top-ranked adduct back to `MetaFeature.adduct`.
- **Pooled batch FDR & signed delta-formula APIs:** `get_summary()` and `estimate_batch_fdr()` report `posterior_error_probability` and collection-wide `batch_q_value`, and `annotate_delta_mass()` / `annotate_network_delta()` annotate signed elemental differences between molecular-network nodes without downloading a model or database.
- **Full evidence-mode fallback:** MS1-only, mass-only, multiply-charged, and `use_spectral=False` queries lazily load the optional compatibility bundle (`ml_compatibility_v1.3.0.joblib`).

## Python quick start

```python
from msbuddy import Msbuddy, MsbuddyConfig

engine = Msbuddy(MsbuddyConfig(ms_instr="orbitrap"))
engine.load_mgf("input_file.mgf")
engine.annotate_formula()
results = engine.get_summary()
```

Use `ms_instr="tof"` for both QTOF and timsTOF, or `"fticr"` as appropriate. The older `"qtof"` and `"timstof"` settings remain accepted aliases for `"tof"`. Acquisition metadata are optional. See the [quick start](https://msbuddy.readthedocs.io/en/latest/quickstart.html) and [configuration guide](https://msbuddy.readthedocs.io/en/latest/config.html) for other inputs and settings.

Halogens (F, Cl, Br, and I) are searched by default. Set `MsbuddyConfig(halogen=False)` only when prior knowledge rules out halogens. Reported confidence values (`posterior_error_probability`, `batch_q_value`, and legacy `estimated_fdr`) are model-based estimates; see [model settings and confidence](https://msbuddy.readthedocs.io/en/latest/models.html) for details.

MS/MS spectra can also be loaded by [USI](https://msbuddy.readthedocs.io/en/latest/datain.html). A [demo MGF](demo/input_file.mgf) and [API reference](https://msbuddy.readthedocs.io/en/latest/pyapi.html) are available. The API also supports direct formula mass searches, fragment subformula annotation, and signed molecular-network delta annotation:

```python
from msbuddy import annotate_delta_mass, annotate_network_delta

changes = annotate_delta_mass(21.98194425, tolerance=0.001)
print(changes[0].formula)  # -H+Na

edge = annotate_network_delta(
    181.0706645565, 203.0526088063, "[M+H]+", "[M+Na]+",
    source_formula="C6H12O6", preset="broad",
)
print(edge[0].formula, edge.adduct_delta)  # 0 -H+Na
```

## Command line

```bash
msbuddy -mgf input_file.mgf -output results -ms orbitrap -details
```

Run `msbuddy --help` for options. The [command-line guide](https://msbuddy.readthedocs.io/en/latest/cmdapi.html) covers USIs, batch input, parallel processing, and exports.

## Upgrading and contributing

See the [migration guide](https://msbuddy.readthedocs.io/en/latest/migration.html) when upgrading older scripts. Release details are in the [version updates](https://msbuddy.readthedocs.io/en/latest/updates.html) and [changelog](changelog.md).

The annotation pipeline is in `msbuddy/` and tests are in `tests/`. Run `python -m unittest discover -s tests -v` after installing development dependencies. See the [development guide](https://msbuddy.readthedocs.io/en/latest/development.html) and [candidate-audit guide](docs/candidate_audit.rst).

## Citation

S. Xing et al. BUDDY: molecular formula discovery via bottom-up MS/MS interrogation. *Nature Methods* (2023). [DOI: 10.1038/s41592-023-01850-x](https://doi.org/10.1038/s41592-023-01850-x)

## License and contact

Apache License 2.0. Contact: sxing@fudan.edu.cn.
