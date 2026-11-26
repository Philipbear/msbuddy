# msbuddy
[![Maintainer](https://img.shields.io/badge/maintainer-Shipei_Xing-blue)](https://scholar.google.ca/citations?user=en0zumcAAAAJ&hl=en)
[![PyPI](https://img.shields.io/pypi/v/msbuddy?color=green)](https://pypi.org/project/msbuddy/)
[![PyPI Downloads](https://static.pepy.tech/badge/msbuddy)](https://pepy.tech/projects/msbuddy)
[![Generic badge](https://img.shields.io/badge/msbuddy-MS_tool-<COLOR>.svg)](https://github.com/Philipbear/msbuddy)

<p align="center">
  <img src="https://github.com/Philipbear/msbuddy/blob/main/logo/logo.svg" alt="Sample Image" height="100"/>
</p>


`msbuddy` is developed for molecular formula analysis in MS-based small molecule (<1500 Da) studies.

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
LightGBM and the CPU ONNX Runtime are inference dependencies. PyTorch and CUDA
are not required. Python 3.8 or newer is supported; Python 3.8 uses NumPy 1.x
and a compatible ONNX Runtime version.

## Unified hybrid engine in 2.0

`msbuddy 2.0` introduces a unified four-stage CPU engine packaged into a single
SHA-256-verified `.joblib` bundle (`21.4 MB`, trained on public libraries only;
no PyTorch or GPU required; `44.4 MB` total with the `23.0 MB` `.msbdb`
formula databases):

- **Stage 0 (Exact-Mass Spectral Transformer + Contrastive Alignment, ONNX):**
  Encodes top-64 exact-mass MS/MS peaks (stacked 7-scale Fourier
  mass/neutral-loss features and local isotope-twin ratios) and 60-dimensional
  query/chemistry metadata to predict per-element distributions, global
  properties, and a 128-d contrastive spectrum embedding.
- **Stage 1–2 (Compiled `nogil` Numba Engine + Batched `FormulaEncoderV3`):**
  Streams `formula_db_v0.3.0.msbdb` into contiguous `int16` memory (~520 MB peak
  RSS) for two-pass candidate generation and contrastive shortlisting, exact
  subformula/neutral-loss assignment (48-d MS/MS physics with MS1–MS2 internal
  self-recalibration), and analytical MS1 isotope scoring across shared-memory
  threads (`parallel=True`, >220 spectra/s on 16–32 CPU cores).
- **Stage 3–4 (FiLM-Conditioned Hybrid Set Reranker + Calibrated Absent-Mass Posteriors):**
  Combines a permutation-equivariant `PureSetRerankerV3` (ONNX, with an explicit
  `[ABSENT]` token) and a `LightGBM` LambdaRank scorer, calibrated via
  two-parameter temperature and absent-bias scaling for posterior probabilities
  and batch FDR. MS1-only, mass-only, multiply-charged, and `use_spectral=False`
  queries fall back lazily to the optional compatibility bundle.
- **Opt-in multi-adduct disambiguation (`adduct="auto"`):** Default behavior is
  unchanged when an adduct is specified or omitted (`[M+H]+`/`[M-H]-`). Passing
  `adduct="auto"` expands each query across a polarity-specific adduct panel
  (including common adducts and dimers), applies Bayesian adduct prevalence
  log-priors to resolve MS/MS-isobaric aliases, normalizes candidates jointly
  against one `[ABSENT]` anchor, and writes the top-ranked adduct back to
  `MetaFeature.adduct`.

## Compact models and databases in 1.3

MS2 and joint MS1+MS2 ranking share one 749,644-parameter spectral encoder
and one candidate scorer. Accurate fragment-mass patterns and soft atom-count
likelihoods complement formula, mass, isotope and fragment evidence.
The default model is **9.37 MB**, replacing the 197 MB all-mode download.
Compatibility weights are downloaded when an older evidence mode or setting
requires them; use `download_data(include_compatibility=True)` for offline use.

The redesigned formula database is **23.0 MB**, down from 436.5 MB, with every
formula, duplicate, index and stored mass bit preserved. Default model and
database downloads total about **32.4 MB**. Existing custom legacy databases
and models remain supported.

On a frozen 4,000-formula evaluation, MS2-only top-1 accuracy improved from
86.55% to 89.63%; joint accuracy improved from 94.38% to 94.95% with simulated
MS1 isotope evidence. A separate 153-formula fluorinated challenge also
improved. Results vary across domains: a smaller external PFAS subset had
lower MS2-only accuracy. See the [complete validation report](https://github.com/Philipbear/msbuddy/blob/main/docs/model_report_1_3.rst)
for paired intervals, missing-metadata results, confidence checks and limitations.

Acquisition metadata are optional. Set `MsbuddyConfig(ms_instr="orbitrap")`
or `ms_instr="tof"` (for QTOF and timsTOF) to configure mass tolerances and provide
the instrument hint. The older `qtof` and `timstof` settings remain supported as aliases for `tof`. `MetaFeature` also accepts `instrument`, `collision_energy`
and `collision_energy_unit` (`"ev"` or `"nce"`). MGF fields
`INSTRUMENT_TYPE` and `COLLISION_ENERGY=30 nce` are recognized. Unknown or
stepped energies are masked; bare numbers need explicit units.

Use `MsbuddyConfig(use_spectral=False)` or `--no-spectral` to select the
established numerical rankers. They are also used for custom candidate budgets
and inputs outside the compact models' validated settings. See the
[model guide](https://github.com/Philipbear/msbuddy/blob/main/docs/models.rst)
for confidence interpretation and supported settings.

## Model update in 1.1

The revised evidence-mode models use explicit elemental and adduct features,
a learned candidate selector, and mode-specific candidate limits.
**Starting with 1.1.1, halogen atoms (F, Cl, Br and I) are included in the search by default.**
This allows both halogen-free and halogen-containing formulas without requiring
prior knowledge of the unknown compound. Only when prior knowledge rules out
halogens should you set `MsbuddyConfig(halogen=False)` or pass `--no-halogen`.
The default fluorine limit is 80 atoms. Set `element_ranges={'F': (0, n)}`
or `--f-max n` when a different limit is justified by prior knowledge.
Use `MsbuddyConfig(max_candidates=350)` or `--max-candidates 350` to override
the recommended candidate budget.

Probability calibration now reserves mass for an absent correct formula.
A singleton candidate no longer becomes certain through normalization.
See the [model guide](https://github.com/Philipbear/msbuddy/blob/main/docs/models.rst)
and [validation report](https://github.com/Philipbear/msbuddy/releases/tag/v1.1.0)
for budgets, confidence interpretation, comparisons and limitations.

## Upgrading to 1.0

The 1.0 Python API contains breaking changes and is not fully compatible with
0.x. Formula compositions now use dictionaries, such as
`{'C': 6, 'H': 12, 'O': 6}`, instead of fixed-order element arrays. `read_formula()` returns a
dictionary, and `Formula()` accepts one. Use `formula_to_string()` and the
lazy `enumerate_subformulas()` iterator in place of the former top-level
array helpers. Invalid formulas and unsupported element symbols raise errors.

Configure element limits with `MsbuddyConfig(element_ranges={'C': (0, 40)})`
instead of separate arguments such as `c_range`. `Msbuddy.add_data()` now
appends data; call `clear_data()` first when replacing the loaded dataset.
See the [migration guide](https://github.com/Philipbear/msbuddy/blob/main/docs/migration.rst) for examples and other changes.

## Python usage 

### Signed differences between network nodes

The delta-formula API annotates **target minus source** with signed atom
counts, including `-H+Na`, without downloading a model or database:

```python
from msbuddy import annotate_delta_mass, annotate_network_delta

changes = annotate_delta_mass(21.98194425, tolerance=0.001)
print(changes[0].formula)  # -H+Na: a possible adduct exchange

edge = annotate_network_delta(
    181.0706645565, 203.0526088063, '[M+H]+', '[M+Na]+',
    source_formula='C6H12O6', preset='broad',
)
print(edge[0].formula, edge.adduct_delta)  # 0, -H+Na
```

The network function accounts for charge and multimers and separates neutral
formula changes from adduct changes. Its search is complete within explicit
bounds; multi-step labels are hypotheses, and ranking scores are not FDR
estimates. See the [delta API guide](docs/delta.rst) for search presets,
mass tolerances, and annotation propagation.

### Quick start ([docs](https://msbuddy.readthedocs.io/en/latest/quickstart.html))
As a quick start, we here load a mgf file and annotate the molecular formula for each query.
All the configurations are specified in the [`MsbuddyConfig`](https://msbuddy.readthedocs.io/en/latest/pyapi.html#msbuddy.MsbuddyConfig) class.
Parallel computing is supported.

```python
from msbuddy import Msbuddy, MsbuddyConfig

# instantiate a MsbuddyConfig object
msb_config = MsbuddyConfig(ms_instr='orbitrap', # supported: "tof", "orbitrap", "fticr" or None
                                                # custom MS1 and MS2 tolerance will be used if None
                           ppm=True,  # use ppm for mass tolerance
                           ms1_tol=5,  # MS1 tolerance in ppm or Da
                           ms2_tol=10,  # MS2 tolerance in ppm or Da
                           halogen=True)  # default: allow F, Cl, Br and I

# instantiate a Msbuddy object
msb_engine = Msbuddy(msb_config)

# load data, here we use a mgf file as an example
msb_engine.load_mgf('input_file.mgf')

# annotate molecular formula
msb_engine.annotate_formula()

# retrieve the annotation result summary
result = msb_engine.get_summary()
```

The result summary reports `posterior_error_probability` for each top formula
and `batch_q_value` across the complete loaded collection. After annotation,
`msb_engine.estimate_batch_fdr(fdr_threshold=0.05)` returns aligned selection
flags and the estimated false fraction of the selected top annotations.
These estimates depend on probability calibration and the application domain.
The legacy `estimated_fdr` field describes a candidate prefix within one
spectrum; at rank one it is the individual error probability, not a batch
q-value. See the [confidence documentation](https://msbuddy.readthedocs.io/en/latest/models.html#probability-and-fdr-interpretation).
See [demo mgf file](https://github.com/Philipbear/msbuddy/tree/main/demo).
One of the following fields is recommended to be included, which will be used as the query identifier: `TITLE`, `FEATURE_ID`, `SPECTRUMID`, or `SPECTRUM_ID`.

If prior knowledge establishes that the compounds cannot contain halogens,
exclude them explicitly:

```python
msb_config = MsbuddyConfig(ms_instr='orbitrap', halogen=False)
```

For direct mass searches, pass `halogen=False` to `mass_to_formula()` or
`mz_to_formula()` under the same condition; these methods also include halogens
by default.


MS/MS spectra can be loaded via their [USIs](https://www.biorxiv.org/content/10.1101/2020.05.09.086066v2) if they are publicly available:
```python
# you can load multiple USIs at once
msb_engine.load_usi(['mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036',
                     'mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740037'])
```

Here is a test case on [Google Colab](https://colab.research.google.com/drive/1oLIapoylGbh37ivhcAN-X4tK6l5kIIpf?usp=sharing).

### Other API functions

```python
from itertools import islice
from msbuddy import Formula, MsbuddyConfig, enumerate_subformulas, formula_to_string, read_formula

composition = read_formula('C6H12O6')  # {'C': 6, 'H': 12, 'O': 6}
formula = Formula(composition)          # neutral by default
print(formula.mass, formula.composition, formula_to_string(composition))
print(list(islice(enumerate_subformulas(composition), 5)))

config = MsbuddyConfig(halogen=True, element_ranges={'C': (0, 40), 'Cl': (0, 2)})
```

Supported elements are C, H, Br, Cl, F, I, K, N, Na, O, P, and S.
Unknown symbols raise `UnsupportedElementError`, a subclass of `ValueError`.
You can also assign subformulas from MS/MS data with `assign_subformula()`, or
use `Msbuddy.mass_to_formula()` and `Msbuddy.mz_to_formula()` to search by mass.

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
msbuddy -csv <csv_file> -output <output_folder> -ms tof -details
```

For mgf file or csv file, you can omit the `-output` option, and results will be written in the same folder as the input file.


Here is an example of processing a mgf file from Orbitrap (`-ms orbitrap`) in parallel (`-p`) using 12 cores (`-n_cpu 12`). Detailed results are written (`-d`).
Halogen atoms (F, Cl, Br and I) are considered by default.
```bash
msbuddy -mgf <mgf_file> -ms orbitrap -p -n_cpu 12 -d
```

Only if prior knowledge rules out halogens, add `--no-halogen`:

```bash
msbuddy --mgf <mgf_file> --ms-instr orbitrap --no-halogen
```

The existing `--halogen`, `-halogen` and `-hal` flags remain supported.

Run the following command to see the full list of options:
```bash
msbuddy --help
```

## Development

The annotation pipeline lives in `msbuddy/`, with regression tests in `tests/`
and repeatable scaling checks in `benchmarks/`. After installing the project
dependencies, run the tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

Run `python -m benchmarks.benchmark_workflows` to measure synthetic input
loading, candidate matching, feature assembly, and subformula mass calculation.
See the [development guide](https://github.com/Philipbear/msbuddy/blob/main/docs/development.rst) for the project layout and
benchmark options.

The [candidate-audit guide](docs/candidate_audit.rst) covers labelled-library
sampling, optional MassDesk fragment search, candidate-recall diagnostics,
and formula-disjoint ranker experiments.

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
Please contact me via **sxing@fudan.edu.cn**.
