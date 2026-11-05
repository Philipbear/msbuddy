Development
===========

Project layout
--------------

The public Python API is exposed by ``msbuddy/__init__.py``. Its entry points
load the relevant modules on demand. The main annotation workflow is organized
as follows:

* ``load.py`` reads input spectra and loads the formula databases and models.
* ``base.py`` defines formulas, spectra, adducts, and metabolic features.
* ``main.py`` coordinates preprocessing, candidate generation, subformula
  assignment, ranking, and batching.
* ``query.py`` searches the formula databases, and ``cand.py`` builds and
  refines candidate formulas.
* ``ml.py`` computes model features, predicts probabilities, and estimates FDR.
* ``export.py`` writes results, while ``main_cmd.py`` provides the command line.
* ``utils.py`` contains formula parsing and enumeration utilities.

The ``tests`` directory contains regression tests, ``benchmarks`` contains local
performance checks, and ``demo`` provides sample input. The historical
``db_prep`` scripts are used for preparing databases rather than the normal
annotation workflow.

Tests
-----

Install the project dependencies, then run from the repository root::

   python -m unittest discover -s tests -v

The regression tests use synthetic inputs and do not download the databases or
models. A full annotation run also requires the database files and LightGBM, as
described in the installation and quick-start guides.

GitHub Actions are currently disabled for this repository. If Actions are
re-enabled, the ``Tests`` workflow runs only when started manually from the
Actions tab; pushes and pull requests do not trigger it. Run the regression
suite and package checks locally before publishing a release. A manual hosted
run also requires GitHub Actions to be available for the account.

Package metadata
----------------

The version is defined in ``msbuddy/__init__.py`` and used by the command line,
documentation, and package builds. Most package metadata is in
``pyproject.toml``; ``setup.py`` supplies license metadata compatible with
Python 3.8's build tools. Build the source distribution and wheel with
``python -m build``.

Performance checks
------------------

Run the synthetic scaling benchmark from the repository root::

   python -m benchmarks.benchmark_workflows

Use ``--help`` to change the number of spectra, peaks, candidates, formula
rows, and repetitions. The benchmark reports median elapsed times after a warmup;
it measures individual operations and uses a stand-in for feature calculation
when timing matrix assembly. It does not measure a complete annotation run or
download any data. Compare timings on the same machine and environment. Initial
startup and the first annotation can take longer because databases and models
are loaded and Numba compiles the numerical functions.
