msbuddy changelog
=================

0.1.2 (2023-10-03)
- First release of a working version on PyPI.

0.2.0 (2023-10-15)
- Update deep learning features & models for molecular formula annotation.
- Update API function 'mass_to_formula', mass errors are output as well. Formula results are sorted by mass errors.
- Add new API function 'mz_to_formula', for charged ions. Ion type (adduct form) is required.

0.2.1 (2023-10-18)
- Bug fixes for load_mgf.
- Add mgf demo file.

0.2.2 (2023-10-18)
- Add new API function 'assign_subformula'.

0.2.3 (2023-11-05)
- Update export: pd.Dataframe 'append' deprecated.

0.2.4 (2023-11-09, stable version)
- Bug fixes for export results in cmd line version.
- Add version control for downloaded data (loaded DBs).

0.2.5 (2023-11-22)
- Update MsbuddyConfig class and PreprocessedMS2 class.

0.2.6 / 0.2.7 (2023-12-09)
- Update base class "Adduct".

0.2.8 (2023-12-10)
- Bug fix for load_mgf (now can load metabolite features with MS1 and MS2 spectra using the same identifier).
