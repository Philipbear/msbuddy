msbuddy changelog
=================

0.1.2 (2023-10-03)
- First release of a working version on PyPI.

0.2.0 (2023-10-15)
- Update deep learning features & models for molecular formula annotation.
- Update API function 'mass_to_formula', mass errors are output as well. Formula results are sorted by mass errors.
- Add new API function 'mz_to_formula', for charged ions. Ion type (adduct form) is required.
- **Note**: this version outputs different results from previous versions.

0.2.1 (2023-10-18)
- Bug fixes for load_mgf.
- Add mgf demo file.

0.2.2 (2023-10-18)
- Add new API function 'assign_subformula'.

0.2.3 (2023-11-05)
- Update export: pd.Dataframe 'append' deprecated.
