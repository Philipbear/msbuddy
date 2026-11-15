Lossless formula database format
================================

The current databases use lossless indexed storage. Version 0.4.0 changes
compression and block size while preserving formula rows, masses, indexes,
ordering and duplicates. Every decoded array retains its dtype and shape,
including the original IEEE 754 mass bits. Sizes and validation results are
in :ref:`update-1-4` and :ref:`update-1-3`.

Storage and loading
-------------------

The ``.msbdb`` files are deterministic ZIP archives containing a versioned
JSON manifest and independently compressed blocks. Version 0.4.0 uses
8,192-row LZMA blocks for the formula database. The manifest records the
arrays, dtypes, dimensions, block layout and SHA-256 checksums.

Integer formula columns use the smallest exact unsigned representation
after subtracting a stored column offset. Constant columns require no row
payload. Floating-point arrays use an exact fixed-point representation only
when the writer proves that decoding restores every original bit; other
values use their original bytes. This includes an exact fallback for values
outside the fixed-point grid, signed zero and nonfinite bit patterns. Masses
are never recalculated from molecular formulas or new atomic mass constants.

The reader loads the small mass and index arrays into memory and decodes
formula rows on demand. Its block-payload cache is bounded at 32 MiB per
database object. File handles and caches are local to each process; worker
processes reopen the archive when needed. The format uses Python's standard
library compression support and adds no compression dependency.

Compatibility and custom databases
----------------------------------

Existing custom ``.joblib`` databases remain supported through ``data_files``.
Standard release downloads use ``.msbdb`` and pinned SHA-256 checksums. An
upgrade downloads the new standard assets under new names; it does not
overwrite a user's legacy or custom database.

To convert a trusted custom legacy database, use::

   python -m msbuddy._database formula.joblib formula.msbdb --role formula
   python -m msbuddy._database common.joblib common.msbdb --role common

Then pass the resulting paths through ``MsbuddyConfig(data_files=...)``.
Legacy joblib files are Python pickle objects and should come from a trusted
source. The numerical archive reader itself does not deserialize Python
objects.
