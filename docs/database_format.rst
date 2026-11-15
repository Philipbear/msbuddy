Lossless formula database format
================================

The version 0.4.0 databases store the same formula rows, masses, indexes,
ordering and duplicate entries as version 0.3.0. They change compression and
block size rather than the chemical contents.

The formula asset is 18,405,871 bytes, compared with 22,990,650 bytes in
version 0.3.0 and 436,457,267 bytes in the legacy asset. The common-fragment
and neutral-loss asset remains 2,496 bytes. Every decoded array retains its
original dtype and shape. Every stored mass restores the original IEEE 754 bit
pattern.

Storage and loading
-------------------

The ``.msbdb`` files are deterministic ZIP archives containing a versioned
JSON manifest and independently compressed blocks. Version 0.4.0 uses
8,192-row LZMA blocks for the formula database. The manifest records the
arrays, dtypes, dimensions, block layout and SHA-256
checksums.

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

Validation and runtime
----------------------

Release checks compare every decoded array against the previous database,
including all 4,194,390 formula rows, both complete mass arrays, both index
arrays and the common-fragment and neutral-loss arrays. The earlier 0.3.0
format was also checked through precursor and fragment searches, radicals,
adducts, duplicates, indexed mass boundaries, and threaded and spawned-process
access. Those query-path tests are not repeated here for the 0.4.0 compression
change.

For the 0.4.0 compression change, a 64-spectrum matched annotation check
reproduced every candidate order and probability exactly. Its measured
annotation times were 17.52 seconds with the 0.3.0 database and 16.41 seconds
with 0.4.0 on one machine; these single runs include other inference work and
do not establish a speedup. Random indexed reads can take longer with LZMA.

For the earlier 0.3.0 redesign, five fresh-process trials per format on one
machine found that median peak process
memory during database loading decreased from 535 MB to 223 MB. Median loader
time increased from 0.067 to 0.198 seconds. These measurements exclude model
loading and candidate generation; operating-system file caches were not
controlled. A separate 8,644-query check preserved all results and found
slightly higher individual lookup latency with lazy decoding. These are
loader and lookup measurements, not an end-to-end speed claim.
