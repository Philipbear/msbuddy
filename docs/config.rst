Configuration
------------------

All **msbuddy** parameters are stored in a single class called :class:`msbuddy.MsbuddyConfig`.

Here is a quick example of how to create a :class:`msbuddy.MsbuddyConfig` object and pass it to a :class:`msbuddy.Msbuddy` object.
Of note, parallel processing is available for **msbuddy**. To use it, set ``parallel=True`` and specify the number of CPUs to use with ``n_cpu``.
We would recommend considering parallel processing for large datasets (e.g. >1000 query metabolic features).

.. code-block:: python

    from msbuddy import Msbuddy, MsbuddyConfig

    # create a config object
    msb_config = MsbuddyConfig(
        ms_instr='orbitrap', # highly recommended to fill for the best annotation performance
                             # supported MS types: "tof", "orbitrap" and "fticr"
        ppm=True, # use ppm for mass tolerance, otherwise use Da
        ms1_tol=5, # ms1 tolerance, see below for default values for each MS instrument
        ms2_tol=10, # ms2 tolerance, see below for default values for each MS instrument
        halogen=True, # default: allow F, Cl, Br and I
        # parallel=True, # enable parallel processing, see note below
        # n_cpu=12, # number of CPUs to use
        timeout_secs=600, # timeout for each query in seconds
        batch_size=1000, # number of queries to process in each batch, a larger batch size will use more memory but will be faster
        element_ranges={'C': (0, 100), 'H': (0, 150)}, # inclusive atom-count limits
        ... # other parameters
        )

    # create a Msbuddy object with the specified configuration
    msb_engine = Msbuddy(msb_config)


It is **highly recommended** to set up ``ms_instr`` in the :class:`msbuddy.MsbuddyConfig` to obtain the best annotation performance. The supported MS types are ``"tof"``, ``"orbitrap"`` and ``"fticr"``. Both QTOF and timsTOF use ``"tof"``; the older ``"qtof"`` and ``"timstof"`` inputs remain accepted aliases.

The following mass tolerances will be used for each MS instrument:

``tof``: ms1 tolerance = 10 ppm, ms2 tolerance = 20 ppm

``orbitrap``: ms1 tolerance = 5 ppm, ms2 tolerance = 10 ppm

``fticr``: ms1 tolerance = 2 ppm, ms2 tolerance = 5 ppm


If you do need to use a different mass tolerance, you can set the ``ppm``, ``ms1_tol`` and ``ms2_tol`` parameters in the :class:`msbuddy.MsbuddyConfig` to the desired values.

.. code-block:: python

   msb_config = MsbuddyConfig(ppm=True, ms1_tol=10, ms2_tol=10)


The ``halogen`` parameter defaults to ``True``. Both halogen-free and
halogen-containing formulas are considered, so prior knowledge of whether the
unknown compounds contain F, Cl, Br or I is unnecessary. Only when prior
knowledge rules out halogens should you set ``halogen=False``:

.. code-block:: python

   msb_config = MsbuddyConfig(ms_instr='orbitrap', halogen=False)

For the command-line interface, use ``--no-halogen`` under the same condition.

The default fluorine range is ``(0, 80)``. This allows highly fluorinated
formulas without requiring a PFAS label. To set a different explicit limit,
use ``element_ranges={'F': (0, n)}`` or ``--f-max n``. Element-count likelihoods
in the compact model are soft ranking evidence; they do not remove formulas
through a binary halogen classifier. A formula outside a configured range
cannot be recovered by a ranker.

``element_ranges`` maps supported element symbols to inclusive ``(minimum,
maximum)`` pairs. Unspecified elements retain their defaults. Halogen ranges
are zero when ``halogen=False``; a nonzero explicit halogen range requires
``halogen=True``. K and Na have default ranges of ``(0, 0)``. Unsupported
symbols and invalid or noninteger ranges raise errors.
The effective ``config.element_ranges`` mapping is read-only; construct a new
configuration to change the limits.

.. note::
    To enable parallel processing, set ``parallel=True`` and specify the number of CPUs to use with ``n_cpu``. The code has to be run in ``if __name__ == '__main__':`` block.


For more information on the parameters, see :class:`msbuddy.MsbuddyConfig`.
