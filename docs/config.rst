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
                             # supported MS types: "qtof", "orbitrap" and "fticr"
        ppm=True, # use ppm for mass tolerance, otherwise use Da
        ms1_tol=5, # ms1 tolerance, see below for default values for each MS instrument
        ms2_tol=10, # ms2 tolerance, see below for default values for each MS instrument
        halogen=True, # enable halogen atoms in molecular formula annotation
        # parallel=True, # enable parallel processing, see note below
        # n_cpu=12, # number of CPUs to use
        timeout_secs=600, # timeout for each query in seconds
        batch_size=1000, # number of queries to process in each batch, a larger batch size will use more memory but will be faster
        c_range=(0, 100), # range of carbon numbers to consider
        h_range=(0, 150), # range of hydrogen numbers to consider
        ... # other parameters
        )

    # create a Msbuddy object with the specified configuration
    msb_engine = Msbuddy(msb_config)


It is **highly recommended** to set up ``ms_instr`` in the :class:`msbuddy.MsbuddyConfig` to obtain the best annotation performance. The supported MS types are ``"qtof"``, ``"orbitrap"`` and ``"fticr"``.

The following mass tolerances will be used for each MS instrument:

``qtof``: ms1 tolerance = 10 ppm, ms2 tolerance = 20 ppm

``orbitrap``: ms1 tolerance = 5 ppm, ms2 tolerance = 10 ppm

``fticr``: ms1 tolerance = 2 ppm, ms2 tolerance = 5 ppm


If you do need to use a different mass tolerance, you can set the ``ppm``, ``ms1_tol`` and ``ms2_tol`` parameters in the :class:`msbuddy.MsbuddyConfig` to the desired values.

.. code-block:: python

   msb_config = MsbuddyConfig(ppm=True, ms1_tol=10, ms2_tol=10)


Note that the ``halogen`` parameter is set to ``False`` by default. If you are working with halogenated compounds, you will need to set this to ``True`` to enable halgoen atoms in molecular formula annotation.

.. note::
    To enable parallel processing, set ``parallel=True`` and specify the number of CPUs to use with ``n_cpu``. The code has to be run in ``if __name__ == '__main__':`` block.


For more information on the parameters, see :class:`msbuddy.MsbuddyConfig`.
