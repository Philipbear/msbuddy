Data Import
-----------

**msbuddy** provides functions to import data from `USI <https://doi.org/10.1101/2020.05.09.086066>`_ (**load_usi**) and mgf files (**load_mgf**). Custom data import can be achieved by creating :class:`msbuddy.base.MetaFeature` objects.

Built-in Data Import
====================
.. function:: load_usi (usi_list: Union[str, List[str]], adduct_list: Union[None, str, List[str]] = None)

    Read from a single USI string or a sequence of USI strings, and load the data into the ``data`` attribute of the :class:`msbuddy.Msbuddy` object. The USI string will be used as the unique identifier in `msbuddy`.

    :param usi_list: str or List[str]. A single USI string or a sequence of USI strings.
    :param optional adduct_list: str or List[str]. A single adduct string or a sequence of adduct strings, which will be applied to all USI strings accordingly.
    :returns: None. A list of :class:`msbuddy.base.MetaFeature` objects will be stored in the ``data`` attribute of the :class:`msbuddy.Msbuddy` object.

Example Usage:

.. code-block:: python

   from msbuddy import Msbuddy

   # instantiate a Msbuddy object
   engine = Msbuddy()

   # load a single USI
   engine.load_usi('mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036')

   # load a list of USIs
   engine.load_usi(['mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036',
                    'mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000845027'])

   # load USIs with adducts
   engine.load_usi(usi_list=['mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036',
                             'mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000845027'],
                   adduct_list=['[M+H]+', '[M-H2O+H]+'])


.. function:: load_mgf (mgf_file: str)

   Read a single mgf file, and load the data into the ``data`` attribute of the :class:`msbuddy.Msbuddy` object. In mgf files, the field ``TITLE``, ``SPECTRUMID`` or ``SPECTRUM_ID`` will be used as the unique identifier in `msbuddy`. Please include one of these fields in the mgf file. See `demo mgf file <https://github.com/Philipbear/msbuddy/tree/main/demo>`_

   :param mgf_file: str. The path to the mgf file.
   :returns: None. A list of :class:`msbuddy.base.MetaFeature` objects will be stored in the ``data`` attribute of the :class:`msbuddy.Msbuddy` object.

Example Usage:

.. code-block:: python

   from msbuddy import Msbuddy

   # instantiate a Msbuddy object
   engine = Msbuddy()

   # load a single mgf file
   engine.load_mgf('input_file.mgf')



Custom Data Import
==================

Users can also import data from custom data sources using the :class:`msbuddy.base.MetaFeature` class and the :class:`msbuddy.base.Spectrum` class.

We first show an example of importing an MS/MS spectrum only.

.. code-block:: python

   from msbuddy import Msbuddy
   from msbuddy.base import MetaFeature, Spectrum
   import numpy as np

   # instantiate a Msbuddy object
   engine = Msbuddy()

   # create a Spectrum object
   ms2_spec = Spectrum(mz_array = np.array([100, 200, 300]),
                       int_array = np.array([1, 2, 3]))

   # create a MetaFeature object
   metafeature = MetaFeature(identifier = 0,  # unique identifier for the MetaFeature object
                             mz = 400.00,  # precursor m/z
                             rt = None,  # retention time, can be None if not available
                             charge = 1,  # precursor charge
                             ms2 = ms2_spec)

   # add to the Msbuddy object, List[MetaFeature] is accepted
   engine.add_data([metafeature])


Note that for :class:`msbuddy.base.MetaFeature` class, the ``identifier``, ``mz`` and ``charge`` attributes are required; attributes ``rt``, ``ms1`` and ``ms2`` are optional. If they are not provided, ``None`` will be assigned.

A more complicated example with both MS1 isotope pattern and MS/MS is shown below. You can also specify the adduct type.

.. code-block:: python

   from msbuddy import Msbuddy
   from msbuddy.base import MetaFeature, Spectrum
   import numpy as np

   # instantiate a Msbuddy object
   engine = Msbuddy()

   # create a Spectrum object for MS1 isotope pattern
   ms1_spec = Spectrum(mz_array = np.array([400, 401, 402]),
                       int_array = np.array([100, 20, 5]))

   # create a Spectrum object for MS/MS
   ms2_spec = Spectrum(mz_array = np.array([100, 200, 300]),
                       int_array = np.array([1, 2, 3]))

   # create a MetaFeature object
   metafeature = MetaFeature(identifier = 0,
                             mz = 400.00,
                             charge = 1,
                             adduct = '[M+H]+',
                             ms1 = ms1_spec,
                             ms2 = ms2_spec)

   # add to the Msbuddy object, List[MetaFeature] is accepted
   engine.add_data([metafeature])
