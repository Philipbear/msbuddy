Data Import
-----------

**msbuddy** provides functions to import data from `USI <https://doi.org/10.1101/2020.05.09.086066>`_ and mgf files. Custom data import can be achieved by creating :class:`MetaFeature` objects.


.. function:: load_usi (usi_list: Union[str, List[str]], adduct_list: Union[None, str, List[str]] = None)

    Read from a single USI string or a sequence of USI strings, and return a list of :class:`MetaFeature` objects.

    :param usi_list: str or List[str]. A single USI string or a sequence of USI strings.
    :param optional adduct_list: str or List[str]. A single adduct string or a sequence of adduct strings, which will be applied to all USI strings accordingly.
    :returns: A list of :class:`MetaFeature` objects.

Example Usage:

.. code-block:: python

   from msbuddy import Buddy, load_usi

   # instantiate a Buddy object
   buddy = Buddy()

   # load a single USI
   buddy.load_usi('mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036')

   # load a list of USIs
   buddy.load_usi(['mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036',
                   'mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000845027'])

   # load USIs with adducts
   buddy.load_usi(usi_list=['mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036',
                            'mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000845027'],
                  adduct_list=['[M+H]+', '[M-H2O+H]+'])


.. function:: load_mgf (mgf_file: str)

   Read a single mgf file, and return a list of :class:`MetaFeature` objects.

   :param mgf_file: str. The path to the mgf file.
   :returns: A list of :class:`MetaFeature` objects.

Example Usage:

.. code-block:: python

   from msbuddy import Buddy, load_mgf

   # instantiate a Buddy object
   buddy = Buddy()

   # load a single mgf file
   buddy.load_mgf('input_file.mgf')



Custom Data Import
==================

Users can also import data from custom data sources using the :class:`MetaFeature` class and the :class:`Spectrum` class.

We first show an easy example of importing data from a pandas DataFrame (with columns 'mz', 'intensity') containing an MS/MS spectrum .

.. code-block:: python

   from msbuddy import Buddy, MetaFeature, Spectrum
   import pandas as pd
   import numpy as np

   # instantiate a Buddy object
   buddy = Buddy()

   # read an MS/MS spectrum from a pandas DataFrame, with columns 'mz', 'intensity'
   ms2_df = pd.read_csv('input_file.csv')

   # create a Spectrum object
   ms2_spec = Spectrum(mz_array = np.array(ms2_df['mz']),
                       int_array = np.array(ms2_df['intensity']))

   # create a MetaFeature object
   metafeature = MetaFeature(mz = 123.4567,
                             charge = 1,
                             ms2 = ms2_spec)


A more complicated example with MS1 and MS/MS spectra is shown below.

.. code-block:: python

   from msbuddy import Buddy, MetaFeature, Spectrum
   import pandas as pd
   import numpy as np

   # instantiate a Buddy object
   buddy = Buddy()

   # read an MS1 spectrum from a pandas DataFrame, with columns 'mz', 'intensity'
   ms1_df = pd.read_csv('input_file.csv')

   # create a Spectrum object
   ms1_spec = Spectrum(mz_array = np.array(ms1_df['mz']),
                       int_array = np.array(ms1_df['intensity']))

   # read an MS/MS spectrum from a pandas DataFrame, with columns 'mz', 'intensity'
   ms2_df = pd.read_csv('input_file.csv')

   # create a Spectrum object
   ms2_spec = Spectrum(mz_array = np.array(ms2_df['mz']),
                       int_array = np.array(ms2_df['intensity']))

   # create a MetaFeature object
   metafeature = MetaFeature(mz = 123.4567,
                             charge = 1,
                             ms1 = ms1_spec,
                             ms2 = ms2_spec)
