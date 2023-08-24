Data Import
-------------

**msbuddy** provides functions to import data from `USI <https://doi.org/10.1101/2020.05.09.086066>`_ and mgf files.


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
   buddy.load_usi('mzspec:MSV000084615:QC1_1:scan:1')

   # load a list of USIs
   buddy.load_usi(['mzspec:MSV000084615:QC1_1:scan:1', 'mzspec:MSV000084615:QC1_1:scan:2'])

   # load USIs with adducts
   buddy.load_usi(['mzspec:MSV000084615:QC1_1:scan:1', 'mzspec:MSV000084615:QC1_1:scan:2'],
                  ['[M+H]+', '[M+Na]+'])

