msbuddy
=======

.. image:: https://img.shields.io/badge/maintainer-Shipei_Xing-blue
   :alt: Maintainer
.. image:: https://img.shields.io/pypi/v/msbuddy?color=green
   :alt: PyPI
   :target: https://pypi.org/project/msbuddy/
.. image:: https://readthedocs.org/projects/msbuddy/badge/?version=latest
   :alt: docs
   :target: https://msbuddy.readthedocs.io/en/latest/?badge=latest

`msbuddy <https://github.com/Philipbear/msbuddy>`_ is developed for molecular formula annotation in mass spectrometry-based small molecule analysis.

**msbuddy** is able to provide formula annotations for:

  * queries with or without MS1 isotopic pattern
  * queries with or without MS/MS spectra
  * both positive and negative ionization modes

The minimum requirement for a **msbuddy** query is a single m/z value and its ionization polarity.

**msbuddy** enables candidate space shrinkage via bottom-up MS/MS interrogation, accurate formula annotation via machine learning and false discovery rate estimation.
Please refer to `our paper <https://doi.org/10.1038/s41592-023-01850-x>`_ for more details.

Source codes are available at `GitHub <https://github.com/Philipbear/msbuddy>`_.

.. toctree::
   :caption: Contents
   :maxdepth: 1

   install
   quickstart
   paramset
   datain
   dataout
   pyapi
   cmdapi
   contact

Citation
--------

When using **msbuddy**, please cite:

  * Xing, S., Shen, S., Xu, B. et al. `BUDDY: molecular formula discovery via bottom-up MS/MS interrogation <https://doi.org/10.1038/s41592-023-01850-x>`_. *Nature Methods* **20**, 881–890 (2023).

Contact
-------

Shipei Xing: s1xing@health.ucsd.edu  /  philipxsp@hotmail.com

License
-------

**msbuddy** is licensed under the terms of the Apache License 2.0.
