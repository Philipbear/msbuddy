Installation
------------

**msbuddy** requires Python version 3.9+. There are a variety of ways to install **msbuddy**.

Option 1: Using pip (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**msbuddy** is available on PyPI, you can install it via ``pip``:

.. code-block:: bash

   pip install msbuddy

To perform formula annotation, you also have to install the `lightgbm <https://github.com/microsoft/LightGBM/tree/master/python-package>`_ package.

.. code-block:: bash

   pip install lightgbm


Option 2: From source
~~~~~~~~~~~~~~~~~~~~~
You can also install **msbuddy** from source:

.. code-block:: bash

   git clone https://github.com/Philipbear/msbuddy.git

Option 3: From PyPI release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Alternatively, you can download the latest release from `PyPI <https://pypi.org/project/msbuddy/#history>`_ and install it:

.. code-block:: bash

   tar -zxvf msbuddy-0.3.0.tar.gz


Data files
~~~~~~~~~~
For backwards compatibility, **msbuddy** still uses the ``msbuddy/data`` directory beside the installed package when no data options are supplied.

For a system installation, a writable directory can instead be selected in the existing configuration object:

.. code-block:: python

   from msbuddy import Msbuddy, MsbuddyConfig

   config = MsbuddyConfig(data_dir="~/msbuddy-data")
   engine = Msbuddy(config)

Compatible files can also use different names. Relative names are resolved inside ``data_dir`` and absolute paths are used directly:

.. code-block:: python

   config = MsbuddyConfig(
       data_dir="~/msbuddy-data",
       data_files={
           "formula_db": "formula-custom.joblib",
           "ml_model": "/shared/msbuddy/model-custom.joblib",
       },
   )

The supported keys are ``common_db``, ``formula_db``, and ``ml_model``. A partial mapping is allowed; unspecified files keep their standard names.

A custom file that already exists is loaded as-is. A missing custom file raises a clear error instead of silently substituting another file. To place the standard release asset under a custom name, call ``download_data`` explicitly.

The file paths and standard download links can be inspected, or all files can be downloaded in advance:

.. code-block:: python

   from msbuddy import download_data, get_data_files

   print(get_data_files(config.data_dir, config.data_files))
   download_data(config.data_dir, config.data_files)

The existing command-line interface provides matching options. Examples use the conventional double-dash form for descriptive option names.

Existing single-dash spellings such as ``-mgf`` remain available as aliases for backwards compatibility.

.. code-block:: bash

   msbuddy --download-data --data-dir ~/msbuddy-data
   msbuddy --mgf input.mgf --data-dir ~/msbuddy-data \
       --formula-db formula-custom.joblib \
       --ml-model /shared/msbuddy/model-custom.joblib

Omitting all new options preserves the original location and automatic download behavior. Permission errors name the failed path and show the writable-directory alternatives.
