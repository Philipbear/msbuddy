Command-line API
----------------

We also provide a command-line interface for **msbuddy**. To use it, you need to install the package first:

.. code-block:: bash

        pip install msbuddy

Then you can use the `msbuddy` command to perform the formula annotation job.
By default, the result summary will be written as a tsv file in the output folder (``-output`` or ``-o`` for short).
Here is an example for loading a mgf file from orbitrap instruments (``-ms``) and performing formula annotation in parallel (``-parallel`` or ``-p`` for short) using 12 processes:

.. code-block:: bash

        msbuddy -mgf <mgf_file> -output <output_folder> -ms orbitrap -p -n_cpu 12


Alternatively, you can use the ``-usi`` option for a single USI string (from QTOF instruments, for example):

.. code-block:: bash

        msbuddy -usi <usi_string> -output <output_folder> -ms qtof

or the ``-csv`` option for a csv file containing a list of USI strings in the first column (one USI per line):


.. code-block:: bash

        msbuddy -csv <csv_file> -output <output_folder> -ms fticr -p -n_cpu 12

If you want detailed results for each query (all candidate formulas and their scores), you can use the ``-details`` option (or ``-d`` for short).

Here is an example of processing a mgf file from Orbitrap (``-ms orbitrap``) in parallel (``-p``) using 12 cores (``-n_cpu``).
Detailed results are written (``-d``).
Halogen atoms (FClBrI) are also considered (``-hal``).

.. code-block:: bash

        msbuddy -mgf <mgf_file> -output <output_folder> -ms orbitrap -p -n_cpu 12 -d -hal

For mgf file or csv file, if you want to use the default output folder (``./msbuddy_output``), you can omit the ``-output`` option.
Results will be written in the same folder as the input file.
For USI string, the output folder is required.


**In all cases, ``-ms`` option is strongly recommended. (qtof, orbitrap or fticr)**


Please check out the ``--help`` (or ``-h``) option to see all the available options:

.. code-block:: bash

        msbuddy --help
