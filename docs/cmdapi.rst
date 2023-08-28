Command-line API
----------------

We also provide a command-line interface for **msbuddy**. To use it, you need to install the package first:

.. code-block:: bash

        pip install msbuddy

Then you can use the `msbuddy` command to perform the formula annotation job.
By default, the result summary will be written as a tsv file in the output folder.
Here is an example for loading a mgf file:

.. code-block:: bash

        msbuddy --mgf <mgf file> --output <output folder>


Alternatively, you can use the `--usi` option for a single USI string:

.. code-block:: bash

        msbuddy --usi <usi string> --output <output folder>

or the `--csv` option for a csv file containing a list of USI strings in the first column (one USI per line):


.. code-block:: bash

        msbuddy --csv <csv file> --output <output folder>

If you want detailed results for each query (all candidate formulas and their scores), you can use the `--details` option:

.. code-block:: bash

        msbuddy --mgf <mgf file> --output <output folder> --details


Please check out the `--help` option to see all the available options:

.. code-block:: bash

        msbuddy --help
