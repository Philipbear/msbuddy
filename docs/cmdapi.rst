Command-line API
----------------

We also provide a command-line interface for **msbuddy**. To use it, you need to install the package first:

.. code-block:: bash

        pip install msbuddy

Then you can use the `msbuddy` command to perform formula annotation on a given mgf file:

.. code-block:: bash

        msbuddy --mgf <mgf file> --output <output folder>


Alternatively, you can use the `--usi` option for a single USI string:

.. code-block:: bash

        msbuddy --usi <usi string> --output <output folder>

or the `--csv` option for a csv file containing a list of USI strings in the first column (one USI per line):


.. code-block:: bash

        msbuddy --csv <csv file> --output <output folder>



You can also use the `--help` option to see all the available options:

.. code-block:: bash

        msbuddy --help
