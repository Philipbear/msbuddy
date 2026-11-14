Command-line API
----------------

We also provide a command-line interface for **msbuddy**. To use it, you need to install the package first:

.. code-block:: bash

        pip install msbuddy

Then you can use the ``msbuddy`` command to perform the formula annotation job.
By default, the result summary will be written as a tsv file in the output folder (``-output`` or ``-o`` for short).
Here is an example for loading a mgf file from orbitrap instruments (``-ms``) and performing formula annotation in parallel (``-parallel`` or ``-p`` for short) using 12 processes:

.. code-block:: bash

        msbuddy -mgf <mgf_file> -output <output_folder> -ms orbitrap -p -n_cpu 12


Alternatively, you can use the ``-usi`` option for a single USI string (from QTOF instruments, for example):

.. code-block:: bash

        msbuddy -usi <usi_string> -output <output_folder> -ms qtof

or the ``-csv`` option for a csv file containing a list of USI strings:

(one USI per line, USIs in the first column. An optional ``usi`` header and an adduct column are supported. `Template files <https://github.com/Philipbear/msbuddy/tree/main/demo>`_)

.. code-block:: bash

        msbuddy -csv <csv_file> -output <output_folder> -ms fticr -p -n_cpu 12

If you want detailed results for each query (all candidate formulas and their scores), you can use the ``-details`` option (or ``-d`` for short).

Here is an example of processing a mgf file from Orbitrap (``-ms orbitrap``) in parallel (``-p``) using 12 cores (``-n_cpu``).
Detailed results are written (``-d``).
Halogen atoms (F, Cl, Br and I) are considered by default.

.. code-block:: bash

        msbuddy -mgf <mgf_file> -output <output_folder> -ms orbitrap -p -n_cpu 12 -d

Both halogen-free and halogen-containing formulas remain candidates. Only when
prior knowledge rules out halogens should you add ``--no-halogen``:

.. code-block:: bash

        msbuddy --mgf <mgf_file> --ms-instr orbitrap --no-halogen

The existing ``--halogen``, ``-halogen`` and ``-hal`` flags remain supported and
explicitly enable the default behavior. They cannot be combined with
``--no-halogen``. The Nextflow wrapper also includes halogens by default;
its numeric ``-hal 0`` option excludes them when prior knowledge rules them out.

For mgf file or csv file, if you want to use the default output folder (``./msbuddy_output``), you can omit the ``-output`` option.
Results will be written in the same folder as the input file.
For USI string, the output folder is required.


.. note::
    In all cases, ``-ms`` option is strongly recommended ('qtof', 'orbitrap' or 'fticr').


Option naming
~~~~~~~~~~~~~

Every option also accepts a conventional double-dash form, so ``--mgf`` and ``-mgf`` are equivalent, as are ``--n-cpu`` and ``-n_cpu``.

Descriptive names use hyphens in the double-dash form (``--ms-instr``, ``--batch-size``, ``--top-n-per-50-da``), while genuine one-character shortcuts stay single-dash (``-o``, ``-d``, ``-p``, ``-t``).

Every spelling shown above continues to work unchanged; the double-dash forms are additions, not replacements.


Please check out the ``--help`` (or ``-h``) option to see all the available options:

.. code-block:: bash

        msbuddy --help
