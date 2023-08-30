Parameter Settings
------------------

All **msbuddy** parameters are stored in a single class called :class:`msbuddy.BuddyParamSet`.

Here is a quick example of how to create a :class:`msbuddy.BuddyParamSet` object and pass it to a :class:`msbuddy.Buddy` object.
Of note, parallel processing is available for **msbuddy**. To use it, set ``parallel=True`` and specify the number of CPUs to use with ``n_cpu``.

.. code-block:: python

    from msbuddy import Buddy, BuddyParamSet

    # create a parameter set
    buddy_param_set = BuddyParamSet(
        ppm=True,
        ms1_tol=10,
        ms2_tol=20,
        halogen=True,
        parallel=True,
        n_cpu=4,
        timeout_secs=600)

    # create a Buddy object with the specified parameter set
    buddy = Buddy(buddy_param_set)


For more information on the parameters, see :class:`msbuddy.BuddyParamSet`.
