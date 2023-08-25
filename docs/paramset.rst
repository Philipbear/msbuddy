Parameter Settings
------------------

All **msbuddy** parameters are stored in a single class called :class:`BuddyParamSet`.

Here is a quick example of how to create a :class:`BuddyParamSet` object and pass it to a :class:`Buddy` object.

.. code-block:: python

    from msbuddy import Buddy, BuddyParamSet

    # create a parameter set
    buddy_param_set = BuddyParamSet(
        ppm=True,
        ms1_tol=10,
        ms2_tol=20,
        halogen=True,
        timeout_secs=600)

    # create a Buddy object with the specified parameter set
    buddy = Buddy(buddy_param_set)



