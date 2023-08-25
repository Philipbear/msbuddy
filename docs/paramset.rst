Parameter Settings
------------------

All **msbuddy** parameters are stored in a single class called :class:`BuddyParamSet`.

.. class:: BuddyParamSet (ppm: bool = True, ms1_tol: float = 5, ms2_tol: float = 10, halogen: bool = False, timeout_secs: float = 300, c_range: Tuple[int, int] = (0, 80), h_range: Tuple[int, int] = (0, 150), n_range: Tuple[int, int] = (0, 20), o_range: Tuple[int, int] = (0, 30), p_range: Tuple[int, int] = (0, 10), s_range: Tuple[int, int] = (0, 15), f_range: Tuple[int, int] = (0, 20), cl_range: Tuple[int, int] = (0, 15), br_range: Tuple[int, int] = (0, 10), i_range: Tuple[int, int] = (0, 10), isotope_bin_mztol: float = 0.02, max_isotope_cnt: int = 4, ms2_denoise: bool = True, rel_int_denoise: bool = True, rel_int_denoise_cutoff: float = 0.01, max_noise_frag_ratio: float = 0.85, max_noise_rsd: float = 0.20, max_frag_reserved: int = 50, use_all_frag: bool = False)

   It is a class to store all the parameter settings for **msbuddy**.

   :param ppm: bool. If True, the mass tolerance is in ppm. If False, the mass tolerance is in Da. Default is True.
   :param ms1_tol: float. The mass tolerance for MS1 spectra. Default is 5 ppm.
   :param ms2_tol: float. The mass tolerance for MS/MS spectra. Default is 10 ppm.
   :param halogen: bool. If True, the halogen elements (F, Cl, Br, I) are considered. Default is False.
   :param timeout_secs: float. The timeout in seconds for each query. Default is 300 seconds.
   :param c_range: Tuple[int, int]. The range of carbon atoms. Default is (0, 80).
   :param h_range: Tuple[int, int]. The range of hydrogen atoms. Default is (0, 150).
   :param n_range: Tuple[int, int]. The range of nitrogen atoms. Default is (0, 20).
   :param o_range: Tuple[int, int]. The range of oxygen atoms. Default is (0, 30).
   :param p_range: Tuple[int, int]. The range of phosphorus atoms. Default is (0, 10).
   :param s_range: Tuple[int, int]. The range of sulfur atoms. Default is (0, 15).
   :param f_range: Tuple[int, int]. The range of fluorine atoms. Default is (0, 20).
   :param cl_range: Tuple[int, int]. The range of chlorine atoms. Default is (0, 15).
   :param br_range: Tuple[int, int]. The range of bromine atoms. Default is (0, 10).
   :param i_range: Tuple[int, int]. The range of iodine atoms. Default is (0, 10).
   :param isotope_bin_mztol: float. The mass tolerance for MS1 isotope binning, in Da. Default is 0.02 Da.
   :param max_isotope_cnt: int. The maximum number of isotopes to consider. Default is 4.
   :param ms2_denoise: bool. If True, the MS/MS spectra are denoised (see details in `our paper <https://doi.org/10.1038/s41592-023-01850-x>`_). Default is True.
   :param rel_int_denoise: bool. If True, the MS/MS spectra are denoised based on relative intensity. Default is True.
   :param rel_int_denoise_cutoff: float. The cutoff for relative intensity denoising. Default is 0.01 (1%).
   :param max_noise_frag_ratio: float. The maximum ratio of noise fragments to total fragments. Default is 0.85 (85%).
   :param max_noise_rsd: float. The maximum relative standard deviation of noise fragments. Default is 0.20 (20%).
   :param max_frag_reserved: int. The maximum number of fragments to reserve. Default is 50.
   :param use_all_frag: bool. If True, all fragments are used. If False, only the top fragments are used. Default is False.


Example Usage:

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

    # continue as demonstrated in Quick Start

