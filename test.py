# test msbuddy
# ver 0.3.0

def test_main():
    from msbuddy import Msbuddy, MsbuddyConfig
    # instantiate a MsbuddyConfig object
    msb_config = MsbuddyConfig(ms_instr='orbitrap', halogen=False)

    # instantiate a Msbuddy object
    msb_engine = Msbuddy(msb_config)

    # you can load multiple USIs at once
    msb_engine.load_mgf('/Users/shipei/Documents/projects/msbuddy/demo/input_file.mgf')

    # annotate molecular formula
    msb_engine.annotate_formula()

    # retrieve the annotation result summary
    result = msb_engine.get_summary()

    print(result)


# test other msbuddy APIs
def test_formula():
    from msbuddy import read_formula

    formula_array = read_formula("C10H20O5")
    print(formula_array)

    from msbuddy import form_arr_to_str

    formula_str = form_arr_to_str([10, 20, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0])
    print(formula_str)

    from msbuddy import enumerate_subform_arr

    all_subform_arr = enumerate_subform_arr([10, 20, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0])
    print(all_subform_arr)


def test_mass_formula():
    from msbuddy import Msbuddy

    # create a Msbuddy object
    engine = Msbuddy()

    # convert mass to formula
    formula_list = engine.mass_to_formula(300, 10, True)

    # print results
    for f in formula_list:
        print(f.formula, f.mass_error, f.mass_error_ppm)

    # convert mz to formula
    formula_list = engine.mz_to_formula(300, "[M+H]+", 10, True)

    # print results
    for f in formula_list:
        print(f.formula, f.mass_error, f.mass_error_ppm)


def test_subformula():
    from msbuddy import assign_subformula

    subformla_list = assign_subformula([107.05, 149.02, 209.04, 221.04, 230.96],
                                       precursor_formula="C15H16O5", adduct="[M+H]+",
                                       ms2_tol=0.02, ppm=False, dbe_cutoff=-1.0)
    print(len(subformla_list))


if __name__ == '__main__':
    test_main()
    test_formula()
    test_mass_formula()
    test_subformula()
