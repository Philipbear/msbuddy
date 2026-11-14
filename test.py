# Example msbuddy workflows.

def test_main():
    from msbuddy import Msbuddy, MsbuddyConfig
    # instantiate a MsbuddyConfig object
    msb_config = MsbuddyConfig(ms_instr='orbitrap')

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
    from itertools import islice
    from msbuddy import enumerate_subformulas, formula_to_string, read_formula

    composition = read_formula("C10H20O5")
    print(composition)

    formula_str = formula_to_string(composition)
    print(formula_str)

    print(list(islice(enumerate_subformulas(composition), 5)))


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
