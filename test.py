from msbuddy import Msbuddy, MsbuddyConfig

# instantiate a MsbuddyConfig object
msb_config = MsbuddyConfig(# highly recommended to specify
                           ms_instr='orbitrap',  # supported: "qtof", "orbitrap" and "fticr"
                           # whether to consider halogen atoms FClBrI
                           halogen=False)

# instantiate a Msbuddy object
msb_engine = Msbuddy(msb_config)

# you can load multiple USIs at once
msb_engine.load_usi(['mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036',
                     'mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740037'])

# annotate molecular formula
msb_engine.annotate_formula()

# retrieve the annotation result summary
result = msb_engine.get_summary()


# test other msbuddy APIs



#
# # test parallel computing
# if __name__ == '__main__':
#     # instantiate a MsbuddyConfig object
#     msb_config = MsbuddyConfig(ms_instr='orbitrap', # supported: "qtof", "orbitrap" and "fticr"
#                                                     # highly recommended to specify
#                                halogen=False, # whether to consider halogen atoms FClBrI
#                                parallel=True, n_cpu=4)
#
#     # instantiate a Msbuddy object
#     msb_engine = Msbuddy(msb_config)
#
#     # load data, here we use a mgf file as an example
#     msb_engine.load_usi(['mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740036',
#                          'mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00003740037'])
#
#     # annotate molecular formula
#     msb_engine.annotate_formula()
#
#     # retrieve the annotation result summary
#     result = msb_engine.get_summary()
