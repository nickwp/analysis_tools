'''## PID using the beam monitors
This code is the default way of identifying particles using the beam monitor. It is providing only a template event selection that is not optimised for any Physics analyses. It should serve as an exmple around which to build your own beam particle identification code. '''

#Step 0, import libraries
import numpy as np
import importlib
#this is the file with the necessary functions for performing PID 
import sys
#path to analysis tools - change with where your path is, though it might just work with mine on eos
sys.path.append("../") #neeed to acess the analysis_tools folded to acess
from analysis_tools import BeamAnalysis # as bm
# import cProfile

from analysis_tools import ReadBeamRunInfo 

import argparse



def parse_args():
    parser = argparse.ArgumentParser(
        description="Beam analysis configuration loader"
    )

    parser.add_argument(
        "--run",
        dest="run_number",
        type=int,
        required=True,
        help="Run number to analyse"
    )

    return parser.parse_args()


#Step 0: read  from the json file which run you want and its properties

args = parse_args()
target_run_number = args.run_number


run_info = ReadBeamRunInfo()

#The beam config holds information about the colimator slit status, in case it's needed
run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5, beam_config = run_info.get_info_run_number(target_run_number)
run_info.print_run_summary(there_is_ACT5)

#choose the number of events to read in, set to -1 if you want to read all events
n_events = -1

#output_filename
filename = f"../data/beam_data_analysis/beam_analysis_R{run_number}.root"


#Set up a beam analysis class 
ana = BeamAnalysis(run_number, run_momentum, n_eveto_group, n_tagger_group, there_is_ACT5)


# cProfile.run("ana.open_file(n_events, require_t5 = True)", sort="cumtime")

#Store into memory the number of events desired, 
# set require_t5 to False if you do not require that the particle reaches T5
ana.open_file(n_events, require_t5 = True)


#Step 2: Adjust the 1pe calibration: need to check the accuracy on the plots
# which are stored in plots/PID_run{run_number}_p{run_momentum}.pdf
ana.adjust_1pe_calibration()

#Step 3: proton and heavier particle tagging with T0-T1 TOF
#We need to tag protons before any other particles to avoid double-counting
ana.tag_protons_TOF()
#TODO: identify protons that produce knock-on electrons


#Step 4: tag electrons using ACT0-2 finding the minimum in the cut line
ana.tag_electrons_ACT02()

#Step 5: check visually that the electron and proton removal makes sense in ACT35
ana.plot_ACT35_left_vs_right()

#Step 6: make the muon/pion separation, using the muon tagger in case 
#at least 0.5% of muons and pions are above the cut line (at hiogh momentum). This is necessary in case the 
#Number of particles is too high to clearly see a minimum between the muons and pions
ana.tag_muons_pions_ACT35()

#This corrects any offset in the TOF (e.g. from cable length) that can cause the TOF 
#of electrons to be different from L/c This has to be calibrated to give meaningful momentum 
#estimates later on
ana.measure_particle_TOF() 

#This function extimates both the mean momentum for each particle type and for each trigger
#We take the the error on the tof for each trigger is the resolution of the TS0-TS1 measurement
#Taken as the std of the gaussian fit to the electron TOF
ana.estimate_particle_momentum()

#estimate the number of events per POT
ana.plot_number_particles_per_POT()

#Step X: end_analysis, necessary to cleanly close files 
ana.end_analysis()

#Output to a root file
ana.output_to_root(filename)

