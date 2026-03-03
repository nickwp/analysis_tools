# Installation
Installation:

git clone <git repo location>
cd analysis_tools
pip install -e .

The -e flag allows you to edit the package 
If using on lxplus you will need to setup this in a python virtual environment 

## Contribution Rules
Main branch on WCTE/analysis_tools is protected - please open a pull request (either from your own branch or fork) to push changes to main branch

# Package classes and functions

##  WaveformProcessing

Waveform processing contains a copy of the CFD used in the test stand repository
WaveformProcessingTeststand.cfd_teststand_method() processes the CFD using that method returning 
the charge and the time for a pulse in that waveform - including non-linearity corrections 
for both

Additionally the same CFD and charge calculation method used online by the mPMT is included 
in WaveformProcessingmPMT the versions to run on single waveforms and vectorised versions to run on arrays of waveforms are given 

## do_pulse_finding and do_pulse_finding_vect

Finds pulses in the waveforms using the same method as run online on the mPMT. do_pulse_finding_vect is
a vectorised version

## CalibrationDBInterface

Interfaces with the calibration database see more instruction here
https://wcte.hyperk.ca/documents/calibration-db-apis/v1-api-endpoints-documentation
Currently processed for the test database - to be updated when the production database 
is ready. The authentication requires a credential text file ./.wctecaldb.credential 
to be in the current working directory - more details in the database interface above

## PMTMapping 

PMTMapping is a class containing the mapping of the WCTE PMTs slot and position ids to the
electronics channel and mPMT card ids and vice versa
Usage:
mapping = PMTMapping()
mapping.get_slot_pmt_pos_from_card_pmt_chan(card_id,pmt_channel) returns the slot and pmt position
and 
mapping.get_card_pmt_chan_from_slot_pmt_pos(slot_id,pmt_position) returns the card and channel
The mapping json is located in the package

## DetectorGeometry

Class to load PMT positions, directions and calculate time of flight.

# Processing Scripts

## production v_0
### add_timing_constants.py

The earliest version of production script for production v_0 which added timing constants to self-trigger data

## production v_0_5
### process_data_v0_5.py

Self-trigger data production v 0_5 (see https://wcte.hyperk.ca/wg/simulation-and-analysis/data-production-2) 
for self-trigger data which includes timing constants and data quality flags

## production v_1
### hardware_trigger_wf_processing_v1.py

Does pulse finding and charge/time determination for hardware trigger data

### calibrate_hits.py

Adds timing constants to either self-trigger data or hits found during hardware trigger processing

### self_trigger_dq_flags.py

Adds data quality flags for self trigger runs, determining the good channel list from slow control 
and channels with calibration constants. Adds trigger level flags for bad triggers from slow control
and triggers effected by 67ms

# Beam monitor PID

This code performs the 1pe calibration of the ACT PMTs as well as the *basic* event PID based on monitor information (TOF, 
charge deposited in ACTs, etc...). 
The beam PID code is called by the script/WCTE_beam_analysis.py code, which in turn calls 
the BeamAnalysis class living in the analysis_tools/beam_monitors_pid.py python script. To call the script on the lxplus machines, run:
'''python3  script/WCTE_beam_analysis.py --run <run_number>'''
the code looks up the run information from the .json file of run information compiled by Laurence from the Google sheet. 


All of the information about this version of the code (made for the winter 2025-2026 data production is available here: 
https://www.overleaf.com/read/fshzdwxbjwnn#c0127f
please read ! Email Alie or Bruno if you have any questions. 

All the plots needed for visualising the selection are saved under  
notebooks/plots and any user of the code should refer to them for sanity checks, an example is provided. The
outputs of the selection are saved in a separate root file called beam_analysis_output_R{run_number}.root

The code also calculates the mean momentum for each particle type before exiting the CERN beam pipe (upstream of T0) 
and right after exiting the WCTE beam window (i.e. into the tank). These momenta are also estimated for each trigger,
based on the estimated PID (shown on plots but not shared due to the large error and general unreliability). 
Note the error on these momenta (propagated from the time of flight resultion, taken as the
standard deviation of the TOF distribution for electrons) is very large for slow particles. Protons and deuterons events are 
identified using their time of flight but 3He nuclei aren't. The total charge in the TOF detector is saved in the output file but
no cut is placed on it. The "is_kept" branch of the output file stores information on whether the trigger passes the basic beam
requirements (no hits above threshold in the hole counters, hits in all T0, T1, T2 PMTs etc...)

The next version of this code will include additional tools for performing PID, helpful for analyses with tighter PID requirements
and some form of PID likelihood useful for estimating the confidence we have in each particle identification. 
