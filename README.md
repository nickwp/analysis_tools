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

Production v_1 processes both self-trigger and hardware-trigger data through a multi-step pipeline.
Each pipeline is orchestrated by a runner script that calls the intermediate scripts in sequence.
Intermediate outputs are written to step-specific subdirectories under `<output_base>/<run_number>/`.

### Pipeline runner scripts

#### `run_self_trigger_pipeline.py`

Orchestrates self-trigger data processing through two steps:

```
Step 1 (calibrate) : calibrate_hits.py        → calibrated_hits/
Step 2 (dq)        : self_trigger_dq_flags.py → dq_flags/
```

Usage:
```bash
python run_self_trigger_pipeline.py \
  -i <input_file(s)> -r <run_number> -o <output_base_dir> [--debug]
```

#### `run_hw_trigger_pipeline.py`

Orchestrates hardware-trigger data processing through three steps:

```
Step 1 (wf)        : hw_trigger_wf_processing.py → processed_waveforms/
Step 2 (calibrate) : calibrate_hits.py            → calibrated_hits/
Step 3 (dq)        : hw_trigger_dq_flags.py       → dq_flags/
```

Usage:
```bash
python run_hw_trigger_pipeline.py \
  -i <input_file(s)> -r <run_number> -o <output_base_dir> [--debug]
```

Both pipeline scripts support `--from-step <step>` to skip completed earlier steps and reuse
their outputs, e.g. `--from-step dq` reprocesses only the DQ flags stage.

### Intermediate scripts

#### `hw_trigger_wf_processing.py`

Runs pulse finding and charge/time determination on hardware-trigger raw waveform data.
Reads `WCTEReadoutWindows` trees and writes a `ProcessedWaveforms` ROOT tree.

#### `calibrate_hits.py`

Applies timing constants from the calibration database to hit times.
Works on both self-trigger (`WCTEReadoutWindows`) and waveform-processed (`ProcessedWaveforms`) input files.
Writes a `CalibratedHits` ROOT tree and a `Configuration` tree recording the git hash,
timing constant revision, and list of PMTs with timing constants.

#### `self_trigger_dq_flags.py`

Applies data quality flags for self-trigger runs. Determines the good channel list from the
intersection of slow-control stable channels and channels with calibration constants.
Applies trigger-level bitmask flags for slow-control excluded periods and the 67 ms periodic issue.
Applies hit-level bitmask flags for missing timing constants and slow-control unstable channels.

#### `hw_trigger_dq_flags.py`

Applies data quality flags for hardware-trigger runs. In addition to the channel-level flags above,
applies trigger-level flags for missing waveforms, missing trigger signals, and mismatched waveform lengths.

### Output structure

```
<output_base>/<run_number>/
  processed_waveforms/   (hw-trigger only)
    <base>_processed_waveforms.root
  calibrated_hits/
    <base>_calibrated_hits.root
  dq_flags/
    <base>_[self|hw]_trigger_dq_flags.root
    <base>_[self|hw]_trigger_dq_flags_status.json
  merged/                (reserved for future merge step)
```

### Status sidecar files

Each DQ script writes a `_status.json` file alongside the output ROOT file containing QC metrics:

```json
{
  "status": "ok",
  "metrics": {
    "n_good_pmt_channels": 1562,
    "n_triggers": 50000,
    "n_bad_triggers": 724,
    "bad_trig_pct": 1.45,
    "n_hits": 3733472,
    "n_bad_hits": 5432,
    "bad_hit_pct": 0.15
  },
  "warnings": [],
  "errors": []
}
```

The pipeline runners read these sidecars after the DQ step and print a `QC WARNING` if configurable
thresholds (defined at the top of each pipeline runner) are exceeded. Files with warnings should
not be included in the merge step without manual review.

### Shared utilities

Common functions used across all production scripts are in `analysis_tools/production_utils.py`:
- `get_git_descriptor` — git provenance for output Configuration trees
- `file_sha256` — hash of the slow-control input file used
- `get_run_database_data`, `get_stable_mpmt_list_slow_control` — slow-control data access
- `get_slow_control_trigger_mask`, `get_67ms_mask` — trigger-level DQ masks
- `slot_pos_from_card_chan_list` — PMT channel mapping
- `write_status_json`, `read_status_json` — status sidecar I/O


# Beam monitor PID

This code performs the 1pe calibration of the ACT PMTs as well as the *basic* event PID based on monitor information (TOF, 
charge deposited in ACTs, etc...). 
The beam PID code is called by the notebooks/WCTE_beam_analysis.ipynb notebook, which in turn calls 
the BeamAnalysis class living in the notebooks/beam_monitors_pid.py python script.  

All the plots needed for visualising the selection are saved under  
notebooks/plots and any user of the code should refer to them for sanity checks, an example is provided. The
outputs of the selection are saved in a separate root file called beam_analysis_output_R{run_number}.root

The code also calculates the mean momentum for each particle type before exiting the CERN beam pipe (upstream of T0) 
and right after exiting the WCTE beam window (i.e. into the tank). These momenta are also estimated for each trigger,
based on the estimated PID. Note the error on these momenta (propagated from the time of flight resultion, taken as the
standard deviation of the TOF distribution for electrons) is very large for slow particles. Protons and deuterons events are 
identified using their time of flight but 3He nuclei aren't. The total charge in the TOF detector is saved in the output file but
no cut is placed on it. The "is_kept" branch of the output file stores information on whether the trigger passes the basic beam
requirements (no hits above threshold in the hole counters, hits in all T0, T1, T2 PMTs etc...)

The next version of this code will include additional tools for performing PID, helpful for analyses with tighter PID requirements
and some form of PID likelihood useful for estimating the confidence we have in each particle identification. 
