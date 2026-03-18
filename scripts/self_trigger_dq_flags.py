## This script creates a new root file with data quality flags on a trigger by trigger basis
import numpy as np
import uproot
import awkward as ak
import argparse
import os
from data_quality_flags import HitMask, TriggerMask
from analysis_tools.production_utils import (
    get_git_descriptor,
    file_sha256,
    get_run_database_data,
    get_stable_mpmt_list_slow_control,
    get_slow_control_trigger_mask,
    get_67ms_mask,
    slot_pos_from_card_chan_list,
    write_status_json,
)
from analysis_tools import CalibrationDBInterface

SLOW_CONTROL_GOOD_RUN_LIST_PATH = '/eos/experiment/wcte/configuration/slow_control_summary/all_run_list_v7.json'



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Apply data quality flags to self-trigger data.")
    parser.add_argument("-i", "--input_files", required=True, nargs='+', help="Path to WCTEReadoutWindows ROOT file(s)")
    parser.add_argument("-c", "--input_calibrated_file_directory", required=True, help="Path to calibrated hits directory")
    parser.add_argument("-r", "--run_number", required=True, help="Run number")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to write output file")
    parser.add_argument("--debug", action="store_true", help="Enable debug - disables checks allowing for test runs")
    args = parser.parse_args()
    
    git_hash = get_git_descriptor(debug=args.debug)
    
    #check that the run number is correct 
    for input_file in args.input_files:
        if f"R{args.run_number}" not in input_file:
            raise Exception(f"Input file {input_file} does not match run number {args.run_number}")
    
    #make a list of calibrated input files
    calibrated_input_files = []

    #counters for statistics
    run_total_triggers = 0
    run_total_bad_triggers = 0
    run_total_hits = 0
    run_total_bad_hits = 0

    for input_file in args.input_files:
        base = os.path.splitext(os.path.basename(input_file))[0]
        calibrated_file_name = f"{base}_calibrated_hits.root" 
        calibrated_input_file_path = os.path.join(args.input_calibrated_file_directory, calibrated_file_name)
        calibrated_input_files.append(calibrated_input_file_path)
        if not os.path.exists(calibrated_input_file_path):
            raise Exception(f"Calibrated input file {calibrated_input_file_path} does not exist")
    
    #slow control file for good run list
    good_run_list_path = SLOW_CONTROL_GOOD_RUN_LIST_PATH

    #get hash of slow control file used
    full_hash = file_sha256(good_run_list_path)
    short_hash = full_hash[:10]

    #get run configuration from slow control
    run_data = get_run_database_data(good_run_list_path, args.run_number)
    run_configuration = run_data["trigger_name"]

    #get stable list of channels from slow control
    enabled_channels, channel_mask = get_stable_mpmt_list_slow_control(run_data)
    #the channels that are enabled less the channels that are determined as unstable
    stable_channels_card_chan = enabled_channels - channel_mask
    #map slow control channel list in card and channel to the mpmt slot and position
    slow_control_stable_channels = slot_pos_from_card_chan_list(stable_channels_card_chan)

    # query calibration database for manually-masked PMTs for this run
    caldb = CalibrationDBInterface()
    manually_masked_pmts = caldb.get_bad_pmts(args.run_number)
    print(f"Found {len(manually_masked_pmts)} manually masked PMTs in calibration DB for run {args.run_number}")
    
    #loop over each file    
    first_file_pmts_with_timing_constant = None
    for readout_window_file_name, calibrated_input_file_name in zip(args.input_files, calibrated_input_files):
        
        with uproot.open(readout_window_file_name) as readout_window_file:
            with uproot.open(calibrated_input_file_name) as calibrated_file:
                
                config_tree = calibrated_file["Configuration"]
                pmts_with_timing_constant = config_tree["wcte_pmts_with_timing_constant"].array().to_numpy()[0]
                
                #this list should be the same for all files in the run
                if first_file_pmts_with_timing_constant is None:
                    first_file_pmts_with_timing_constant = pmts_with_timing_constant
                else:
                    if not np.array_equal(first_file_pmts_with_timing_constant, pmts_with_timing_constant):
                        raise Exception("PMTs with timing constants do not match between files in the same run")
                
                # Construct output path
                base = os.path.splitext(os.path.basename(readout_window_file_name))[0]
                new_filename = f"{base}_self_trigger_dq_flags.root" 
                os.makedirs(args.output_dir, exist_ok=True)
                output_file_name = os.path.join(args.output_dir, new_filename)

                #create the output file
                with uproot.recreate(output_file_name) as outfile:
            
                    config_tree = outfile.mktree("Configuration", {
                        "git_hash": "string",
                        "run_configuration": "string",
                        "good_wcte_pmts": "var * int32",
                        "wcte_pmts_with_timing_constant": "var * int32",
                        "wcte_pmts_slow_control_stable": "var * int32",
                        "manually_masked_pmts": "var * int32",
                        "slow_control_file_name": "string",
                        "slow_control_file_hash": "string",
                        "readout_window_file": "string",
                        "calibrated_hit_file": "string"
                    })
                    print("There were",len(stable_channels_card_chan),"enabled channels not masked out")
                    print("There were",len(pmts_with_timing_constant),"channels with timing constants")
                    good_wcte_pmts = set(pmts_with_timing_constant) & set(slow_control_stable_channels) - set(manually_masked_pmts)
                    print("In total there are",len(good_wcte_pmts),"good channels with timing constants and stable in slow control")
                    
                    config_tree.extend({
                        "git_hash": [git_hash],
                        "run_configuration": [run_configuration],
                        "good_wcte_pmts": ak.Array([list(good_wcte_pmts)]),
                        "wcte_pmts_with_timing_constant": ak.Array([pmts_with_timing_constant]),
                        "wcte_pmts_slow_control_stable": ak.Array([slow_control_stable_channels]),
                        "manually_masked_pmts": ak.Array([list(manually_masked_pmts)]),
                        "slow_control_file_name": [good_run_list_path],
                        "slow_control_file_hash": [short_hash],
                        "readout_window_file": [readout_window_file_name],
                        "calibrated_hit_file": [calibrated_input_file_name]
                    })
                    
                    # Create a TTree to store the flags
                    tree = outfile.mktree("DataQualityFlags", { #only WCTE detector hits are stored here (not trigger mainboard hits)
                        "hit_pmt_readout_mask": "var * int32",
                        "window_data_quality_mask": "int32",
                        "readout_number": "int32" #the unique readout window number for this event in the run
                    })
                    
                    #batch load to get the
                    readout_window_tree_entries = readout_window_file["WCTEReadoutWindows"].num_entries 
                    calibrated_tree_entries = calibrated_file["CalibratedHits"].num_entries
                    if  readout_window_tree_entries!=calibrated_tree_entries:
                        if args.debug:
                            print("Warning: Input file problem different number of entries between calibrated and original file, but continuing due to debug mode.")
                            print("debug mode: override")
                            readout_window_tree_entries = min(readout_window_tree_entries,calibrated_tree_entries)
                        else:
                            raise Exception("Input file problem different number of entries between calibrated and original file")
                        
                    
                    batch_size = 10_000 #can use large batches as only a couple of branches are loaded
                    for start in range(0, readout_window_tree_entries, batch_size):  
                        stop = min(start + batch_size, readout_window_tree_entries)
                        print(f"Loading entries {start} → {stop}")
                        branches_to_load = ["window_time","readout_number"]
                        readout_window_tree = readout_window_file["WCTEReadoutWindows"]
                        readout_window_events = readout_window_tree.arrays(branches_to_load,library="ak", entry_start=start, entry_stop=stop)
                        
                        branches_to_load = ["readout_number","hit_mpmt_slot","hit_pmt_pos"]
                        calibrated_tree = calibrated_file["CalibratedHits"]
                        calibrated_file_events = calibrated_tree.arrays(branches_to_load,library="ak", entry_start=start, entry_stop=stop)
                        
                        if not np.array_equal(readout_window_events["readout_number"].to_numpy(),calibrated_file_events["readout_number"].to_numpy()):
                            raise Exception("Batch start",start,"different events being compared between two files")
                        
                        #trigger level flags
                        sc_good_trigger_mask = get_slow_control_trigger_mask(args.run_number,readout_window_events["window_time"].to_numpy(),run_data)
                        periodic_67ms_missing_mask = get_67ms_mask(args.run_number,readout_window_events["window_time"].to_numpy())
                        #make the trigger level bitmask 
                        trigger_mask = np.zeros_like(sc_good_trigger_mask, dtype=np.int32)                        
                        trigger_mask |= ~periodic_67ms_missing_mask * TriggerMask.PERIODIC_67_ISSUE.value
                        trigger_mask |= ~sc_good_trigger_mask * TriggerMask.SLOW_CONTROL_EXCLUDED.value
                         
                        #hit level flags
                        hit_global_id = (100*calibrated_file_events["hit_mpmt_slot"])+calibrated_file_events["hit_pmt_pos"]
                        hit_global_id_flat = ak.to_numpy(ak.flatten(hit_global_id))
                        
                        has_time_constant = np.isin(hit_global_id_flat,pmts_with_timing_constant)
                        is_sc_stable = np.isin(hit_global_id_flat,slow_control_stable_channels)
                        is_manually_masked = np.isin(hit_global_id_flat,manually_masked_pmts)
                        #make the hit level bitmask
                        hit_mask_flat = np.zeros_like(hit_global_id_flat, dtype=np.int32)
                        hit_mask_flat |= ~has_time_constant * HitMask.NO_TIMING_CONSTANT.value
                        hit_mask_flat |= ~is_sc_stable * HitMask.SLOW_CONTROL_EXCLUDED.value
                        hit_mask_flat |= is_manually_masked * HitMask.MANUALLY_MASKED.value
                        hit_mask = ak.unflatten(hit_mask_flat,ak.num(hit_global_id))
                        
                        #append to tree
                        tree.extend({
                            "hit_pmt_readout_mask": hit_mask,
                            "window_data_quality_mask": trigger_mask,
                            "readout_number": readout_window_events["readout_number"].to_numpy()
                        })
                        run_total_triggers += len(trigger_mask)
                        run_total_bad_triggers +=np.sum(trigger_mask!=0)
                        run_total_hits += len(hit_mask_flat)
                        run_total_bad_hits +=np.sum(hit_mask_flat!=0)
                        print("Batch processed",np.sum(trigger_mask==0),"/",len(trigger_mask),"good triggers", f"{np.sum(trigger_mask==0)/len(trigger_mask):.2%}")
                        print("Processed",np.sum(hit_mask_flat==0),"/",len(hit_mask_flat),"good hits", f"{np.sum(hit_mask_flat==0)/len(hit_mask_flat):.2%}")
    
    print("Finished processing run", args.run_number, "across", len(args.input_files), "files")

    bad_trig_pct = 100.0 * run_total_bad_triggers / run_total_triggers if run_total_triggers else 0.0
    bad_hit_pct  = 100.0 * run_total_bad_hits / run_total_hits if run_total_hits else 0.0

    metrics = {
        "n_good_pmt_channels": int(len(good_wcte_pmts)),
        "n_triggers":          int(run_total_triggers),
        "n_bad_triggers":      int(run_total_bad_triggers),
        "bad_trig_pct":        round(bad_trig_pct, 2),
        "n_hits":              int(run_total_hits),
        "n_bad_hits":          int(run_total_bad_hits),
        "bad_hit_pct":         round(bad_hit_pct, 2),
    }
    write_status_json(output_file_name, metrics)
    print("*** Self trigger DQ flags script complete ***")
