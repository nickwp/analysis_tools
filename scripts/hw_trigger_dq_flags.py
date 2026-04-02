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
    slot_pos_from_card_chan_list,
    get_channels_masked_by_problem,
    write_status_json,
)
from analysis_tools import CalibrationDBInterface

SLOW_CONTROL_GOOD_RUN_LIST_PATH = '/eos/experiment/wcte/configuration/slow_control_summary/all_run_list_v7.json'


def mask_windows_missing_waveforms(good_channel_list, readout_window_events):
    """
    Inputs:
    good_channel_list - list or array of integers in slot-position format (e.g. 203 for slot 2 position 3)
    readout_window_events - awkward array with fields:
        pmt_waveform_mpmt_slot_ids - awkward array of mPMT slot IDs for waveforms in each readout window
        pmt_waveform_pmt_position_ids - awkward array of PMT position IDs for waveforms in each readout window
    Returns:
    has_all_good_channels - awkward array of booleans indicating whether one waveform for every good channel
                            was present in the readout window
    """
    
    wf_mpmt_slot = readout_window_events["pmt_waveform_mpmt_slot_ids"]
    wf_pmt_pos = readout_window_events["pmt_waveform_pmt_position_ids"]
    wf_glbl_pmt_ids = (100 * wf_mpmt_slot) + wf_pmt_pos
    
    wf_glbl_pmt_ids_flat = ak.flatten(wf_glbl_pmt_ids).to_numpy()
    has_all_good_channels = np.full(len(wf_glbl_pmt_ids), True, dtype=bool)
    #iterate over each good channel and check if it is present exactly once in each readout window
    #if any channel is not then that window is marked as bad
    for ich, channel in enumerate(good_channel_list):
        if ich%100==0:
            print("Checking good channel",ich,"/",len(good_channel_list))
        channel_mask = wf_glbl_pmt_ids_flat == channel
        channel_counts = ak.sum(ak.unflatten(channel_mask, ak.num(wf_glbl_pmt_ids)), axis=1).to_numpy()
        has_all_good_channels = has_all_good_channels & (channel_counts == 1)
        
    return has_all_good_channels    

def mask_windows_missing_waveforms_fast(good_channel_list, readout_window_events):
    """
    Vectorised version of the above
    Inputs:
    good_channel_list - list of integers in slot-position format (e.g. 203 for slot 2 position 3)
    readout_window_events - awkward array with fields:
        pmt_waveform_mpmt_slot_ids - awkward array of mPMT slot IDs for waveforms in each readout window
        pmt_waveform_pmt_position_ids - awkward array of PMT position IDs for waveforms in each readout window
    Returns:
    has_all_good_channels - numpy bool array indicating whether one waveform for every good channel
                            was present in the readout window
    """
    wf_mpmt_slot = readout_window_events["pmt_waveform_mpmt_slot_ids"]
    wf_pmt_pos = readout_window_events["pmt_waveform_pmt_position_ids"]
    wf_glbl_pmt_ids = (100 * wf_mpmt_slot) + wf_pmt_pos

    n_windows = len(wf_glbl_pmt_ids)
    counts = ak.num(wf_glbl_pmt_ids).to_numpy()
    flat_ids = ak.flatten(wf_glbl_pmt_ids).to_numpy()
    window_idx = np.repeat(np.arange(n_windows), counts)

    # Filter to only hits from good channels
    is_good = np.isin(flat_ids, good_channel_list)
    good_flat_ids   = flat_ids[is_good]
    good_window_idx = window_idx[is_good]

    # Re-encode good channel IDs to compact 0..n_good-1 indices so the
    # count matrix is n_windows x n_good — avoids large-matrix column selection
    sorted_channels = np.sort(good_channel_list)
    encoded_ids = np.searchsorted(sorted_channels, good_flat_ids) #gives the index of the good channel in the sorted list   

    n_good = len(good_channel_list)
    count_matrix = np.zeros((n_windows, n_good), dtype=np.int32)
    np.add.at(count_matrix, (good_window_idx, encoded_ids), 1)

    # A window is good iff every good channel appears exactly once
    has_all_good_channels = np.all(count_matrix == 1, axis=1)

    return has_all_good_channels

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Apply data quality flags to hardware-trigger data.")
    parser.add_argument("-i", "--input_files", required=True, nargs='+', help="Path to WCTEReadoutWindows ROOT file(s)")
    parser.add_argument("-c", "--input_calibrated_file_directory", required=True, help="Path to calibrated hits directory")
    parser.add_argument("-hw", "--input_wf_processed_file_directory", required=True, help="Path to waveform-processed files directory")
    parser.add_argument("-r", "--run_number", required=True, help="Run number")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to write output file")
    parser.add_argument("--debug", action="store_true", help="Enable debug - disables checks allowing for test runs")
    args = parser.parse_args()
    
    git_hash = get_git_descriptor(debug=args.debug)

    
    #check that the run number is correct 
    for input_file in args.input_files:
        if f"R{args.run_number}" not in input_file:
            raise Exception(f"Input file {input_file} does not match run number {args.run_number}")
    
    #make a list of calibrated input files - these are needed to determine for the channel list
    #of channels with calibration constants and for the hit list for which the mask is to be applied
    #make list of waveform processed files - these are needed to determine list of channels with short
    #waveforms masked out 
    calibrated_input_files = []
    wf_processed_files = []
    
    #counters for statistics
    run_total_triggers = 0
    run_total_bad_triggers = 0
    run_total_hits = 0
    run_total_bad_hits = 0
    
    for input_file in args.input_files:
        base = os.path.splitext(os.path.basename(input_file))[0]
        calibrated_file_name = f"{base}_processed_waveforms_calibrated_hits.root" 
        calibrated_input_file_path = os.path.join(args.input_calibrated_file_directory, calibrated_file_name)
        calibrated_input_files.append(calibrated_input_file_path)
        if not os.path.exists(calibrated_input_file_path):
            raise Exception(f"Calibrated input file {calibrated_input_file_path} does not exist")

        wf_processed_file_name = f"{base}_processed_waveforms.root" 
        wf_processed_file_path = os.path.join(args.input_wf_processed_file_directory, wf_processed_file_name)        
        wf_processed_files.append(wf_processed_file_path)
        if not os.path.exists(wf_processed_file_path):
            raise Exception(f"Waveform processed input file {wf_processed_file_path} does not exist")
        
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
    for readout_window_file_name, calibrated_input_file_name, wf_processed_file_name in zip(args.input_files, calibrated_input_files, wf_processed_files):
        #open the original file, the calibrated hits file and the waveform processed file
        with uproot.open(readout_window_file_name) as readout_window_file:
            with uproot.open(calibrated_input_file_name) as calibrated_file:
                with uproot.open(wf_processed_file_name) as wf_processed_file:
                    
                    #get the list of pmts with timing constants from the calibrated file
                    config_tree = calibrated_file["Configuration"]
                    pmts_with_timing_constant = config_tree["wcte_pmts_with_timing_constant"].array().to_numpy()[0]
                
                    #Check the lists are the same between files in the same run 
                    if first_file_pmts_with_timing_constant is None:
                        first_file_pmts_with_timing_constant = pmts_with_timing_constant
                    else:
                        if not np.array_equal(first_file_pmts_with_timing_constant, pmts_with_timing_constant):
                            raise Exception("PMTs with timing constants do not match between files in the same run")
                                        
                    #construct the good wcte pmt list
                    good_wcte_pmts = list((set(pmts_with_timing_constant) & set(slow_control_stable_channels)) - set(manually_masked_pmts))
                    
                    # Retrieve mask arrays for each specific problem
                    problem_to_branch = {
                        'bad current': 'slow_control_mask_bad_current',
                        'bad pmt status': 'slow_control_mask_bad_pmt_status',
                        'coarse counter reset': 'slow_control_mask_coarse_counter_reset_failed',
                        'data rate too low': 'slow_control_mask_missing_monitoring_data',
                        'no data': 'slow_control_mask_no_data',
                        'packet rate': 'slow_control_mask_sporadic_monitoring_packets',
                        'trip!': 'slow_control_mask_pmt_trip'
                    }
                    
                    masked_by_problem_slot_pos = {}
                    for problem_name, branch_name in problem_to_branch.items():
                        card_chan_mask = get_channels_masked_by_problem(run_data, problem_name)
                        #only consider channels that are enabled as disabled channels can be not in the detector
                        card_chan_mask = card_chan_mask & enabled_channels
                        masked_by_problem_slot_pos[branch_name] = slot_pos_from_card_chan_list(card_chan_mask)
                    
                    # Construct output path
                    base = os.path.splitext(os.path.basename(readout_window_file_name))[0]
                    new_filename = f"{base}_hw_trigger_dq_flags.root" 
                    os.makedirs(args.output_dir, exist_ok=True)
                    output_file_name = os.path.join(args.output_dir, new_filename)
                    with uproot.recreate(output_file_name) as outfile:
                
                        config_tree = outfile.mktree("Configuration", {
                            "git_hash": "string",
                            "run_configuration": "string",
                            "good_wcte_pmts": "var * int32",
                            "wcte_pmts_with_timing_constant": "var * int32",
                            "wcte_pmts_slow_control_stable": "var * int32",
                            "slow_control_mask_bad_current": "var * int32",
                            "slow_control_mask_bad_pmt_status": "var * int32",
                            "slow_control_mask_coarse_counter_reset_failed": "var * int32",
                            "slow_control_mask_missing_monitoring_data": "var * int32",
                            "slow_control_mask_no_data": "var * int32",
                            "slow_control_mask_sporadic_monitoring_packets": "var * int32",
                            "slow_control_mask_pmt_trip": "var * int32",
                            "manually_masked_pmts": "var * int32",
                            "slow_control_file_name": "string",
                            "slow_control_file_hash": "string",
                            "readout_window_file": "string",
                            "calibrated_hit_file": "string",
                            "wf_processed_file_name": "string"
                        })
                        print("There were",len(stable_channels_card_chan),"enabled channels not masked out")
                        print("There were",len(pmts_with_timing_constant),"channels with timing constants")
                        print("In total there are",len(good_wcte_pmts),"good channels with timing constants and stable in slow control")
                        
                        config_tree.extend({
                            "git_hash": [git_hash],
                            "run_configuration": [run_configuration],
                            "good_wcte_pmts": ak.Array([good_wcte_pmts]),
                            "wcte_pmts_with_timing_constant": ak.Array([pmts_with_timing_constant]),
                            "wcte_pmts_slow_control_stable": ak.Array([slow_control_stable_channels]),
                            "slow_control_mask_bad_current": ak.Array([masked_by_problem_slot_pos['slow_control_mask_bad_current']]),
                            "slow_control_mask_bad_pmt_status": ak.Array([masked_by_problem_slot_pos['slow_control_mask_bad_pmt_status']]),
                            "slow_control_mask_coarse_counter_reset_failed": ak.Array([masked_by_problem_slot_pos['slow_control_mask_coarse_counter_reset_failed']]),
                            "slow_control_mask_missing_monitoring_data": ak.Array([masked_by_problem_slot_pos['slow_control_mask_missing_monitoring_data']]),
                            "slow_control_mask_no_data": ak.Array([masked_by_problem_slot_pos['slow_control_mask_no_data']]),
                            "slow_control_mask_sporadic_monitoring_packets": ak.Array([masked_by_problem_slot_pos['slow_control_mask_sporadic_monitoring_packets']]),
                            "slow_control_mask_pmt_trip": ak.Array([masked_by_problem_slot_pos['slow_control_mask_pmt_trip']]),
                            "manually_masked_pmts": ak.Array([list(manually_masked_pmts)]),
                            "slow_control_file_name": [good_run_list_path],
                            "slow_control_file_hash": [short_hash],
                            "readout_window_file": [readout_window_file_name],
                            "calibrated_hit_file": [calibrated_input_file_name],
                            "wf_processed_file_name": [wf_processed_file_name]
                        })
                        
                        # Create a TTree to store the flags
                        tree = outfile.mktree("DataQualityFlags", { #only WCTE detector hits are stored here (not trigger mainboard hits)
                            "hit_pmt_readout_mask": "var * int32",
                            "window_data_quality_mask": "int32",
                            "readout_number": "int32" #the unique readout window number for this event in the run
                        })
                        
                        file_total_triggers = 0
                        file_total_bad_triggers = 0
                        file_total_hits = 0
                        file_total_bad_hits = 0

                        #batch load over all entries to apply data quality flags to each trigger and hits
                        readout_window_tree_entries = readout_window_file["WCTEReadoutWindows"].num_entries 
                        calibrated_tree_entries = calibrated_file["CalibratedHits"].num_entries
                        wf_processed_input_file_entries = wf_processed_file["ProcessedWaveforms"].num_entries
                        
                        if  readout_window_tree_entries!=calibrated_tree_entries or wf_processed_input_file_entries!=readout_window_tree_entries:
                            if args.debug:
                                print("Warning: Input file problem different number of entries between calibrated and original file, but continuing due to debug mode.")
                                print("debug mode: override")
                                readout_window_tree_entries = min(readout_window_tree_entries,calibrated_tree_entries)
                                readout_window_tree_entries = min(readout_window_tree_entries,wf_processed_input_file_entries)
                            else:
                                raise Exception("Input file problem different number of entries between calibrated and original file")
                            
                        batch_size = 10_000 #can use large batches as only a couple of branches are loaded
                        for start in range(0, readout_window_tree_entries, batch_size):  
                            stop = min(start + batch_size, readout_window_tree_entries)
                            print(f"Loading entries {start} → {stop}")
                            branches_to_load = ["window_time","readout_number","pmt_waveform_mpmt_slot_ids","pmt_waveform_pmt_position_ids"]
                            readout_window_tree = readout_window_file["WCTEReadoutWindows"]
                            readout_window_events = readout_window_tree.arrays(branches_to_load,library="ak", entry_start=start, entry_stop=stop)
                            
                            branches_to_load = ["readout_number","hit_mpmt_slot","hit_pmt_pos"]
                            calibrated_tree = calibrated_file["CalibratedHits"]
                            calibrated_file_events = calibrated_tree.arrays(branches_to_load,library="ak", entry_start=start, entry_stop=stop)
                            
                            branches_to_load = ["readout_number","trigger_time","missing_trigger_flag","event_bad_waveform_lengths_flag"]
                            wf_processed_tree = wf_processed_file["ProcessedWaveforms"]
                            wf_processed_events = wf_processed_tree.arrays(branches_to_load, library="ak", entry_start=start, entry_stop=stop)
                            
                            if not np.array_equal(readout_window_events["readout_number"].to_numpy(),calibrated_file_events["readout_number"].to_numpy()):
                                raise Exception("Batch start",start,"different events being compared between two files")
                            
                            #trigger level flags
                            missing_waveforms_mask = mask_windows_missing_waveforms_fast(good_wcte_pmts, readout_window_events)
                            print("Checked",len(missing_waveforms_mask),"windows for missing waveforms", np.sum(~missing_waveforms_mask), "bad windows found")
                            
                            #make the trigger/window level bitmask 
                            trigger_mask = np.zeros_like(missing_waveforms_mask, dtype=np.int32)                        
                            trigger_mask |= ~missing_waveforms_mask * TriggerMask.MISSING_WAVEFORMS.value
                            trigger_mask |= np.array(wf_processed_events["missing_trigger_flag"]) * TriggerMask.MISSING_TRIGGER_SIGNAL.value
                            trigger_mask |= np.array(wf_processed_events["event_bad_waveform_lengths_flag"]) * TriggerMask.MISMATCHED_WAVEFORM_LENGTH.value
                            
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
                            
                            file_total_triggers += len(trigger_mask)
                            file_total_bad_triggers += np.sum(trigger_mask!=0)
                            file_total_hits += len(hit_mask_flat)
                            file_total_bad_hits += np.sum(hit_mask_flat!=0)
                        
                            print("Batch processed",np.sum(trigger_mask==0),"/",len(trigger_mask),"good triggers", f"{np.sum(trigger_mask==0)/len(trigger_mask):.2%}")
                            print("Processed",np.sum(hit_mask_flat==0),"/",len(hit_mask_flat),"good hits", f"{np.sum(hit_mask_flat==0)/len(hit_mask_flat):.2%}")
                        
                        bad_trig_pct = 100.0 * file_total_bad_triggers / file_total_triggers if file_total_triggers else 0.0
                        bad_hit_pct  = 100.0 * file_total_bad_hits / file_total_hits if file_total_hits else 0.0
                        
                        metrics_tree = outfile.mktree("Metrics", {
                            "n_good_pmt_channels": "int32",
                            "n_triggers": "int32",
                            "n_bad_triggers": "int32",
                            "bad_trig_pct": "float64",
                            "n_hits": "int32",
                            "n_bad_hits": "int32",
                            "bad_hit_pct": "float64"
                        })
                        metrics_tree.extend({
                            "n_good_pmt_channels": [len(good_wcte_pmts)], 
                            "n_triggers": [int(file_total_triggers)],
                            "n_bad_triggers": [int(file_total_bad_triggers)],
                            "bad_trig_pct": [float(bad_trig_pct)],
                            "n_hits": [int(file_total_hits)],
                            "n_bad_hits": [int(file_total_bad_hits)],
                            "bad_hit_pct": [float(bad_hit_pct)]
                        })
    
    print("Finished processing run", args.run_number, "across", len(args.input_files), "files")
    print("*** Hardware trigger DQ flags script complete ***")
