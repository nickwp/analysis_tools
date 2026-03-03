## This script creates a new root file with the calibration constants applied to the hit times and charges
# It is designed to work on WCTEReadoutWindows trees or trees from waveform processed data files
import numpy as np
import uproot
import awkward as ak
import argparse
import os
import time
from analysis_tools import CalibrationDBInterface
from analysis_tools.production_utils import get_git_descriptor


def process_data(input_file_names, output_dir, config_dict, timing_offsets_glb_pmt_id, timing_offsets_values, debug=False):
        
    timing_offset_lookup = np.zeros(max(timing_offsets_glb_pmt_id)+1, dtype=np.float64)
    timing_offset_lookup[timing_offsets_glb_pmt_id] = timing_offsets_values

    for input_file_name in input_file_names:
        if not os.path.exists(input_file_name):
            print("File",input_file_name," does not exist")
            continue

        # Construct output path
        base = os.path.splitext(os.path.basename(input_file_name))[0]
        new_filename = f"{base}_calibrated_hits.root"
        os.makedirs(output_dir, exist_ok=True)
        output_file_name = os.path.join(output_dir, new_filename)
        
        with uproot.recreate(output_file_name) as outfile:
            # Create a TTree with two variable-length branches
            tree = outfile.mktree("CalibratedHits", { #only WCTE detector hits are stored here (not trigger mainboard hits)
                "hit_pmt_calibrated_times": "var * float64",
                "hit_pmt_charges": "var * float64",
                "hit_mpmt_slot": "var * int32",
                "hit_pmt_pos": "var * int32",
                "hit_missing_time_constant_flag": "var * bool",  #flag if there was no time constant for this hit
                "hit_original_readout_window_index": "var * int32", #the index of the hit in the original readout window array before filtering
                "readout_number": "int32" #the unique readout window number for this event in the run
            })
            
            #store variables from configuration dictionary in a configuration tree
            config_tree = outfile.mktree("Configuration", {
                "git_hash": "string", #git hash of the code used 
                "wcte_pmts_with_timing_constant": "var * int32", #the global pmt id (slot*100 + pos) of pmts with timing constants
                "timing_constant_revision_id": "int32", #the revision id of the timing constant set used
                "timing_constant_insert_time": "string", #the insert time of the timing constants in the db set used
                "timing_constant_official_flag": "int32", #flag if the timing constants used were official
                "input_file_name": "string" # the file name of the input file processed
            })
        
            config_tree.extend({
                "git_hash": [config_dict["git_hash"]],
                "wcte_pmts_with_timing_constant": ak.Array([config_dict["timing_offsets_glb_pmt_id"]]),
                "timing_constant_revision_id": [config_dict["timing_constant_revision_id"]],
                "timing_constant_official_flag": [config_dict["timing_constant_official_flag"]],
                "timing_constant_insert_time": [config_dict["timing_constant_insert_time"] ],
                "input_file_name": [input_file_name]
            })
            
            batch_size = 1000
            
            with uproot.open(input_file_name) as root_file:
                
                #determine if this is an offline data file or waveform processed file
                if "processed_waveforms.root" in input_file_name and ("ProcessedWaveforms" in root_file):
                    print("This is a processed waveforms file")
                    tree_name = "ProcessedWaveforms"
                    hit_mpmt_card_ids_branch_name = "hit_card"
                    hit_pmt_times_branch_name = "hit_time"
                    hit_pmt_charges_branch_name = "hit_charge"
                    hit_mpmt_slot_ids_branch_name = "hit_mpmt_slot"
                    hit_pmt_position_ids_branch_name = "hit_pmt_pos"
                elif "WCTEReadoutWindows" in root_file:
                    print("This is a readout window file")
                    tree_name = "WCTEReadoutWindows"
                    hit_mpmt_card_ids_branch_name = "hit_mpmt_card_ids"
                    hit_pmt_times_branch_name = "hit_pmt_times"
                    hit_pmt_charges_branch_name = "hit_pmt_charges"
                    hit_mpmt_slot_ids_branch_name = "hit_mpmt_slot_ids"
                    hit_pmt_position_ids_branch_name = "hit_pmt_position_ids"
                else:
                    raise Exception(f"Input file {input_file_name} does not contain recognized tree")
                
                input_tree = root_file[tree_name]
                total_entries = input_tree.num_entries
                all_branches = input_tree.keys()
                
                #open the input file in batches
                for start in range(0, total_entries, batch_size):  
                    stop = min(start + batch_size, total_entries)
                    if debug and start >= 5000:
                        print("Stopping after 5000 events for testing")
                        break
                    print(f"Loading entries {start} → {stop}")
                    start_batch = time.time()

                    branches_to_load = [hit_mpmt_card_ids_branch_name, hit_pmt_times_branch_name, hit_pmt_charges_branch_name, hit_mpmt_slot_ids_branch_name, hit_pmt_position_ids_branch_name, "readout_number"]   
                    events = input_tree.arrays(branches_to_load,library="ak", entry_start=start, entry_stop=stop)
                    end = time.time()
                    print(f"Time load batch data: {end - start_batch:.6f} seconds")

                    batch_hit_calibrated_times = []
                    batch_hit_calibrated_charges = []
                    batch_hit_slot = []
                    batch_hit_pmt_pos = []
                    batch_hit_missing_time_constant_flag = []
                    batch_hit_original_readout_window_index = []
                    batch_event_readout_number = []
            
                    for iev, event in enumerate(events):
                        if iev%100==0:
                            print("On iev",iev)
                        
                        start = time.time()
                    
                        #filter only for hits in the detector (cards <130)
                        wcte_hit_filter = event[hit_mpmt_card_ids_branch_name]<130
                        
                        wcte_hit_time = event[hit_pmt_times_branch_name][wcte_hit_filter].to_numpy()
                        wcte_hit_charge = event[hit_pmt_charges_branch_name][wcte_hit_filter].to_numpy()
                        wcte_hit_mpmt_slot = event[hit_mpmt_slot_ids_branch_name][wcte_hit_filter].to_numpy()
                        wcte_hit_pmt_pos = event[hit_pmt_position_ids_branch_name][wcte_hit_filter].to_numpy()
                        wcte_hit_mpmt_cards = event[hit_mpmt_card_ids_branch_name][wcte_hit_filter].to_numpy()
                        event_readout_number = event["readout_number"]
                        
                        hit_glb_pmt_id = wcte_hit_mpmt_slot*100 + wcte_hit_pmt_pos
                        has_time_constant_filter = np.isin(hit_glb_pmt_id,timing_offsets_glb_pmt_id)
                        
                        #timing_offset_lookup returns 0 where there is no constant
                        hit_time_constants = timing_offset_lookup[hit_glb_pmt_id]

                        calibrated_times = wcte_hit_time - hit_time_constants
                        calibrated_charges = wcte_hit_charge
                        
                        #get the original index of the hit in the unfiltered array
                        original_indices = np.arange(len(event[hit_pmt_times_branch_name]))
                        original_indices_filtered = original_indices[wcte_hit_filter]   
                                         
                        end = time.time()
                        if iev==234: print(f"Estimated hit processing: {batch_size*(end - start):.6f} seconds")
                                                
                        batch_hit_calibrated_times.append(calibrated_times.tolist())
                        batch_hit_calibrated_charges.append(calibrated_charges.tolist())
                        batch_hit_slot.append(wcte_hit_mpmt_slot.tolist())
                        batch_hit_pmt_pos.append(wcte_hit_pmt_pos.tolist())
                        batch_hit_missing_time_constant_flag.append(has_time_constant_filter.tolist())
                        batch_event_readout_number.append(event_readout_number)
                        batch_hit_original_readout_window_index.append(original_indices_filtered.tolist())
                    #finished processing batch
                    start = time.time()
                    batch_hit_calibrated_times = ak.Array(batch_hit_calibrated_times)
                    batch_hit_calibrated_charges = ak.Array(batch_hit_calibrated_charges)
                    batch_hit_slot = ak.Array(batch_hit_slot)
                    batch_hit_pmt_pos = ak.Array(batch_hit_pmt_pos)
                    batch_hit_missing_time_constant_flag = ak.Array(batch_hit_missing_time_constant_flag)
                    batch_event_readout_number = ak.Array(batch_event_readout_number)
                    batch_hit_original_readout_window_index = ak.Array(batch_hit_original_readout_window_index)
                    end = time.time()

                    tree.extend({
                        "hit_pmt_calibrated_times": batch_hit_calibrated_times,
                        "hit_pmt_charges": batch_hit_calibrated_charges,
                        "hit_mpmt_slot": batch_hit_slot,
                        "hit_pmt_pos": batch_hit_pmt_pos,
                        "hit_missing_time_constant_flag": batch_hit_missing_time_constant_flag,
                        "hit_original_readout_window_index": batch_hit_original_readout_window_index,
                        "readout_number": batch_event_readout_number
                    })
                    end_batch = time.time()
                    print(f"Time for full batch: {end_batch - start_batch:.6f} seconds")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calibrate hit times and charges using timing constants from the calibration database.")
    parser.add_argument("-i", "--input_files", required=True, nargs='+', help="Path to input ROOT file or files")
    parser.add_argument("-r", "--run_number", required=True, help="Run number")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to write output file")
    parser.add_argument("--not_official_const", action="store_true", help="Flag to set official = false in const lookup")
    parser.add_argument("--debug", action="store_true", help="Enable debug - disables checks allowing for test runs")
    args = parser.parse_args()
    
    #check that the run number is correct 
    for input_file in args.input_files:
        if f"R{args.run_number}" not in input_file:
            raise Exception(f"Input file {input_file} does not match run number {args.run_number}")
    
    #get the git hash of the repo
    config_dict = {}
    git_hash = get_git_descriptor(debug=args.debug)
    config_dict["git_hash"] = git_hash
    
    #get calibration constants
    calibration_db_interface = CalibrationDBInterface()
    official = 1
    if args.not_official_const:
        official = 0     
    print("official is",official,args.not_official_const)
    
    timing_offsets_list, timing_constant_revision_id, timing_constant_insert_time = calibration_db_interface.get_calibration_constants(args.run_number, 0, "timing_offsets", official)
    #split the offset list into two arrays for fast lookup
    timing_offsets_glb_pmt_id = np.array([offset['glb_pmt_id'] for offset in timing_offsets_list])
    timing_offsets_values = np.array([offset['data']['timing_offset'] for offset in timing_offsets_list])

    config_dict["timing_offsets_glb_pmt_id"] = timing_offsets_glb_pmt_id
    config_dict["timing_constant_revision_id"] = timing_constant_revision_id
    config_dict["timing_constant_insert_time"] = timing_constant_insert_time
    config_dict["timing_constant_official_flag"] = official
    start = time.time()
    process_data(args.input_files, args.output_dir,config_dict, timing_offsets_glb_pmt_id, timing_offsets_values, debug=args.debug)
    end = time.time()
    print(f"Process all data time: {end - start:.3f} seconds")
    print("*** Calibrate hits script complete ***")
