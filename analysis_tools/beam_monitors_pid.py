#this code holds the functions necessary for reading in the data and identifying the paticle types based on the WCTE beam monitors 

import os
import json
from matplotlib.backends.backend_pdf import PdfPages
import uproot
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
import pandas as pd
import awkward as ak
import pyarrow as pa, pyarrow.parquet as pq
import gc

#for the nice progress bar
from tqdm import tqdm

import os, shutil, subprocess, time, hashlib

from collections import defaultdict

import sys
# sys.path.append("/eos/user/a/acraplet/analysis_tools/")
#the module for reading the yaml file with the detector distances and dimensions - from Bruno
from .read_beam_detector_distances import DetectorDB as db

#Helper functions for file reading, written by Sahar
def stage_local(src_eos_path: str, min_free_gb=20, min_bytes=1_000_000) -> str:
    st = shutil.disk_usage("/tmp")
    if st.free/1e9 < min_free_gb:
        print("Not enough /tmp space; will stream from EOS")
        return ""

    # Use a unique local name to avoid collisions across different directories
    h = hashlib.md5(src_eos_path.encode()).hexdigest()[:8]
    local = f"/tmp/{os.path.basename(src_eos_path)}.{h}"

    def good(p): return os.path.exists(p) and os.path.getsize(p) >= min_bytes

    if not good(local):
        if os.path.exists(local):
            print("Cached local copy is too small; re-staging…")
            os.remove(local)
        print("Staging ROOT file to local disk (xrdcp)…")
        subprocess.run(["xrdcp", "-f", to_xrootd(src_eos_path), local], check=True)
        if not good(local):
            raise OSError(f"Local file too small after xrdcp: {local}")

    print("Using local copy:", local)
    return local


def to_xrootd(path: str) -> str:
    assert path.startswith("/eos/")
    return "root://eosuser.cern.ch//eos" + path[4:]


def make_blocks(idx: np.ndarray, max_block: int):
    if idx.size == 0:
        return []
    blocks = []
    start = idx[0]
    last  = idx[0]
    for v in idx[1:]:
        # if extending the block stays ≤ max_block, keep extending
        if (v - start) < max_block:
            last = v
        else:
            blocks.append((int(start), int(last)+1))
            start = last = v
    blocks.append((int(start), int(last)+1))
    return blocks


def make_flag_map(flags):
    """
    Build a deterministic mapping from flag name -> bit index (0..).
    Uses sorted order of flag names so mapping is reproducible.
    """
    return {name: idx for idx, name in enumerate(sorted(flags))}

def write_event_quality_mask(flag_dict,
                             flag_map):
    """
    Pack flags into an integer bitmask.
    
    Parameters
    - flag_dict: mapping of flag name -> truthy/falsey value (bool, 0/1, etc.)
    - flag_map: optional mapping flag name -> bit index (int).
                If None, a deterministic mapping from sorted(flag_dict.keys()) is used.
                
    Returns
    - mask: int where bit i (value 2**i) is set when the corresponding flag is true.
    """
    if flag_map is None:
        flag_map = make_flag_map(flag_dict.keys())

    mask = 0
    for flag_name, bit_idx in flag_map.items():
        if flag_name not in flag_dict:
            # Missing flags are treated as False (not set). Change if you want an error.
            continue
        if bool(flag_dict[flag_name]):
            mask |= (1 << bit_idx)
    return mask

def read_event_quality_mask(mask,
                            flag_map):
    """
    Unpack an integer bitmask back into a flag dictionary.
    
    Parameters
    - mask: integer bitmask
    - flag_map: mapping flag name -> bit index used when the mask was created
    
    Returns
    - dict of flag name -> bool (True if that bit is set)
    """
    result: Dict[str, bool] = {}
    for flag_name, bit_idx in flag_map.items():
        result[flag_name] = bool(mask & (1 << bit_idx))
    return result

def _deduplicate_tdc_hits(ids, times):
    "Function used to remove the later TDC hits and corresponding channel ids in case there are more than one"
    seen = set()
    keep_ids = []
    keep_times = []
    duplicates_removed = defaultdict(int)
    stored_times = defaultdict(int)
    
    for ch, t in zip(ids, times):
        if ch in seen:
            #here we are counting how many events are being removed per channel
            duplicates_removed[ch] += 1
            if stored_times[ch] > t:
                #in case they are not ordered correctly, this is important
                raise NotTheFirstHit
            continue
        seen.add(ch)
        keep_ids.append(ch)
        keep_times.append(t)
        stored_times[ch] = t

    #in case we are not removing any channel, can give it back as is
    if not duplicates_removed:
        return ids, times, duplicates_removed

    #ensure that the format on the way out is the same as on the way in
    if isinstance(ids, np.ndarray):
        ids_clean = np.asarray(keep_ids, dtype=ids.dtype)
    else:
        ids_clean = np.asarray(keep_ids)
    if isinstance(times, np.ndarray):
        times_clean = np.asarray(keep_times, dtype=times.dtype)
    else:
        times_clean = np.asarray(keep_times)
    return ids_clean, times_clean, duplicates_removed


def _tdc_requirement_met(group, tdc_set):
    if group["mode"] == "any_pair":
        return any(all(ch in tdc_set for ch in pair) for pair in group["channels"])
    return all(ch in tdc_set for ch in group["channels"])


#these are replaced with more accurate estimates using the travel time of all particles
# proton_tof_cut = 17.5 #ad-hoc but works for most analyses
# deuteron_tof_cut = 35 #35 #ad-hoc but works for most analyses
helium3_tof_cut = 30 #30 #ad-hoc 
tritium_tof_cut = 80 #ad-hoc 
lithium6_tof_cut = 90 #ad-hoc 
        

#Default informations



c = 0.299792458 #in m.ns^-1 do not change the units please as these are called throuhout
L =  444.03 #4.3431
L_t0t4 = 305.68 #2.9485
L_t4t1 = 143.38 #1.3946

# Particle masses in GeV/c^2
particle_masses = {
    "Electrons": 0.000511,
    "Muons": 0.105658,
    "Pions": 0.13957,
    "Protons": 0.938272
}

reference_ids = (31, 46)          # (TDC ref for IDs <31, ref1 for IDs >31)
t0_group     = [0, 1, 2, 3]       # must all be present
t1_group     = [4, 5, 6, 7]       # must all be present
t4_group     = [42, 43]           # must all be present
t4_qdc_cut   = 200                # Only hits above this value
ACT0_group   = (12, 13)                
ACT1_group   = (14, 15)
ACT2_group   = (16, 17)                

ACT3_group   = (18, 19)                
ACT4_group   = (20, 21)                 
ACT5_group   = (22, 23)                
act_eveto_group = [12, 13, 14, 15, 16, 17]   # ACT-eveto channels
act_tagger_group = [18, 19, 20, 21, 22, 23]
hc_group = [9, 10]
hc_charge_cut = 150

t5_b0_group = [48, 56]     #T5, also known as the TOF  detector
t5_b1_group = [49, 57]     #Pairs of siPMs on either side of a given bar
t5_b2_group = [50, 58]     #Both SiPMs need to be above threshold for at least 
t5_b3_group = [51, 59]     #One of the bars for the event to be kept
t5_b4_group = [52, 60]
t5_b5_group = [53, 61]
t5_b6_group = [54, 62]
t5_b7_group = [55, 63]
t5_total_group = [t5_b0_group, t5_b1_group, t5_b2_group, t5_b3_group,
                  t5_b4_group, t5_b5_group, t5_b6_group, t5_b7_group]

#basic functions, necessary in general
def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

#three gaussian fit, necessary for fitting the T4 TOF distributions
def three_gaussians(x, amp, mean, sigma,  amp1, mean1, sigma1,  amp2, mean2, sigma2):
    return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + amp1 * np.exp(-0.5 * ((x - mean1) / sigma1) ** 2) + amp2 * np.exp(-0.5 * ((x - mean2) / sigma2) ** 2)


def landau_gauss_convolution(x, amp, mpv, eta, sigma):
    x = np.asarray(x, dtype=float)
    sigma = max(float(sigma), 1e-3)
    eta = max(float(eta), 1e-3)
    # Keep the integration domain within the physical (positive) region to avoid
    # numerical overflow in the Landau tail.
    t_min = max(mpv - 5.0 * eta - 5.0 * sigma, 0.0)
    t_max = mpv + 15.0 * eta + 5.0 * sigma
    if t_max <= t_min:
        t_max = t_min + max(eta, sigma, 1.0)
    t = np.linspace(t_min, t_max, 2000)
    with np.errstate(over="ignore", under="ignore"):
        log_pdf = moyal.logpdf(t, loc=mpv, scale=eta)
    # clip to keep exponentiation stable
    log_pdf = np.clip(log_pdf, -700, 50)
    landau_pdf = np.exp(log_pdf)
    gauss = np.exp(-0.5 * ((x[:, None] - t[None, :]) / sigma) ** 2) / (
        sigma * np.sqrt(2.0 * np.pi)
    )
    conv = np.trapz(landau_pdf * gauss, t, axis=1)
    return amp * conv



def fit_gaussian(entries, bin_centers):
    # Get bin centers from edges

    amp_guess = np.max(entries)
    mean_guess = bin_centers[np.argmax(entries)]
    sigma_guess = np.std(np.repeat(bin_centers, entries.astype(int)))

    popt, pcov = curve_fit(gaussian, bin_centers, entries,
                           p0=[amp_guess, mean_guess, sigma_guess])
    
    return popt, pcov


def fit_three_gaussians(entries, bin_centers):
    # Get bin centers from edges

    amp_guess = np.max(entries)
    mean_guess = bin_centers[np.argmax(entries)]
    sigma_guess = np.std(np.repeat(bin_centers, entries.astype(int)))

    popt, pcov = curve_fit(three_gaussians, bin_centers, entries,
                           p0=[amp_guess, mean_guess, sigma_guess, amp_guess/100, mean_guess-4, sigma_guess,  amp_guess/100, mean_guess+4, sigma_guess])
    
    return popt, pcov




class BeamAnalysis:
    def __init__(self, run_number, run_momentum, n_eveto, n_tagger, there_is_ACT5, output_dir, pdf_name=None):
        #Store the run characteristics
        self.run_number, self.run_momentum = run_number, run_momentum
        self.n_eveto, self.n_tagger = n_eveto, n_tagger
        self.there_is_ACT5 = there_is_ACT5
        self.output_dir = output_dir
        
        if pdf_name is None:
            pdf_name = f"PID_run{run_number}_p{run_momentum}.pdf"
        pdf_path = os.path.join(output_dir, pdf_name)
        self.pdf_global = PdfPages(pdf_path)
        self.channel_mapping = {12: "ACT0-L", 13: "ACT0-R", 14: "ACT1-L", 15: "ACT1-R", 16: "ACT2-L", 17: "ACT2-R", 18: "ACT3-L", 19: "ACT3-R", 20: "ACT4-L", 21: "ACT4-R", 22: "ACT5-L", 23: "ACT5-R"}
        print("Initialised the BeamAnalysis instance")
        print(f"Plots will be saved to {pdf_path}")
        
        
        
        
        
    def end_analysis(self):
        self.pdf_global.close()
        
    def open_file(self, n_events = -1, require_t5 = False, first_tdc_only = True, enforce_tdc_qdc_match = True, input_file = None):
        '''Read in the data as a pandas dataframe, read in the TOF and the ACt information'''
        
        self.require_t5_hit = require_t5
        
        if input_file is None:
            raise ValueError("input_file must be specified")
        file_path = input_file
        tree_name    = "WCTEReadoutWindows"

        # Open the file and grab the tree
        f    = uproot.open(file_path)
        tree = f[tree_name]

        # Load all four branches into NumPy arrays
        branches = [
            "beamline_pmt_tdc_times",
            "beamline_pmt_tdc_ids",
            "beamline_pmt_qdc_charges",
            "beamline_pmt_qdc_ids",
            "spill_counter"
        ]

        #low number of entries for testing
        if n_events == -1:
            data = tree.arrays(branches, library="np")
        else:
            data = tree.arrays(branches, library="np", entry_start = 0, entry_stop = n_events)
        
        
        #read the calibration file
        with open('../include/1pe_calibration.json', 'r') as file:
            calib_constants = json.load(file)

        # Access the calibration constants
        calibration = calib_constants["BeamCalibrationConstants"][0]
       
        
        calib_map = {
            ch: (gain, ped)
            for ch, gain, ped in zip(
                calibration["channel_id"],
                calibration["gain_value"],
                calibration["pedestal_mean"]
            )
        }
        
        #set those cuts to 0 incase we are in a negative polarity run
        self.proton_tof_cut = 0 
        self.deuteron_tof_cut = 0
        
        if self.run_momentum > 0:
            #Define the cut lines in terms of the expected tof
            _, _, proton_tof_cut_array, _, _ =self.give_theoretical_TOF("Protons", np.array([self.run_momentum * 1.2])) #the smallest TOF value for a proton, considered to be the TOF at 120% of the theoretical proton momentum
            _, _, deuteron_tof_cut_array, _, _ =self.give_theoretical_TOF("Deuteron", np.array([self.run_momentum * 1.2])) #the smallest TOF value for a deuteron, considered to be the TOF at 120% of the theoretical proton momentum

            self.proton_tof_cut = proton_tof_cut_array[0]
            self.deuteron_tof_cut = deuteron_tof_cut_array[0]
        
        
        print(f"At {self.run_momentum} MeV/c, the T0-T1 TOF limit used to identify protons is {self.proton_tof_cut:.2f}ns, and the limits to ID deuterons is {self.deuteron_tof_cut:.2f}ns.")
        
        
        
        nEvents = len(data["beamline_pmt_qdc_ids"])
        
        #Read all the entries
        act0_l, act1_l, act2_l, act3_l, act4_l, act5_l =  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float)
        act0_r, act1_r, act2_r, act3_r, act4_r, act5_r =  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float)

        charge_t0_0, charge_t0_1, charge_t0_2, charge_t0_3, charge_t1_0, charge_t1_1 =  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float)
        charge_t1_2, charge_t1_3, charge_t4_0, charge_t4_1 =  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float)
        charge_t5_0, charge_t5_1, charge_t5_2, charge_t5_3, charge_t5_4, charge_t5_5, charge_t5_6, charge_t5_7 =  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float)
        charge_t5_8, charge_t5_9, charge_t5_10, charge_t5_11, charge_t5_12, charge_t5_13, charge_t5_14, charge_t5_15 =  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float)

        total_TOF_charge =  np.full(nEvents, np.nan, dtype=float)

        act0_time_l, act0_time_r =  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float)
        mu_tag_l, mu_tag_r =  np.full(nEvents, np.nan, dtype=float),  np.full(nEvents, np.nan, dtype=float)

        is_kept = np.full(nEvents, np.nan, dtype=float)
        is_kept_event_id =  np.full(nEvents, np.nan, dtype=float) #keeping track of which events we want to keep

        #also save the time of flight information
        t0_avgs  =  np.full(nEvents, np.nan, dtype=float)
        t1_avgs  =  np.full(nEvents, np.nan, dtype=float)
        t4_avgs  =  np.full(nEvents, np.nan, dtype=float)
        t0_avgs_second_hit  =  np.full(nEvents, np.nan, dtype=float)
        t1_avgs_second_hit  =  np.full(nEvents, np.nan, dtype=float)
        t4_avgs_second_hit  =  np.full(nEvents, np.nan, dtype=float)
        t5_avgs  =  np.full(nEvents, np.nan, dtype=float)
        tof_vals =  np.full(nEvents, np.nan, dtype=float)
        t4_l_array =  np.full(nEvents, np.nan, dtype=float)
        t4_r_array =  np.full(nEvents, np.nan, dtype=float)
        t4_l_array_second_hit =  np.full(nEvents, np.nan, dtype=float)
        t4_r_array_second_hit =  np.full(nEvents, np.nan, dtype=float)
        tof_t0t4_vals =  np.full(nEvents, np.nan, dtype=float)
        tof_t4t1_vals =  np.full(nEvents, np.nan, dtype=float)
        tof_t0t5_vals =  np.full(nEvents, np.nan, dtype=float)
        tof_t1t5_vals =  np.full(nEvents, np.nan, dtype=float)
        tof_t4t5_vals =  np.full(nEvents, np.nan, dtype=float)
        
        act_eveto_sums =  np.full(nEvents, np.nan, dtype=float)
        act_tagger_sums =  np.full(nEvents, np.nan, dtype=float)
        event_id =  np.full(nEvents, np.nan, dtype=float)
        ref0_times =  np.full(nEvents, np.nan, dtype=float)
        ref1_times =  np.full(nEvents, np.nan, dtype=float)
        evt_quality_bitmask =  np.full(nEvents, np.nan, dtype=float)
        digi_issues_bitmask =  np.full(nEvents, np.nan, dtype=float)
        
        #Also save the spill number for that event
        spill_number =  np.full(nEvents, np.nan, dtype=float)
        
        #save the reference bitmap for event quality flags:
        self.reference_flag_map = {
            "event_q_t0_or_t1_missing_tdc": 0,
            "event_q_t4_missing_tdc": 1,
            "event_q_t5_missing_tdc": 2,
            "event_q_hc_hit": 3,
            "event_q_t4_missing_qdc": 4,
            "t5_more_than_one_hit": 5
        }
        
        self.digitisation_issues_flag_map = {
            "qdc_failure": 0,
            "missing_digitiser_times": 1,
        }
        
        
        
        #Modify the code to implement Bruno's T5 cut and general improvement to the event quality cuts, also adding comments
        #default dict is similar to a normal dict with some additional features
        channel_clean_event_counts = defaultdict(int)
        
        #This dict counts the amount of times the tdc/qdc failed 
        tdc_qdc_failure_counts = {
            "qdc_failure": 0,
            "events_skipped": 0,
        }
        
        #the list of required signals to pass the event selection code
        required_groups = [
            {"name": "t0_group", "channels": t0_group, "mode": "all", "check_qdc": True},
            {"name": "t1_group", "channels": t1_group, "mode": "all", "check_qdc": True},
            {"name": "t4_group", "channels": t4_group, "mode": "all", "check_qdc": True},
        ]
        
        #always require T5
        required_groups.append(
            {
                "name": "t5_group",
                "channels": t5_total_group,
                "mode": "any_pair",
                "check_qdc": False,
            }
        )
            
#         #make a list of teh groups that are necessary to have qdc for
#         qdc_groups = [grp for grp in required_groups if grp["check_qdc"]]
#         #initialise the count of the number of requirements that have failed
#         group_requirement_failures = {grp["name"]: 0 for grp in required_groups}
#         #initialise the count of the number of PMT groups that have failed
#         qdc_failure_group_counts = {grp["name"]: 0 for grp in qdc_groups}
#         #Store the number of T4 qdc fails and HC failures
#         t4_qdc_failure_count = 0
#         hc_threshold_failure_count = 0
        
#         #Initialise the count the number of hits in each of the T5 pairs 
#         t5_hit_counts = np.zeros(len(t5_total_group), dtype=int) 
        
        #initialise a list that will hold which T5 group(s) are being hit
        t5_bar_multiplicity = [] #if require_t5 else None
        
        #Initialise the progress bar
        nEvents = len(data[branches[0]])
        
        pbar = tqdm(total=nEvents, desc="Reading in events", unit="evt")
        
        t4_qdc_samples = {ch: [] for ch in t4_group}
        


        for evt_idx in range(nEvents):
            if evt_idx % 100 == 0:
                pbar.update(100)
            
            keep_event = True
            
            #convert the info stored in qdc_ids and charge to be at least 1D (converting scalars to 1D lists)
            qdc_ids_evt = np.atleast_1d(data["beamline_pmt_qdc_ids"][evt_idx])
            qdc_vals_evt = np.atleast_1d(data["beamline_pmt_qdc_charges"][evt_idx])
            
            #if this event does not have any qdc entry, save that
            event_q_no_qdc_entry = (qdc_ids_evt.size == 0)
                 
                
#             #store the qdc charges for each of the T4 PMTs for that event as a dict  
#             event_t4_values = {}
            
            
#             for ch_val, charge in zip(qdc_ids_evt, qdc_vals_evt):
#                 ch = int(ch_val)
                
#                 if ch in t4_qdc_samples and ch in event_t4_values:
#                     previous = event_t4_values[ch]
#                     print(f"We have another T4 entry for channel {ch}, the previous value was: {previous}, this values is {float(charge)}")
                
#                 if ch in t4_qdc_samples and ch not in event_t4_values:
#                     val = float(charge)
#                     t4_qdc_samples[ch].append(val)
#                     event_t4_values[ch] = val
                #testing, what happens if we have multiple entries corresponding to t
               
                    
             #below, here is where Bruno does his T4 multiple particle check. waiting for that to be finalised before including in the main code. I'd recommend first reading in the file and then making the T4 analysis and then applying an additional cut based on that before the rest of the analysis continues. 
            
            #read in the data
            tdc_times = data["beamline_pmt_tdc_times"][evt_idx]
            tdc_ids   = data["beamline_pmt_tdc_ids"][evt_idx]
            qdc_charges = data["beamline_pmt_qdc_charges"][evt_idx]
            qdc_ids     = data["beamline_pmt_qdc_ids"][evt_idx]
            
#             #clean up the data to only include the first TDC hit 
#             if first_tdc_only:
#                 tdc_ids, tdc_times, duplicates_removed = _deduplicate_tdc_hits(tdc_ids, tdc_times)
#                 if duplicates_removed:
#                     for ch in duplicates_removed:
#                         channel_clean_event_counts[ch] += 1
                        
           

#             for idx_t5, pair in enumerate(t5_total_group):
#                 #here we are counting the number of times that the pair is in the tdc_ids 
#                 t5_hit_counts[idx_t5] += int(np.count_nonzero(np.isin(tdc_ids, pair)))
                    
                    
                    
#             #removes the duplicates of the tdc and qdc entries 
#             tdc_set_full = set(int(x) for x in np.atleast_1d(tdc_ids))
#             qdc_set_full = set(int(x) for x in np.atleast_1d(qdc_ids))
            
            
            #here we are checking if we are passing the requirements about the number of hits in each TS counter. 
#             requirement_failed = False
#             for group in required_groups:
#                 if not _tdc_requirement_met(group, tdc_set_full):
#                     group_requirement_failures[group["name"]] += 1
#                     requirement_failed = True
# #             if requirement_failed:
# #                 print(f"The requirement of the TS hits is not passed for event {evt_idx}")
# #                 continue
                
                
#             #check if we have qdc failures in any of the T0, T1, T4
#             qdc_failure_this_event = False
#             for group in qdc_groups:
#                 group_name = group["name"]
#                 group_channels = group["channels"]
#                 #if we have a tdc entry but not a qdc one then we have an issue
#                 missing_qdc_given_tdc = any(ch in tdc_set_full and ch not in qdc_set_full for ch in group_channels)
#                 if missing_qdc_given_tdc:
#                     #then we count the number of qdc fails for that specific channel
#                     qdc_failure_group_counts[group_name] += 1
#                     qdc_failure_this_event = True
                    
                    
#Not summing this up as it slows downs the code a lot
#             if qdc_failure_this_event:
#                 #store the total number of events that fail the qdc count
#                 tdc_qdc_failure_counts["qdc_failure"] += 1
#                 if enforce_tdc_qdc_match:
#                     #if we require a match then skip the event
#                     tdc_qdc_failure_counts["events_skipped"] += 1
#                     #continue

            MAX_CH = 128  #modify the dictionary based data hadnling to array

            corrected = np.full(MAX_CH, np.nan)
            corrected_second_hit = np.full(MAX_CH, np.nan)
            qdc_vals  = np.full(MAX_CH, np.nan)
            pe_vals   = np.full(MAX_CH, np.nan)

            
            # reference-time subtraction & first-hit only
            mask0 = (tdc_ids == reference_ids[0])
            mask1 = (tdc_ids == reference_ids[1])
            missing_digitiser_times = False
            if not mask0.any() or not mask1.any():
                keep_event = False
                ref0 = 0
                ref1 = 0
                missing_digitiser_times = True
            else:
                ref0 = tdc_times[mask0][0]
                ref1 = tdc_times[mask1][0]
                
            ref0_times[evt_idx] = ref0
            ref1_times[evt_idx] = ref1


            for ch, t in zip(tdc_ids, tdc_times):
                #do not store the information for the reference PMT and do not add the time if we already have an entry for that specific channel (That should be taking care of the case where more than one TDC is recorded) 
#                 if ch in reference_ids or ch in corrected:
                if ch in reference_ids:
#                     if ch in corrected:
#                         t_corr = t - (ref0 if ch < reference_ids[0] else ref1)
#                         print(f"We have more than one TDC entry for channel {ch}, the one stored is {corrected[ch]}, the new one would be {t_corr}, we are not keeping it but make sure that the following quantity is positive: {t_corr - corrected[ch]}")
                    continue
                if not np.isnan(corrected[ch]) and np.isnan(corrected_second_hit[ch]):
                    #if we already have a hit in the corrected time 
                    reference_time = ref0 if ch <= reference_ids[0] else ref1
                    corrected_second_hit[ch] = t - reference_time
                    continue
        
                reference_time = ref0 if ch <= reference_ids[0] else ref1
                corrected[ch] = t - reference_time
            
#             qdc_dict = {}
#             tdc_dict = {}
#             pe_dict = {}
            for ch, q in zip(qdc_ids, qdc_charges):
                if np.isnan(qdc_vals[ch]):
                    qdc_vals[ch] = q

                    if (ch in act_eveto_group) or (ch in act_tagger_group):  
                        #use mean gain and mean pedestal, find it in the calibration data base
#                         calib_index = calibration["channel_id"].index(ch)
#                         gain = calibration["gain_value"][calib_index]
#                         pedestal = calibration["pedestal_mean"][calib_index]
                        gain, pedestal = calib_map[ch]
                        pe_vals[ch] = (q-pedestal)/gain 
                    if (ch in t0_group) or (ch in t1_group) or (ch in t4_group):
                        pe_vals[ch] = q 

            
            event_q_t0_or_t1_missing_tdc = False
            # require all channels on T0/T1 before computing averages
#             if not all(ch in corrected for ch in t0_group+t1_group):
            if not all(not np.isnan(corrected[ch]) for ch in t0_group+t1_group):
                keep_event = False
                event_q_t0_or_t1_missing_tdc = True
                t0 = None
                t1 = None
                t0_second_hit = None
                t1_second_hit = None
                
            else:
#                 t0 = np.mean([corrected[ch] for ch in t0_group])
                vals = [corrected[ch] for ch in t0_group]
                t0 = sum(vals) / len(vals)
#                 t1 = np.mean([corrected[ch] for ch in t1_group])
                vals = [corrected[ch] for ch in t1_group]
                t1 = sum(vals) / len(vals)
        
                vals = [corrected_second_hit[ch] for ch in t0_group]
                t0_second_hit = sum(vals) / len(vals)
                
                vals = [corrected_second_hit[ch] for ch in t1_group]
                t1_second_hit = sum(vals) / len(vals)
                

           
            #require already that there is a hit in all t4 PMTs
            #and otherwise that both hits are above threshold
            event_q_t4_missing_tdc = False
            event_q_t4_below_thres = False
            if not all(not np.isnan(corrected[ch]) for ch in t4_group):
                keep_event = False
                event_q_t4_missing_tdc = True
                t4 = None
                t4_l = None
                t4_r = None
                
                t4_second_hit = None
                t4_l_second_hit = None 
                t4_r_second_hit = None
                
            else:
                vals = [corrected[ch] for ch in t4_group]
                t4 = sum(vals) / len(vals)
                t4_l = corrected[t4_group[0]] #if not np.isnan(corrected[t4_group[0]]) else None
                t4_r = corrected[t4_group[1]] #if not np.isnan(corrected[t4_group[1]]) else None
                
                vals = [corrected_second_hit[ch] for ch in t4_group]
                t4_second_hit = sum(vals) / len(vals)
                
                t4_l_second_hit = corrected_second_hit[t4_group[0]] #if not np.isnan(corrected[t4_group[0]]) else None
                t4_r_second_hit = corrected_second_hit[t4_group[1]]
                
                
            event_q_t4_missing_qdc = False
#             if not all(ch in qdc_dict for ch in t4_group):
            if not all(not np.isnan(qdc_vals[ch]) for ch in t4_group):
                keep_event = False
                event_q_t4_missing_qdc = True
            


            #compute the T5 mean time based on the bar that we have selected
            t5_bar_means = []
            for ch0, ch1 in t5_total_group:
                v0 = corrected[ch0]
                v1 = corrected[ch1]
                #checking if we have a second hit in the T5
                w0 = corrected_second_hit[ch0]
                w1 = corrected_second_hit[ch1]

                if not np.isnan(v0) and not np.isnan(v1):
                    t5_bar_means.append(0.5 * (v0 + v1))
                if not np.isnan(w0) and not np.isnan(w1):
                    t5_bar_means.append(0.5 * w0 + w1)
                 
            #in case there is no pairs of t5 hits 
            event_q_t5_missing_tdc = (len(t5_bar_means) == 0)
            # instead of using the mean, take the erliest hit. 
            t5_earliest_time = min(t5_bar_means) if t5_bar_means else None
            t5_more_than_one_hit = (len(t5_bar_means)>1)
            #the multiplicity shows how many bars are being hit, this could be a useful way to identify multiple particle events
#             if t5_bar_multiplicity is not None:
            t5_bar_multiplicity.append(len(t5_bar_means))
                

            #--------- HC cut ----------
            event_q_hc_hit = False
#             if (any(qdc_dict.get(ch, 0) >= hc_charge_cut for ch in hc_group)):
#                 keep_event = False  # if either HC channel fired with charge ≥ threshold, skip event
#                 event_q_hc_hit = True
            for ch in hc_group:
                if qdc_vals[ch] >= hc_charge_cut:
                    keep_event = False
                    event_q_hc_hit = True

            #Keep all of the entries but then df is only the ones that we keep 
            t0_avgs[evt_idx] = t0
            t1_avgs[evt_idx] = t1
            t4_avgs[evt_idx] = t4
            
            t0_avgs_second_hit[evt_idx] = t0_second_hit
            t1_avgs_second_hit[evt_idx] = t1_second_hit
            t4_avgs_second_hit[evt_idx] = t4_second_hit
            
            
            t4_l_array[evt_idx] = t4_l
            t4_r_array[evt_idx] = t4_r
            
            t4_l_array_second_hit[evt_idx] = t4_l_second_hit
            t4_r_array_second_hit[evt_idx] = t4_r_second_hit
            
            t5_avgs[evt_idx] = t5_earliest_time
            
            
            tof        = None
            tof_t0t5   = None
            tof_t1t5   = None
            tof_t0t4   = None
            tof_t4t1   = None
            tof_t4t5   = None

            if not event_q_t0_or_t1_missing_tdc:
                tof = t1 - t0

                if t5_earliest_time is not None:
                    tof_t0t5 = t5_earliest_time - t0
                    tof_t1t5 = t5_earliest_time - t1

                if not event_q_t4_missing_tdc:
                    tof_t0t4 = t4 - t0
                    tof_t4t1 = t1 - t4
                    tof_t4t5 = (
                        t5_earliest_time - t1
                        if t5_earliest_time is not None
                        else None
                    )

            tof_vals[evt_idx] = tof
            tof_t0t5_vals[evt_idx] = tof_t0t5
            tof_t1t5_vals[evt_idx] = tof_t1t5
            tof_t0t4_vals[evt_idx] = tof_t0t4
            tof_t4t1_vals[evt_idx] = tof_t4t1
            tof_t4t5_vals[evt_idx] = tof_t4t5
                

            #svae the charge 
            act0_l[evt_idx] = pe_vals[12]
            act0_r[evt_idx] = pe_vals[13]
            act1_l[evt_idx] = pe_vals[14]
            act1_r[evt_idx] = pe_vals[15]
            act2_l[evt_idx] = pe_vals[16]
            act2_r[evt_idx] = pe_vals[17]
            act3_l[evt_idx] = pe_vals[18]
            act3_r[evt_idx] = pe_vals[19]
            act4_l[evt_idx] = pe_vals[20]
            act4_r[evt_idx] = pe_vals[21]
            act5_l[evt_idx] = pe_vals[22]
            act5_r[evt_idx] = pe_vals[23]

            #add the charge for the T0, T1 and T4 PMTs 
            charge_t0_0[evt_idx] = pe_vals[0]
            charge_t0_1[evt_idx] = pe_vals[1]
            charge_t0_2[evt_idx] = pe_vals[2]
            charge_t0_3[evt_idx] = pe_vals[3]
            charge_t1_0[evt_idx] = pe_vals[4]
            charge_t1_1[evt_idx] = pe_vals[5]   
            charge_t1_2[evt_idx] = pe_vals[6]
            charge_t1_3[evt_idx] = pe_vals[7]
            charge_t4_0[evt_idx] = pe_vals[42]
            charge_t4_1[evt_idx] = pe_vals[43]
            charge_t5_0[evt_idx] = pe_vals[48]
            charge_t5_1[evt_idx] = pe_vals[49]
            charge_t5_2[evt_idx] = pe_vals[50]
            charge_t5_3[evt_idx] = pe_vals[51]
            charge_t5_4[evt_idx] = pe_vals[52]
            charge_t5_5[evt_idx] = pe_vals[53]
            charge_t5_6[evt_idx] = pe_vals[54]
            charge_t5_7[evt_idx] = pe_vals[55]
            charge_t5_8[evt_idx] = pe_vals[56]
            charge_t5_9[evt_idx] = pe_vals[57]
            charge_t5_10[evt_idx] = pe_vals[58]
            charge_t5_11[evt_idx] = pe_vals[59]
            charge_t5_12[evt_idx] = pe_vals[60]
            charge_t5_13[evt_idx] = pe_vals[61]
            charge_t5_14[evt_idx] = pe_vals[62]
            charge_t5_15[evt_idx] = pe_vals[63]


            
            spill_id = data["spill_counter"][evt_idx]
            spill_number[evt_idx] = spill_id

#             total_TOF_charge.append(qdc_dict.get(48, 0)+ qdc_dict.get(49, 0)+ qdc_dict.get(50, 0)+ qdc_dict.get(51, 0)+ qdc_dict.get(52, 0)+ qdc_dict.get(53, 0)+ qdc_dict.get(54, 0)+ qdc_dict.get(55, 0)+ qdc_dict.get(56, 0)+ qdc_dict.get(57, 0)+ qdc_dict.get(58, 0)+ qdc_dict.get(59, 0)+ qdc_dict.get(60, 0)+ qdc_dict.get(61, 0)+ qdc_dict.get(62, 0)+ qdc_dict.get(63, 0))
            
            event_id[evt_idx] = evt_idx


            mu_tag_l[evt_idx] = qdc_vals[24]        
            mu_tag_r[evt_idx] = qdc_vals[25]

            act0_time_l[evt_idx] = corrected[12]
            act0_time_r[evt_idx] = corrected[13]
                
            is_kept[evt_idx] = keep_event

            flags = {
                "event_q_t0_or_t1_missing_tdc": event_q_t0_or_t1_missing_tdc,
                "event_q_t4_missing_tdc": event_q_t4_missing_tdc,
                "event_q_t5_missing_tdc": event_q_t5_missing_tdc,
                "event_q_hc_hit": event_q_hc_hit,
                "event_q_t4_missing_qdc": event_q_t4_missing_qdc,
                "t5_more_than_one_hit": t5_more_than_one_hit
            }
            
            bitmask = write_event_quality_mask(flags, self.reference_flag_map)
            evt_quality_bitmask[evt_idx] = bitmask
            
            flags_digi = {
                "qdc_failure": event_q_no_qdc_entry,
                "missing_digitiser_times": missing_digitiser_times,
               
            }
            
            #the bitmask that handles issues related to digitisation 
            bitmask_digitisation = write_event_quality_mask(flags_digi, self.digitisation_issues_flag_map)
            digi_issues_bitmask[evt_idx] = bitmask_digitisation 
            
            
            
            #here we are obtaining the bitmask corresponding to this specific set of flags 

            #End of the loop over events  
        act_arrays = [act0_l, act1_l, act2_l, act3_l, act4_l, act5_l,
                      act0_r, act1_r, act2_r, act3_r, act4_r, act5_r]

        act_arrays = [np.array(arr, dtype=float) for arr in act_arrays]

        act0_time_l = np.array(act0_time_l)
        act0_time_r = np.array(act0_time_r)

        #store them back in their initial name
        (act0_l, act1_l, act2_l, act3_l, act4_l, act5_l,
         act0_r, act1_r, act2_r, act3_r, act4_r, act5_r) = act_arrays

        
        mu_tag_l, mu_tag_r = np.array(mu_tag_l), np.array(mu_tag_r) 
        
        is_kept = np.array(is_kept)

        
        data_dict = {
            "t0_time": t0_avgs,
            "t1_time": t1_avgs,
            "t4_time": t4_avgs,
            "t0_time_second_hit": t0_avgs_second_hit,
            "t1_time_second_hit": t1_avgs_second_hit,
            "t4_time_second_hit": t4_avgs_second_hit,
            "t5_time": t5_avgs,
            "t4_l": t4_l_array,
            "t4_r": t4_r_array,
            "t4_l_second_hit": t4_l_array_second_hit,
            "t4_r_second_hit": t4_r_array_second_hit,
            "act0_l": act0_l,
            "act1_l": act1_l,
            "act2_l": act2_l,
            "act3_l": act3_l,
            "act4_l": act4_l,
            "act5_l": act5_l,
            "act0_r": act0_r,
            "act1_r": act1_r,
            "act2_r": act2_r,
            "act3_r": act3_r,
            "act4_r": act4_r,
            "act5_r": act5_r,
            "charge_t0_0": np.array(charge_t0_0),
            "charge_t0_1": np.array(charge_t0_1),
            "charge_t0_2": np.array(charge_t0_2),
            "charge_t0_3": np.array(charge_t0_3),
            "charge_t1_0": np.array(charge_t1_0),
            "charge_t1_1": np.array(charge_t1_1),
            "charge_t1_2": np.array(charge_t1_2),
            "charge_t1_3": np.array(charge_t1_3),
            "charge_t4_0": np.array(charge_t4_0),
            "charge_t4_1": np.array(charge_t4_1),
            "charge_t5_0": np.array(charge_t5_0),
            "charge_t5_1": np.array(charge_t5_1),
            "charge_t5_2": np.array(charge_t5_2),
            "charge_t5_3": np.array(charge_t5_3),
            "charge_t5_4": np.array(charge_t5_4),
            "charge_t5_5": np.array(charge_t5_5),
            "charge_t5_6": np.array(charge_t5_6),
            "charge_t5_7": np.array(charge_t5_7),
            "charge_t5_8": np.array(charge_t5_8),
            "charge_t5_9": np.array(charge_t5_9),
            "charge_t5_10": np.array(charge_t5_10),
            "charge_t5_11": np.array(charge_t5_11),
            "charge_t5_12": np.array(charge_t5_12),
            "charge_t5_13": np.array(charge_t5_13),
            "charge_t5_14": np.array(charge_t5_14),
            "charge_t5_15": np.array(charge_t5_15),
            "event_id":event_id,
#             "total_TOF_charge":total_TOF_charge,
            "act0_time_l": act0_time_l,
            "act0_time_r": act0_time_r,
            "tof": tof_vals,
            "tof_t0t4": tof_t0t4_vals,
            "tof_t4t1": tof_t4t1_vals,
            "tof_t0t5": tof_t0t5_vals,
            "tof_t1t5": tof_t1t5_vals,
            "tof_t4t5": tof_t4t5_vals,
            "mu_tag_l": mu_tag_l,
            "mu_tag_r": mu_tag_r,
            "is_kept": is_kept,
            "ref0_time":ref0_times,
            "ref1_time":ref1_times,
            "spill_number":spill_number,
            "evt_quality_bitmask":evt_quality_bitmask,
            "digi_issues_bitmask":digi_issues_bitmask
          
        }
        
        # create DataFrame, much more robust than having many arrays 
        self.df_all = pd.DataFrame(data_dict)

        
        #add the combined branches that can be useful
        self.df_all["mu_tag_total"] = self.df_all["mu_tag_l"] + self.df_all["mu_tag_r"]
        
        
        
        
        
        if self.there_is_ACT5:
            self.PMT_list = ["act0_l", "act0_r", "act1_l",  "act1_r", "act2_l", "act2_r", "act3_l", "act3_r", "act4_l", "act4_r", "act5_l", "act5_r"]
           
            
        else:
            self.PMT_list = ["act0_l", "act0_r", "act1_l",  "act1_r", "act2_l", "act2_r", "act3_l", "act3_r", "act4_l", "act4_r"]
            
        
        #here make the subset sample that we are keeping for analysis:
        if self.require_t5_hit:
            self.df = self.df_all[self.df_all["evt_quality_bitmask"] == 0].copy()
        else:
            self.df = self.df_all[(self.df_all["evt_quality_bitmask"] == 0) | (self.df_all["evt_quality_bitmask"] == 4)].copy()
            
            
        print(f"\n \n When T5 requirement is {self.require_t5_hit} the fraction of events kept for analysis is {len(self.df)/len(self.df_all) * 100}% \n \n")
        
        #this will be necessary for identifying events later
        self.is_kept = is_kept
        
        #store the id of the events of interest
        self.is_kept_event_id = is_kept_event_id                          
        pbar.close()
        
#         print(f"Total weight of self.df: {self.df.memory_usage(deep=True).sum()/ (1024**2):.2f}Mb")
        
#         summary_groups = ", ".join(grp["name"] for grp in required_groups)
#         print(f"Required channel presence summary ({summary_groups}) — TDC only:")
#         if any(group_requirement_failures.values()):
#             for group in required_groups:
#                 group_name = group["name"]
#                 failures = group_requirement_failures[group_name]
#                 percent = (failures / nEvents * 100) if nEvents else 0
#                 print(f"  {group_name}: skipped {failures}/{nEvents} events ({percent:.2f}%) due to missing TDC IDs")
#         else:
#             print("  No events skipped for missing required TDC channels.")

#         print("QDC failure summary (TDC present, missing QDC):")
#         if qdc_failure_group_counts and any(qdc_failure_group_counts.values()):
#             for group in qdc_groups:
#                 group_name = group["name"]
#                 failures = qdc_failure_group_counts[group_name]
#                 percent = (failures / nEvents * 100) if nEvents else 0
#                 print(f"  {group_name}: {failures}/{nEvents} events ({percent:.2f}%) had TDC with no QDC match")
#             total_failures = tdc_qdc_failure_counts["qdc_failure"]
#             print(f"  Total events with QDC failures: {total_failures}/{nEvents}")
#         else:
#             print("  No QDC failures detected in required channels.")

#         print("T4 QDC threshold summary:")
#         if t4_qdc_failure_count:
#             percent = (t4_qdc_failure_count / nEvents * 100) if nEvents else 0
#             print(f"  Failed T4 QDC cut: {t4_qdc_failure_count}/{nEvents} events ({percent:.2f}%) removed.")
#         else:
#             print("  All events passed the T4 QDC cut.")

#         print("HC charge veto summary:")
#         if hc_threshold_failure_count:
#             percent = (hc_threshold_failure_count / nEvents * 100) if nEvents else 0
#             print(f"  Failed HC charge cut: {hc_threshold_failure_count}/{nEvents} events ({percent:.2f}%) removed.")
#         else:
#             print("  All events passed the HC charge cut.")

#         if first_tdc_only:
#             print("TDC duplicate-removal summary:")
#             if channel_clean_event_counts:
#                 for ch in sorted(channel_clean_event_counts):
#                     events_cleaned = channel_clean_event_counts[ch]
#                     percent = (events_cleaned / nEvents * 100) if nEvents else 0
#                     print(f"  Channel {ch}: {events_cleaned}/{nEvents} events ({percent:.2f}%) had repeated TDC IDs cleaned")
#             else:
#                 print("No duplicate TDC hits detected.")
        
        
    def adjust_1pe_calibration(self):

        bins = np.linspace(-1,3, 100)
        
        self.PMT_value_ped = []
        self.PMT_1pe_scale = []
        for PMT in self.PMT_list:
            fig, ax = plt.subplots(figsize = (8, 6))    
            h, _, _ = ax.hist(self.df[PMT], bins = bins, histtype = "step", label = "Config-file 1pe calib")

            index_ped = np.argmax(h)
            value_ped = 0.5 * (bins[index_ped] + bins[index_ped+1])
            #actually change the array: pedestal shifted: can do as many times as we want, will just substract 0 all the n>1 times we do it
            self.df[PMT] -= value_ped 
            self.df_all[PMT] -= value_ped 
#             self.PMT_value_ped.append(value_ped)
            
            h, _, _ = ax.hist(self.df[PMT], bins = bins, histtype = "step", label = "+ pedestal shift")
            #plot the pedestal and 1 pe peak, for now reading from the pre-made calibration, eventually, will have to be made run 
            #check the position of the 1pe peak 
            #restict the portion to fit
            xmin = 0.5
            xmax = 1.5
            bin_centers = (bins[:-1] + bins[1:]) / 2
            is_onepe_region = (bin_centers >= xmin) & (bin_centers <= xmax)
            bins_onepe = bin_centers[is_onepe_region]
            entries_onepe = h[is_onepe_region]

            #fit a gaussian 
            try:
                popt, pcov = fit_gaussian(entries_onepe, bins_onepe)
            except:
                print("We did not manage to fit the 1pe peak, we are therefore not correcting its position")
                popt = np.array([0, 1, 0])
            #ax.plot(bins_onepe, gaussian(bins_onepe, *popt), "k--", label = f"Gaussian fit to 1pe peak:\nMean: {popt[1]:.2f} PE, std: {popt[2]:.2f} PE")

            self.df[PMT] /= popt[1] #actually change the array: pedestal shifted: can do as many times as we want, will just substract 0 all the n>1 times we do it
            self.df_all[PMT] /= popt[1] #actually change the array: pedestal shifted: can do as many times as we want, will just substract 0 all the n>1 times we do it
            
            
            self.PMT_1pe_scale.append(popt[1])
            
            h, _, _ = ax.hist(self.df[PMT], bins = bins, histtype = "step", label = "+ charge scale")
            entries_onepe = h[is_onepe_region]
            try:
                popt, pcov = fit_gaussian(entries_onepe, bins_onepe)
                ax.plot(bins_onepe, gaussian(bins_onepe, *popt), "k--", label = f"Gaussian fit to 1pe peak:\nMean: {popt[1]:.2f} PE, std: {popt[2]:.2f} PE")
            except:
                #in case we do not manage to fit the 1pe position, still do not crash the whole thing
                print("We did not manage to fit the 1pe peak")
            

            #now that we have the correct 1pe calibration, make the combination of ACT charges 
            self.df["act_eveto"] = self.df["act0_l"]+self.df["act0_r"]+self.df["act1_l"]+self.df["act1_r"]+self.df["act2_l"]+self.df["act2_r"]
            
            if self.there_is_ACT5:
                self.df["act_tagger"] = self.df["act3_l"]+self.df["act3_r"]+self.df["act4_l"]+self.df["act4_r"]+self.df["act5_l"]+self.df["act5_r"]
            
            else:
                self.df["act_tagger"] = self.df["act3_l"]+self.df["act3_r"]+self.df["act4_l"]+self.df["act4_r"]

            
            
            ax.set_yscale("log")
            ax.legend(fontsize = 16)
            ax.set_xlabel("Charge collected (PE)", fontsize = 18)
            ax.set_ylabel("Number of entries", fontsize = 18)
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - {PMT}", fontsize = 18)
            ax.grid()
            self.pdf_global.savefig(fig)
            plt.close()
            
        print("One PE calibration finished, please don't forget to check that it is correct")
        
        
        
    def tag_electrons_ACT02(self, tightening_factor = 0):
        '''Tagging the electrons based on the charge deposited in the upstream ACTs, add an additional scale factor to tighten the cut some more '''
        bins = np.linspace(0, 40, 200)
        fig, ax = plt.subplots(figsize = (8, 6))    
        h, _, _ = ax.hist(self.df["act_eveto"], bins = bins, histtype = "step")
        

        #automatically find the middle of the least populated bin within 4 and 15 PE
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        x_min = 0.2
        x_max = 23
        mask = (bin_centers >= x_min) & (bin_centers <= x_max)
        # Find the bin index with the minimum count in that range
        min_index = np.argmin(h[mask])

        # Get the actual bin index in the original array
        index = np.where(mask)[0]
        self.eveto_cut = bin_centers[index[min_index]]

        ax.axvline(self.eveto_cut, linestyle = '--', color = 'black', label = f'Optimal electron rejection veto: {self.eveto_cut:.1f} PE')
        
        if tightening_factor!=0:
            self.eveto_cut = self.eveto_cut * (1 - tightening_factor/100)
            ax.axvline(self.eveto_cut, linestyle = '--', color = 'red', label = f'with tightening factor ({tightening_factor}%): {self.eveto_cut:.1f} PE')
        ax.set_yscale("log")
        ax.set_xlabel("ACT0-2 total charge (PE)", fontsize = 18)
        ax.set_ylabel("Number of entries", fontsize = 18)
        ax.legend(fontsize = 16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)- ACT0-2", fontsize = 20)
#         pdf.savefig()
        self.pdf_global.savefig(fig)
        plt.close()
        
        #make sure that the particle is not already a slow travelling particle, proton or deuterium
        if self.run_momentum > 350:
            self.df["is_electron"] = np.where(self.df["act_eveto"]>self.eveto_cut, (self.df["tof"]<self.proton_tof_cut), False)
        else:
            self.df["is_electron"] = (self.df["act_eveto"]>self.eveto_cut)
                                               
        n_electrons = sum(self.df["is_electron"])
        n_triggers = len(self.df["is_electron"])
        print(f"A total of {n_electrons} electrons are tagged with ACT02 out of {n_triggers}, i.e. {n_electrons/n_triggers * 100:.1f}% of the dataset")
        
        
    def tag_multiple_particle_events(self):
        '''Tagging the events with more than one particle, based on the presence of a higher charge than the Mip charge in T0, T1, T4 or T5.
        A diagnostic plot of the total T0 charge vs TOF is saved to the global PDF.'''
        
        # flag events with more than one hit
        self.df["multiple_particles"] = (
            self.df["t0_time_second_hit"].notna()
            | self.df["t1_time_second_hit"].notna()
            | self.df["t4_time_second_hit"].notna()
            | (self.df["t5_time"].isna())
        )

        n_multiple_particles = sum(self.df["multiple_particles"])
        n_triggers = len(self.df["multiple_particles"])
        print(
            f"A total of {n_multiple_particles} events are tagged as having multiple particles "
            f"based on the presence of multiple hits in T0, T1, T4 or T5, out of {n_triggers}, "
            f"i.e. {n_multiple_particles/n_triggers * 100:.1f}% of the dataset"
        )

        # diagnostic plot: total T0 charge (channels 0..3) vs TOF
        t0_charge = self.df["charge_t0_0"]+ self.df["charge_t0_1"]+ self.df["charge_t0_2"]+ self.df["charge_t0_3"]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        x_bins = np.linspace(10, 50, 100)
        y_bins = np.linspace(2000, 20000, 100)
        h = ax.hist2d(self.df["tof"],t0_charge, bins=(x_bins, y_bins), norm=LogNorm())
        
    
        ax.set_ylabel("Total T0 charge (QDC)", fontsize=18)
        ax.set_xlabel("TOF (ns)", fontsize=18)
        ax.set_title(
            f"Run {self.run_number} ({self.run_momentum} MeV/c) - T0 charge vs TOF",
            fontsize=20,
        )
        self.pdf_global.savefig(fig)
        plt.close()

        #for particle with TOF less than 17ns, plot a 1d histogram of the total T0 charge
        fig, ax = plt.subplots(figsize=(8, 6))  
        bins = np.linspace(10, 50, 100)
        h, _, _ = ax.hist(t0_charge[self.df["tof"]<17], bins = y_bins, histtype = "step")
        ax.set_xlabel("Total T0 charge (QDC)", fontsize=18)
        ax.set_title(
            f"Run {self.run_number} ({self.run_momentum} MeV/c) - Total T0 charge for particles with TOF < 17ns",
            fontsize=20,
        )
        ax.grid()
        self.pdf_global.savefig(fig)
        plt.close()

    def tag_electrons_ACT35(self, cut_line = 0):
        '''Tagging the electrons based on the charge deposited in the downstream ACTs, to 'clean up the edges'''
        
        n_electrons_initial = sum(self.df["is_electron"])
        
        #Here plot visually the electron cutline to check that it is correct
        self.plot_ACT35_left_vs_right(cut_line, "muon/electron")
        
        ### identify the particles above the cut as electrons
        #make sure that the particle is not already a slow travelling particle, proton or deuterium and that it stays an electron if it has already been identified by ACT02 but isn't above the cutline
        self.df["is_electron"] = np.where(self.df["act_tagger"]>cut_line, (self.df["tof"]<self.proton_tof_cut), self.df["is_electron"])
        
        self.act35_e_cut = cut_line
        
 
          
        
        n_electrons = sum(self.df["is_electron"])
        n_triggers = len(self.df["is_electron"])
        print(f"A total of {n_electrons-n_electrons_initial} additional electrons are tagged with ACT35, on top of the {n_electrons_initial} that were tagged with ACT02")
        
        
        
        
        
    def plot_ACT35_left_vs_right(self, cut_line = None, cut_line_label = "pion/muon"):
        bins = np.linspace(0, 70, 100)
        fig, ax = plt.subplots(figsize = (8, 6))
        act_tagger_l = self.df["act3_l"]+self.df["act4_l"]+self.df["act5_l"] * int(self.there_is_ACT5)
        act_tagger_r =self.df["act3_r"]+self.df["act4_r"]+self.df["act5_r"]* int(self.there_is_ACT5)
        
        h = ax.hist2d(act_tagger_l, act_tagger_r, bins = (bins, bins),norm=LogNorm())
        fig.colorbar(h[3], ax=ax)
        if cut_line != None:
                ax.plot(bins, cut_line - bins, "r--", label = f"{cut_line_label} cut line: ACT3-5 = {cut_line:.1f} PE")
                ax.legend(fontsize = 14)
        ax.set_xlabel("ACT3-5 left (PE)", fontsize = 18)
        ax.set_ylabel("ACT3-5 right (PE)", fontsize = 18)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)\nACT3-5 all particles", fontsize = 20)
        self.pdf_global.savefig(fig)

        plt.close()
        

        try:
#         if True:
        
            not_electrons = ~self.df["is_electron"]

            fig, ax = plt.subplots(figsize = (8, 6))
            h = ax.hist2d(act_tagger_l[not_electrons], act_tagger_r[not_electrons], bins = (bins, bins), norm=LogNorm())
            fig.colorbar(h[3], ax=ax)
            if cut_line != None:
                ax.plot(bins, cut_line - bins, "r--", label = f"{cut_line_label} cut line: ACT3-5 = {cut_line:.1f} PE")
                ax.legend(fontsize = 14)
            ax.set_xlabel("ACT3-5 left (PE)", fontsize = 18)
            ax.set_ylabel("ACT3-5 right (PE)", fontsize = 18)
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) \n ACT35 After eveto", fontsize = 20)
            self.pdf_global.savefig(fig)

            plt.close()
            print("ACT35 left vs right plots have been made please check that they are sensible, electrons should be in the top right corner")
            
        except:
            print("Please make the electron selection using tag_electrons_ACT02 before checking the ACT35 left vs right plot, otherwise you will have all of the entries")
            return 0
        
        try:
            not_protons = ~self.df["is_proton"]
            not_protons = not_protons&(~self.df["is_deuteron"])

            fig, ax = plt.subplots(figsize = (8, 6))
            h = ax.hist2d(act_tagger_l[not_electrons&not_protons], act_tagger_r[not_electrons&not_protons], bins = (bins, bins), norm=LogNorm())
            fig.colorbar(h[3], ax=ax)
            if cut_line != None:
                ax.plot(bins, cut_line - bins, "r--", label = f"{cut_line_label} cut line: ACT3-5 = {cut_line:.1f} PE")
                ax.legend(fontsize = 14)
            ax.set_xlabel("ACT3-5 left (PE)", fontsize = 18)
            ax.set_ylabel("ACT3-5 right (PE)", fontsize = 18)
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) \n ACT3-5 After eveto and p removal", fontsize = 20)
            self.pdf_global.savefig(fig)

            plt.close()
            print("ACT35 left vs right plots have been made please check that they are sensible, protons should be in the bottom left corner")
            
        except:
            print("Please make the proton selection using tag_protons_TOF before checking the ACT35 left vs right plot to get the most of out it")
            return 0
        
        
        
        
        
    def tag_protons_TOF(self):
        '''Simple identification of the protons based on the time of flight, cutline fixed at 17.5ns for now, can be modified later if needed'''
        
        
        if self.run_momentum < 0:
            self.df["is_proton"] = False
            self.df["is_deuteron"] = False
            
        else:
             self.df["is_proton"] = np.where(self.df["tof"]>self.proton_tof_cut, self.df["tof"]<helium3_tof_cut, False)      
             self.df["is_deuteron"] = np.where(self.df["tof"]>self.deuteron_tof_cut, self.df["tof"]<tritium_tof_cut, False)
             
            
            
        
        if self.run_momentum < 700:
            self.df["is_helium3"] = False
            self.df["is_tritium"] = False           
            self.df["is_lithium6"] = False
        else:
            
            self.df["is_helium3"] = np.where(self.df["tof"]>helium3_tof_cut, self.df["tof"]<self.deuteron_tof_cut, False)
            self.df["is_tritium"] = np.where(self.df["tof"]>tritium_tof_cut, self.df["tof"]<lithium6_tof_cut, False)
            self.df["is_lithium6"] = np.where(self.df["tof"]>lithium6_tof_cut, True, False)

                
                
        n_protons = sum(self.df["is_proton"]==True)
        n_deuteron = sum(self.df["is_deuteron"]==True)
        
        n_helium3 = sum(self.df["is_helium3"]==True)
        n_tritium = sum(self.df["is_tritium"]==True)
        n_lithium6 = sum(self.df["is_lithium6"]==True)
        
        
        n_triggers = len(self.df["is_proton"])
        
        print(f"A total of {n_protons} protons and {n_deuteron} deuterons nuclei are tagged using the TOF out of {n_triggers}, i.e. {n_protons/n_triggers * 100:.1f}% of the dataset are protons and {n_deuteron/n_triggers * 100:.1f}% are deuteron")
        
        
        print(f"A total of {n_helium3} helium3 nuclei, {n_tritium} tritium nuclei and {n_lithium6} lithium 6 nuclei are tagged using the TOF out of {n_triggers}, i.e. {n_helium3/n_triggers * 100:.2f}% of the dataset are helium3, {n_tritium/n_triggers * 100:.1f}% are tritium, {n_lithium6/n_triggers * 100:.2f} lithium 6 nuclei")
        
        
        
        
        
    def tag_muons_pions_ACT35(self):
        '''Function to identify the muons and pions based on the charge deposited in ACT35, potentially using the muon tagger'''
        
        #step 1: find the optimal cut line in the muon tagger and decide if it is useful to implement it (that is, in case there are still some non-electrons left after the cut
        bins = np.linspace(120, 800, 100)
           
        
        #make sure they are boolean first
        self.df["is_proton"] = self.df["is_proton"].astype(bool)
        self.df["is_electron"] = self.df["is_electron"].astype(bool)
        self.df["is_deuteron"] = self.df["is_deuteron"].astype(bool)
        self.df["is_helium3"] = self.df["is_helium3"].astype(bool)
        self.df["is_tritium"] = self.df["is_tritium"].astype(bool)
        self.df["is_lithium6"] = self.df["is_lithium6"].astype(bool)
        

        mu_tag_tot = self.df["mu_tag_l"]+self.df["mu_tag_r"]
        #cannot be any other particle already
        if self.run_momentum > 300:
            muons_pions = (self.df["tof"] < self.proton_tof_cut) & (~self.df["is_electron"]) 
        else:
            muons_pions = (~self.df["is_electron"]) 
        
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.hist(mu_tag_tot, bins = bins, label = 'All particles', histtype = "step")
        ax.hist(mu_tag_tot[self.df["is_electron"]], bins = bins, label = 'Electrons', histtype = "step")
        ax.hist(mu_tag_tot[self.df["is_proton"]], bins = bins, label = 'Protons', histtype = "step")
        h, _, _ = ax.hist(mu_tag_tot[muons_pions], bins = bins, label = 'Muons and pions', histtype = "step")
        ax.set_xlabel(f"Total charge in muon-tagger (QDC)", fontsize = 18)
        ax.set_ylabel("Number of events", fontsize = 18)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - Muon Tagger charge", fontsize = 20)


        #implement automatic muon tagger cut
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        x_min = 150
        x_max = 300
        mask = (bin_centers >= x_min) & (bin_centers <= x_max)
        # Find the bin index with the minimum count in that range
        min_index = np.argmin(h[mask])

        # Get the actual bin index in the original array
        index = np.where(mask)[0]
        mu_tag_cut = bin_centers[index[min_index]]
        
        #minimum fraction of muons and pions that have to be above the mu tag threshold (set to 0.5%)
        min_fraction_above_cut = 0.005
        
        n_electrons_above_cut = np.sum(self.df["is_electron"][mu_tag_tot>mu_tag_cut])
        n_muons_pions_above_cut = np.sum(muons_pions[mu_tag_tot>mu_tag_cut])
        self.n_muons_pions = np.sum(muons_pions)

        ax.axvline(mu_tag_cut, color = "k", linestyle = "--", label = f"Muon tagger cut: {mu_tag_cut:.1f} QDC \n {n_muons_pions_above_cut/self.n_muons_pions * 100:.1f}% of all muons and pions are above cut")
        ax.legend(fontsize = 16)

        ax.set_yscale("log")
        self.pdf_global.savefig(fig)
        plt.close()
        
        
       
        print(f"The muon tagger charge has been plotted. The optimal cut line is at {mu_tag_cut:.1f} a.u., there are {n_electrons_above_cut} electrons above the cut line and {n_muons_pions_above_cut} muons and pions, i.e. {n_muons_pions_above_cut/self.n_muons_pions * 100:.1f}% of all muons and pions ({self.n_muons_pions})...")
        
        
        self.mu_tag_cut = mu_tag_cut
        
        bins = np.linspace(0, 80, 100)
        bins_act35 = np.linspace(0, 50, 100)
        
        if n_muons_pions_above_cut/self.n_muons_pions> min_fraction_above_cut:
            print(f"there are more than {min_fraction_above_cut * 100:.1f}% of muons and pions above the muon tagger cut, we are applying it. (Please verify this on the plots)")
            
            self.using_mu_tag_cut = True
                  
            #apply the cut on the muon tagger and then find the optimal ACT35 cut (based on pions and muons)
            fig, ax = plt.subplots(figsize = (8, 6))
            
            electron_above_mu_tag = (mu_tag_tot>mu_tag_cut) & (self.df["is_electron"])
            muons_pions_above_mu_tag = (mu_tag_tot>mu_tag_cut) & (muons_pions)
            
            h, _, _ = ax.hist(self.df["act_tagger"][muons_pions_above_mu_tag], bins = bins, label = "Muons/pions passing muon tagger cut", histtype = "step")
            ax.hist(self.df["act_tagger"][muons_pions], bins = bins, label = "All muons/pions", histtype = "step")
            
            
            ax.hist(self.df["act_tagger"][electron_above_mu_tag], 
                    bins = bins, label = "Electrons passing muon tagger cut", histtype = "step", color = "red")

            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            x_min = 0.5
            x_max = 10
            mask = (bin_centers >= x_min) & (bin_centers <= x_max)
            # Find the bin index with the minimum count in that range
            min_index = np.argmin(h[mask])

            # Get the actual bin index in the original array
            index = np.where(mask)[-1]
            self.act35_cut_pi_mu = bin_centers[index[min_index]]

            ax.axvline(self.act35_cut_pi_mu, label = f"pion/muon cut line: ACT3-5 = {self.act35_cut_pi_mu:.1f} PE", color = "k", linestyle = "--")
            ax.legend(fontsize = 12)
            ax.set_ylabel("Number of events", fontsize = 18)
            ax.set_xlabel("ACT3-5 total charge (PE)", fontsize = 18)
            ax.set_yscale("log")
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - ACT3-5 with mu-tag cut", fontsize = 20)
            self.pdf_global.savefig(fig)
            plt.close()
            
            
            ############## Estimate the likelihood using the muon tagger cut distribution

            fig, ax = plt.subplots(figsize = (8, 6))
            bins = np.linspace(0, 80, 110)

            #make the event masks to identify events which pass the mu tag cut and are muons and pions
            if self.run_momentum > 300:
                mask_muons_pions = (self.df["is_electron"] == 0) & (self.df["tof"] < self.proton_tof_cut)
            else:
                mask_muons_pions = (self.df["is_electron"] == 0)
            mask_pass_mu_tag = (self.df["mu_tag_total"] > self.mu_tag_cut)
            #both (muons or pion) and passing muon tag 
            mask_both = mask_muons_pions & mask_pass_mu_tag


            h, _, _ = ax.hist(self.df["act_tagger"][mask_both],  histtype = "step", bins = bins, label = f"All Muon or pions above mu_tag cut ({sum(mask_both)}) events")
            h_all, _, _ = ax.hist(self.df["act_tagger"][mask_muons_pions],  histtype = "step", bins = bins, label = f"All Muon or pions: ({sum(mask_muons_pions)}) events")
            

            #Weight up the events passing the muon tagger cut so the maximas align (the muon peak)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            x_min = 10
            x_max = 30
            mask = (bin_centers >= x_min) & (bin_centers <= x_max)
            # Find the bin index with the maximum count in that range
            max_index = np.argmax(h_all[mask])

            index = np.where(mask)[0]
            idx_muon_peak = index[max_index]
            muon_scale = h_all[idx_muon_peak]/h[idx_muon_peak]

            #create a new histogram of the distibution passing the muon tagger cut scaled up to the muon peak
            h_muon_scaled = h * muon_scale
            
            
                  
            # plot the scaled histogram 
            ax.step(bin_centers, h_muon_scaled, where='mid', label=f"Mu/pi above cut scaled to muon peak ({sum(h_muon_scaled):.1f}) events")
            
            ### look at electrons
            n_electrons = sum(self.df["is_electron"])
            h_electron, _, _ =  ax.hist(self.df["act_tagger"][self.df["is_electron"] == 1],  histtype = "step", bins = bins, label = f"Tagged electrons ({n_electrons:.1f}) events")
            
            
            #### here, get the difference between the all and the scaled one
            h_all_minus_h_scaled = h_all - h_muon_scaled
            
            
             #clip that difference to be positive
            h_all_minus_h_scaled = h_all_minus_h_scaled.clip(0)
            
            #split the leftovers into electron (after the mu peak) and pions (before the muon peak)
            h_all_minus_h_scaled_pion = np.where(bin_centers<bin_centers[idx_muon_peak], h_all_minus_h_scaled, 0)
            
            h_all_minus_h_scaled_electron = np.where(bin_centers>=bin_centers[idx_muon_peak], h_all_minus_h_scaled, 0)
            
            ax.step(bin_centers, h_all_minus_h_scaled_pion,  where = 'mid', label = f"Pion-like distribution")

            
            h_pion = h_all_minus_h_scaled_pion

            ax.set_yscale("log")
            # ax.set_xlim(0, 80)
            ax.set_xlabel("ACT35 charge (PE)")
            ax.set_ylabel("Number of triggers")
            ax.legend()
            
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
            ax.grid()
#             self.pdf_global.savefig(fig)
            plt.close()
#             return 0
            
            
            #################### Once we have the distribution, remove the pion contamination 
            fig, ax = plt.subplots(figsize = (8, 6))
            
            ax.step(bin_centers, h_muon_scaled, where='mid', color = "black", label=f"Mu/pi above cut scaled to muon peak", linewidth = 4)
            
            ax.step(bin_centers, h_pion, where = 'mid', color = "magenta", label = f"Pion-like distribution")
            
            
            
            pion_scalling = h_muon_scaled[0]/h_pion[0] 
            h_pion_scaled = h_pion * pion_scalling
            
            ax.step(bin_centers, h_pion_scaled, where='mid', label=f"Pion distr. scaled to Mu/pi above mutag cut bin 0")
            
            h_muon = h_muon_scaled - h_pion_scaled
            h_muon = h_muon.clip(0)
            
            ax.step(bin_centers, h_muon, where='mid', label=f"Mu/pi above mutag cut bin 0 minus scaled pion distr. => muon population")
            
            
            ax.set_yscale("log")
            # ax.set_xlim(0, 80)
            ax.set_xlabel("ACT35 charge (PE)")
            ax.set_ylabel("Number of triggers")
            ax.legend()
            ax.grid()
            
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
#             self.pdf_global.savefig(fig)
            plt.close()
            
            ###############################################################################
            ############ Calculate effiency and purity as a function of the cut line
            h_pion_tot = h_pion + h_pion_scaled
            h_muon_tot = h_muon
            
            fig, ax = plt.subplots(figsize = (8, 6))
            
            ax.step(bin_centers, h_pion_tot, where='mid', color = "magenta", label=f"Total number of pions")
            ax.step(bin_centers, h_muon_tot, where='mid', color = "green", label=f"Total number of muons")
            
            ax.set_yscale("log")
            # ax.set_xlim(0, 80)
            ax.set_xlabel("ACT35 charge (PE)")
            ax.set_ylabel("Number of triggers")
            ax.legend()
            ax.grid()
            
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
#             self.pdf_global.savefig(fig)
            plt.close()
            
            ###### for my own sanity
            n_pions_left = [sum(h_pion_tot[:b]) for b in range(len(bin_centers))]
            n_muons_left = [sum(h_muon_tot[:b]) for b in range(len(bin_centers))]
            
            n_pions_right = [sum(h_pion_tot[b:]) for b in range(len(bin_centers))]
            n_muons_right = [sum(h_muon_tot[b:]) for b in range(len(bin_centers))]
            
            fig, ax = plt.subplots(figsize = (8, 6))
            ax.step(bin_centers, n_pions_left, where='mid', color = "magenta", label=f"Number of pions on the left of the cut line")
            ax.step(bin_centers, n_muons_left, where='mid', color = "green", label=f"Number of muons on the left of the cut line")
            
            ax.step(bin_centers, n_pions_right, where='mid', linestyle = "--", color = "magenta", label=f"Number of pions on the right of the cut line")
            ax.step(bin_centers, n_muons_right, where='mid', linestyle = "--", color = "green", label=f"Number of muons on the right of the cut line")
            
            ax.set_yscale("log")
            # ax.set_xlim(0, 80)
            ax.set_xlabel("ACT35 charge (PE)", fontsize = 12)
            ax.set_ylabel("Number of triggers", fontsize = 12)
            ax.legend()
            
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
            ax.grid()
#             self.pdf_global.savefig(fig)
            plt.close()
            
            ##### rejection factors function of efficiency 
            n_pions_left = np.array([sum(h_pion_tot[:b]) for b in range(len(bin_centers))])
            n_muons_left = np.array([sum(h_muon_tot[:b]) for b in range(len(bin_centers))])
            
            n_pions_right = np.array([sum(h_pion_tot[b:]) for b in range(len(bin_centers))])
            n_muons_right = np.array([sum(h_muon_tot[b:]) for b in range(len(bin_centers))])
            
            pion_efficiency = n_pions_left/sum(h_pion_tot)
            muon_efficiency = n_muons_right/sum(h_muon_tot)
            
            #number of pions rejected per muon accepted in the pion selection (i.e. left)
            muon_rejection = n_muons_right/n_muons_left
            #number of muons rejected per pion accepted in the in muon selection (i.e. right)
            pion_rejection = n_pions_left/n_pions_right
            
            #Purity calculations
            pion_purity = n_pions_left/(n_pions_left+n_muons_left)
            muon_purity = n_muons_right/(n_pions_right+n_muons_right)
            
            
            fig, ax = plt.subplots(figsize = (8, 6))
            ax.plot(pion_efficiency, muon_rejection, marker = "x", color = "magenta")
            
            ax.set_yscale("log")
            ax.set_ylim(0.5, None)
            
            ax.set_xlabel("Pion selection efficiency", fontsize = 12)
            ax.set_ylabel("# mu rejected per mu in sample", fontsize = 12)
            ax.grid()
            
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
#             self.pdf_global.savefig(fig)
            plt.close()
            
            fig, ax = plt.subplots(figsize = (8, 6))
            ax.plot(muon_efficiency, pion_rejection, marker = "x", color = "green")
            
            ax.set_yscale("log")
            ax.set_ylim(0.5, None)
            ax.set_xlabel("Muon selection efficiency", fontsize = 12)
            ax.set_ylabel("# pi rejected per pi in sample", fontsize = 12)
            ax.grid()
            
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
#             self.pdf_global.savefig(fig)
            plt.close()
            
            
            fig, ax = plt.subplots(figsize = (8, 6))
            ax.step(bin_centers, pion_purity, where='mid', color = "blue", label = "pion purity")
            ax.step(bin_centers, pion_efficiency, where='mid', color = "red", label = "pion efficiency")
            
            ax.set_xlabel("Cut line in ACT35 (PE)", fontsize = 12)
            ax.set_ylabel("")
            ax.legend()
            ax.grid()
            
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - Pions", fontsize = 20)
#             self.pdf_global.savefig(fig)
            plt.close()
            
            fig, ax = plt.subplots(figsize = (8, 6))
            ax.step(bin_centers, muon_purity, where='mid', color = "blue", label = "muon purity")
            ax.step(bin_centers, muon_efficiency, where='mid', color = "red", label = "muon efficiency")
            ax.grid()
            
#             ax.set_yscale("log")
            # ax.set_xlim(0, 80)
            ax.set_xlabel("Cut line in ACT35 (PE)", fontsize = 12)
            ax.set_ylabel("")
            ax.legend()
            ax.grid()
            
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - Muons", fontsize = 20)
#             self.pdf_global.savefig(fig)
            plt.close()
            
            
            
            
            
            
            


            
        else:
            print(f"there are not more than {min_fraction_above_cut * 100:.1f}% of muons and pions above the muon tagger cut, we are not applying it. (Please verify this on the plots)" )#
            
            self.using_mu_tag_cut = False
            
            fig, ax = plt.subplots(figsize = (8, 6))

            
            h, _, _ = ax.hist(self.df["act_tagger"][muons_pions], bins = bins, label = "Muons and pions", histtype = "step")
            
#             print(sum(muons_pions))
            
            
            ax.hist(self.df["act_tagger"][self.df["is_electron"]], 
                    bins = bins, label = "Electrons", histtype = "step", color = "red")

            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            x_min = 1.5
            
            if abs(self.run_momentum) > 700: 
                x_max = 10
            else:
                x_max = 25
            mask = (bin_centers >= x_min) & (bin_centers <= x_max)
            # Find the bin index with the minimum count in that range
            min_index = np.argmin(h[mask])

            # Get the actual bin index in the original array
            index = np.where(mask)[-1]
            self.act35_cut_pi_mu = bin_centers[index[min_index]]

            ax.axvline(self.act35_cut_pi_mu, label = f"pion/muon cut line: ACT3-5 = {self.act35_cut_pi_mu:.1f} PE", color = "k", linestyle = "--")
            ax.legend(fontsize = 12)
            ax.set_ylabel("Number of events", fontsize = 18)
            ax.set_xlabel("ACT3-5 total charge (PE)", fontsize = 18)
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)\nACT3-5 without muon tagger cut", fontsize = 20)
            ax.set_yscale("log")
            self.pdf_global.savefig(fig)
            plt.close()
            
        #at the end check visually that things are ok  
        self.plot_ACT35_left_vs_right(self.act35_cut_pi_mu)
        if self.run_momentum > 300:
            self.df["is_muon"] = (~self.df["is_electron"]) & (self.df["tof"] < self.proton_tof_cut) & (self.df["act_tagger"]>self.act35_cut_pi_mu)
            self.df["is_pion"] = (~self.df["is_electron"]) & (self.df["tof"] < self.proton_tof_cut) & (self.df["act_tagger"]<=self.act35_cut_pi_mu)
        else:
            self.df["is_muon"] = (~self.df["is_electron"]) & (self.df["act_tagger"]>self.act35_cut_pi_mu)
            self.df["is_pion"] = (~self.df["is_electron"]) & (self.df["act_tagger"]<=self.act35_cut_pi_mu)
        
        f_muons = sum(self.df["is_muon"])/self.n_muons_pions * 100
        f_pions = sum(self.df["is_pion"])/self.n_muons_pions * 100
        
        
        self.pion_purity = np.nan #pion_eff_new
        self.pion_efficiency =np.nan # pion_purity_new

        self.muon_purity = np.nan #muon_eff_new
        self.muon_efficiency = np.nan #muon_purity_new
        
        
        print(f"The pion/muon separtion cut line in ACT35 is {self.act35_cut_pi_mu:.1f}, out of {self.n_muons_pions:.1f} pions and muons, {f_muons:.1f}% are muons and {f_pions:.1f} are pions")
            
            
        
        
    def write_output_particles(self, particle_number_dict, store_PID_info, filename = None):
        """This functions writes out the WCTE tank information as well as the additional beam variables (TOF, ACT charges) necessary for making the selection, This function also stores the particle type guess obtained from the beam data but we encourage each analyser to develop their own selection"""
        
        
        if store_PID_info:
            #get the particle identification from the beam analysis
            index_mu = np.array(self.is_kept_event_id) * np.array(self.df["is_muon"])
            index_pi =  np.array(self.is_kept_event_id) * np.array(self.df["is_pion"])
            index_electron =  np.array(self.is_kept_event_id) * np.array(self.df["is_electron"])
            index_proton =  np.array(self.is_kept_event_id) * np.array(self.df["is_proton"])

            #Remove the events that are not of the given particle type
            index_mu = index_mu[index_mu !=0][:particle_number_dict["muon"]]
            index_pi = index_pi[index_pi !=0][:particle_number_dict["pion"]]
            index_electron = index_electron[index_electron !=0][:particle_number_dict["electron"]]
            index_proton = index_proton[index_proton !=0][:particle_number_dict["proton"]]
            

        # ---- inputs ------------------------------------------------------
        file_path = f"/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v0_5/{self.run_number}/WCTE_offline_R{self.run_number}S0_VME_matched.root"
        tree_name = "WCTEReadoutWindows"

        #these are the WCTE tank info branches
        branches = [
            "hit_pmt_calibrated_times",
            "window_data_quality",
            "hit_mpmt_card_ids", "hit_pmt_readout_mask",
            "hit_mpmt_slot_ids", "hit_pmt_position_ids",
            "hit_pmt_channel_ids", "hit_pmt_charges",
        ]

        BLOCK_MAX_EVENTS = 500
        WRITE_PARQUET   = True
        if filename == None:
             filename = f"Beam_PID_R{self.run_number}.parquet"
        PARQUET_FILE = filename

        # choose source
        local_copy = stage_local(file_path)  # "" if not staged
        file_for_uproot = local_copy or to_xrootd(file_path)

        # Combine all indices and particle labels
        
        if store_beam_PID:
            all_keep_idx = np.concatenate([
                index_electron,
                index_mu,
                index_pi,
                index_proton
            ])


            particle_labels = (
                ["electron"] * len(index_electron) +
                ["muon"] * len(index_mu) +
                ["pion"] * len(index_pi) +
                ["proton"] * len(index_proton)
            )
            
        else: #if we do not want to save the particle ID
            #only keep the data that is kept? no, all of the data would be better 
            all_keep_idx = self.is_kept_event_id[:particle_number_dict["triggers"]]
            
            #do not store the PID, just say that we keep them 
            particle_labels = (
                ["particle"] * len(all_keep_idx)
            )
            
            
            
            
        all_keep_idx = np.array(all_keep_idx)
        particle_labels = np.array(particle_labels)

        # Sort by index for block-wise reading
        sorted_order = np.argsort(all_keep_idx)
        all_keep_idx = all_keep_idx[sorted_order]
        particle_labels = particle_labels[sorted_order]

        # Open the file to get total entries
        with uproot.open(file_for_uproot) as f:
            tree = f[tree_name]
            n_entries = tree.num_entries

        # Keep only valid indices
        mask_valid = (all_keep_idx >= 0) & (all_keep_idx < n_entries)
        all_keep_idx = all_keep_idx[mask_valid]
        particle_labels = particle_labels[mask_valid]

        # Split into blocks
        blocks = make_blocks(all_keep_idx, BLOCK_MAX_EVENTS)
        print(f"{len(all_keep_idx)} selected entries → {len(blocks)} blocks (max {BLOCK_MAX_EVENTS} ev/block)")

        t0 = time.time()

        if WRITE_PARQUET:
            writer = None
            written = 0

            for (s, e) in blocks:
                # Read block
                with uproot.open(file_for_uproot) as f:
                    arr_block = f[tree_name].arrays(branches, library="ak", entry_start=s, entry_stop=e)

                # Select events in this block
                mask_block = (all_keep_idx >= s) & (all_keep_idx < e)
                if not np.any(mask_block):
                    del arr_block
                    gc.collect()
                    continue

                local_idx = all_keep_idx[mask_block] - s
                sel = arr_block[local_idx]

                # Add particle type column
                sel_particle = particle_labels[mask_block]
                sel = ak.with_field(sel, sel_particle, "beam_pid")    # Awkward Array

                # Convert to Arrow table
                tbl = ak.to_arrow_table(sel, list_to32=True, string_to32=True)

                # Add run number column
                run_arr = pa.array([self.run_number] * len(sel), type=pa.int32())
                tbl = tbl.append_column("run", run_arr)

                # Create Parquet writer on first block
                if writer is None:
                    meta = dict(tbl.schema.metadata or {})
                    meta.update({
                        b"wcte.run_number": str(self.run_number).encode(),
                        b"wcte.source_path": file_path.encode(),
                        b"wcte.tree_name":  tree_name.encode(),
                    })
                    schema_with_meta = tbl.schema.with_metadata(meta)
                    writer = pq.ParquetWriter(PARQUET_FILE, schema_with_meta, compression="snappy")

                writer.write_table(tbl)
                written += len(sel)

                # Clean memory
                del arr_block, sel, tbl, run_arr
                gc.collect()

            if writer is not None:
                writer.close()

            print(f"Wrote {written} rows to {PARQUET_FILE} in {time.time()-t0:.2f}s")
            
            
            
    def TOF_particle_in_ns(self, particle_name, momentum, L = 4.3431):
        ''' returns the TOF of particles of a given momentum'''
        momentum  = momentum #(give them in MeV/c)

        # masses in MeV/c^2
        masses = {
            "Electrons": 0.511,
            "Muons": 105.658,
            "Pions": 139.57,
            "Protons": 938.272,
            "Deuteron": 1876.123,
            "Helium3": 2808.392,
            "Lithium6": 5601.52,
            "Tritium": 2808.921,
        }

        if particle_name not in masses:
            raise ValueError(f"Unknown particle: {particle_name}")

        m = masses[particle_name]
        c = 2.99792458e8  # m/s

        # gamma and beta
        gamma = np.sqrt(1 + (momentum/m)**2)
        beta = np.sqrt(1 - 1/gamma**2)
        v = beta * c

        tof_seconds = L / v
        return tof_seconds * 1e9  # ns

      
    def return_losses(self, n_step, dist, particle_name, momentum, total_tof, total_length, psp, verbose = False):
        '''This function takes in the material that we are crossing, the associated energy lost table and the number of steps that we want to divide the crossing in and returns the time taken to travel the whole material and the total energy lost within it'''
        
        masses = { #MeV/c
            "Electrons": 0.511,
            "Muons": 105.658,
            "Pions": 139.57,
            "Protons": 938.272,
            "Deuteron": 1876.123,
            "Helium3": 2808.392,
            "Lithium6": 5601.52,
            "Tritium": 2808.921,
            
        }
        
        
        factor = 1 #for all other particles we have the correct table, no need for a multiplicative factor
        
        if particle_name == "Helium3":
            factor = 36
        if particle_name == "Lithium6":
            factor = 321
        
        #we do not have the energy loss tables for helium3, lithium6 or tritium, we are extrapolating from those of Deuteron crudely accounting for the mass
        
        psp = psp.reset_index(drop=True) #so we can use index entries instead of float ones
        
        g4_energy = psp["#Kinectic_energy [GeV]"].to_numpy() * 1e3 

        
        for step in range(n_step):
            delta_L = dist/n_step 
            delta_t = self.TOF_particle_in_ns(particle_name, momentum, delta_L)

            total_tof += delta_t
            total_length += delta_L


            #account for the momentum lost
            for i in range(len(momentum)):
                p = np.argmin(np.abs(g4_energy - momentum[i]))  # index of closest value to the g4 energy
                if p > len(psp["Total_st_pw [MeV/mm]"])-2:
                    p = len(psp["Total_st_pw [MeV/mm]"])-2
                    

                particle_kinetic_energy = np.sqrt(momentum[i]**2 + masses[particle_name]**2) - masses[particle_name] 


                #now modify the momentum 
                stoppingPower = (psp["Total_st_pw [MeV/mm]"].iloc[p+1] - psp["Total_st_pw [MeV/mm]"].iloc[p]) / (psp["#Kinectic_energy [GeV]"][p+1] - psp["#Kinectic_energy [GeV]"].iloc[p]) * (particle_kinetic_energy *10**(-3) - psp["#Kinectic_energy [GeV]"].iloc[p]) + psp["Total_st_pw [MeV/mm]"].iloc[p]

                particle_kinetic_energy -= stoppingPower * factor * delta_L * 1e3

                momentum[i] = np.sqrt((particle_kinetic_energy + masses[particle_name])**2 - masses[particle_name]**2)

        return momentum, total_tof, total_length
    
    def give_theoretical_TOF(self, particle, initial_momentum_guess):
        '''This function returns the T0-T1, T0-T4 and T4-T1 TOFs that a given particle would have  for a given initial momentum (which can be a scalar or an array). It is a stepper function that propagates the momentum at each step, adding up the travel time to form the total TOF and taking into account the momentum lost at each step based on pre-calculated G4 tables and accurate beam material budget surveys.'''
    
        #read the detector positions and dimensions from the yaml file 
        det_module = db.from_yaml("../include/wcte_beam_detectors.yaml")
        
        if self.run_momentum < 0:
            if particle == "Electrons":
                p_name = "electron"
            if particle == "Muons":
                p_name = "muMinus"
            if particle == "Pions":
                p_name = "piMinus"
            if particle == "Proton":
                return 0

        elif self.run_momentum > 0:
            if particle == "Electrons":
                p_name = "positron"
            if particle == "Muons":
                p_name = "muPlus"
            if particle == "Pions":
                p_name = "piPlus"
            if particle == "Protons":
                p_name = "proton"
            if particle == "Deuteron":
                p_name = "deuteron"
            if particle == "Tritium":
                p_name = "triton"
                
        str_n_eveto = str(self.n_eveto)
        str_n_tagger = str(self.n_tagger)
        
  
        # Read in the theoretical losses from G4 tables provided by Arturo
        losses_dataset_air = f"../include/{p_name}StoppingPowerAirGeant4.csv"
        losses_dataset_plasticScintillator = f"../include/{p_name}StoppingPowerPlasticScintillatorGeant4.csv"
        losses_dataset_mylar = f"../include/{p_name}StoppingPowerMylarGeant4.csv"
        #ACT tables
        losses_dataset_upstream = f"../include/{p_name}StoppingPowerAerogel1p{str_n_eveto[2:]}Geant4.csv"
        losses_dataset_downstream = f"../include/{p_name}StoppingPowerAerogel1p{str_n_tagger[2:]}Geant4.csv"
   
        

        #Open all the files
        with open(losses_dataset_air, mode = 'r') as file:
            psp_air = pd.read_csv(file) #psp = particle stopping power

        with open(losses_dataset_plasticScintillator, mode = 'r') as file:
            psp_plasticScintillator = pd.read_csv(file) 

        with open(losses_dataset_upstream, mode = 'r') as file:
            psp_upstreamACT = pd.read_csv(file)

        with open(losses_dataset_downstream, mode = 'r') as file:
            psp_downstreamACT = pd.read_csv(file)

        with open(losses_dataset_mylar, mode = 'r') as file:
            psp_mylar = pd.read_csv(file)
        
        #these are the reference particle tables for this specific particle
        reference_tables = {"mylar": psp_mylar,
                            "scintillator": psp_plasticScintillator,
                            "upstreamAerogel": psp_upstreamACT,
                            "downstreamAerogel": psp_downstreamACT,
                            "air": psp_air,
                            "vinyl": psp_plasticScintillator,
                            }
        
        

        #Here we are creating a large array holding all of the materials in the beamline and the distance between them.

        if self.there_is_ACT5:
            list_all_detectors = ["Mylar_beam_window","T0","T4","ACT0_"+str(self.n_eveto),"ACT1_"+str(self.n_eveto),"ACT2_"+str(self.n_eveto),"ACT3_"+str(self.n_tagger),"ACT4_"+str(self.n_tagger),"ACT5_"+str(self.n_tagger),"T1","T5","WCTE_window"]
        else:
            list_all_detectors = ["Mylar_beam_window","T0","T4","ACT0_"+str(self.n_eveto),"ACT1_"+str(self.n_eveto),"ACT2_"+str(self.n_eveto),"ACT3_"+str(self.n_tagger),"ACT4_"+str(self.n_tagger),"T1","T5","WCTE_window"]


        #These are the arrays holding all of the layers of material that the particles see
        array_layers_thickness = []
        array_layers_material = []
        array_layers_name = []

        for d, det in enumerate(list_all_detectors):

            all_layers_name, all_layers_thickness, all_layers_material = det_module.get_all_layers(det)

            #if it is not the last detector we need to add a layer of air to it
            if d < len(list_all_detectors)-1:
                det1 = det
                det2 = list_all_detectors[d+1]
                
                det1_name = det
                det2_name = list_all_detectors[d+1]
                
                #when looking at the distance we only care about the number of the detector
                if det1[0:3]=="ACT":
                    det1 = det1[0:4]
                    det1_name = f"{det[0:3]}{det[4::]}"
                if det2[0:3]=="ACT":
                    det2 = det2[0:4]
                    det2_name = f"{list_all_detectors[d+1][0:3]}{list_all_detectors[d+1][4:]}"
                    
           
                half_thickness_det1 = det_module.get_total_thickness_m(det1_name)/2
                half_thickness_det2 = det_module.get_total_thickness_m(det2_name)/2
                #account for the air gap between detectors
                distance_to_next_det = det_module.distance_m(det1, det2) - half_thickness_det1 - half_thickness_det2
                
            else:
                distance_to_next_det = 0
               
            
            gap_name = f"{det1}_{det2}_air"

            array_layers_thickness.extend(all_layers_thickness)
            array_layers_material.extend(all_layers_material)
            array_layers_name.extend(all_layers_name)
            
            array_layers_thickness.append(distance_to_next_det)
            array_layers_material.append("air")
            array_layers_name.append(gap_name)

#         print("\nThe list of all materials are: ", array_layers_name, "\n")
        #make a copy of the initial momentum guesses that will be continuously updated to reflect how the momentum is reduced by travelling through matter.
        live_momentum = initial_momentum_guess.copy()
        total_length = 0
        #for each of the momenta we have a TOF
        total_tof = np.zeros(len(live_momentum))

        #now propagate that momentum through all the layers
        for l, layer_name in enumerate(array_layers_name):
            #We need to chose the number of steps per layer
            #if larger than 20 cm then it's probably air and we want somewhat large
            if array_layers_thickness[l] > 20e-2:
                steps = 50
            else:
                steps = 10

            #just starting crossing each of the scintillators we are saving the travel time so we can make the differences afterwards
            if layer_name == "T0_scintillator":
                T0_time = total_tof.copy()

            if layer_name == "T4_scintillator":
                T4_time = total_tof.copy()

            if layer_name == "T1_scintillator":
                T1_time = total_tof.copy()

            if layer_name == "T5_scintillator":
                T5_time = total_tof.copy()
                

#             print("layer name: ", array_layers_name[l])

            live_momentum, total_tof, total_length = self.return_losses(steps, array_layers_thickness[l], particle, live_momentum, total_tof, total_length, reference_tables[array_layers_material[l]], verbose = False)
            
#             print("After this layer, the total leght travelled is: ", total_length, "m")


        #after we have gone through all of the materials, we output the initial guesses and each of the TOFs: T0-T1, T0-T4, T4-T1 (for now) Those are arrays corresponding to the theoretical tof for each of the initial momenta guesses
        #This will be useful to fix the T4 timing offset (maybe) we should see some jitter
        T0T1_theoretical_TOF = T1_time-T0_time
        T0T4_theoretical_TOF = T4_time-T0_time
        T4T1_theoretical_TOF = T1_time-T4_time

        return initial_momentum_guess, live_momentum, T0T1_theoretical_TOF, T0T4_theoretical_TOF, T4T1_theoretical_TOF
    
    def give_tof(self, particle, initial_momentum, n_eveto_group, n_tagger_group, there_is_ACT5, run_monentum, T_pair = "T0T1", verbose = True):
        '''This function returns the T0-T1 TOF that a given particle would have as a function of its initial momentum, later to be compared with the recorded TOF for estimating the inital momentum. It is a stepper function that propagates the momentum at each step and adds up the total TOF, taking into account the momentum lost at each step'''
        #the default number of steps per material is 10 but we do modify it based on the material density
        #to speed up the process whilst keeping up the accuracy
        n_step = 10
        
     
    
        momentum = initial_momentum.copy()

        #Set up the ACTs with dimensions given by Sirous
        if (self.n_eveto == 1.01):
            ACT02_material = "1p01"
            ACT02_thick_per_box = 0.04 #cm

        elif (self.n_eveto == 1.03):
            ACT02_material = "1p03"
            ACT02_thick_per_box = 0.04 #cm
        elif (self.n_eveto == 1.075):
            ACT02_material = "1p075"
            ACT02_thick_per_box = 0.10/3 #cm
            


        if (self.n_tagger == 1.047):
            ACT35_material = "1p047"
            ACT35_thick_per_box = 0.16/3


        elif (self.n_tagger == 1.06):
            ACT35_material = "1p06"
            if not self.there_is_ACT5:
                ACT35_thick_per_box = 0.10/2 #cm (assume we take the two 6cm instead of one of each but i do not know for sure)

        elif (self.n_tagger == 1.015):
            ACT35_material = "1p015"
            ACT35_thick_per_box = 0.06
                

        elif (self.n_tagger == 1.03):
            ACT35_material = "1p03"
            if self.there_is_ACT5:
                ACT35_thick_per_box = 0.12/3 #cm
            else:
                ACT35_thick_per_box = 0.08/2 #cm
                
        elif (self.n_tagger == 1.15):
            ACT35_material = "1p15"
            ACT35_thick_per_box = 0.02 #cm
            if self.there_is_ACT5:
                print("In principle we only have one downstream n = 1.15 ACT, please check the run info")
                raise Error


        #store the total thickness plus 0.34 mm which is for the various reflective sheets and black tape
        ACT02_thickness = 3 * (ACT02_thick_per_box + 0.34e-3)


        if self.there_is_ACT5:
            ACT35_thickness = (ACT35_thick_per_box + 0.34e-3) * 3
        else:
            ACT35_thickness = (ACT35_thick_per_box + 0.34e-3) * 2

        #Improved method for detector distances and dimensions: use Bruno's yaml file.  
        
        det_module = db.from_yaml("../include/wcte_beam_detectors.yaml")

           
        #Trigger scintillators assumed to be all the same thickness, from Bruno's slides
        #This TS thickness is only the acrylic part 
        t0_thickness =  det_module.get_thickness_m("T0", "scintillator") #6.4e-3 #mm to m
        t4_thickness =  det_module.get_thickness_m("T4", "scintillator") #6.4e-3 #mm to m
        t1_thickness =  det_module.get_thickness_m("T1", "scintillator") #6.4e-3 #mm to m
        TOF_thickness = det_module.get_thickness_m("T5", "scintillator") #6.4e-3
        
        T0_mylar_thickness = det_module.get_thickness_m("T0", "mylar")
        T1_mylar_thickness = det_module.get_thickness_m("T1", "mylar")
        T4_mylar_thickness = det_module.get_thickness_m("T4", "mylar")
        TOF_mylar_thickness = det_module.get_thickness_m("T5", "mylar")
        
        T0_vinyl_thickness = det_module.get_thickness_m("T0", "vinyl")
        T1_vinyl_thickness = det_module.get_thickness_m("T1", "vinyl")
        T4_vinyl_thickness = det_module.get_thickness_m("T4", "vinyl")
        TOF_vinyl_thickness = det_module.get_thickness_m("T5", "vinyl")
        
        T4_total_thickness = det_module.get_total_thickness_m("T4")
        T1_total_thickness = det_module.get_total_thickness_m("T1")

        #Distance between the TS
        L =  det_module.distance_m("T0", "T1")  #444.03e-2  #T0 to T1
        L_t0t4 = det_module.distance_m("T0", "T4")  
        #305.68e-2 - 5.67e-2/2  + additional_length   #Need to correct for 5.67cm offset
        L_t4t1 = det_module.distance_m("T4", "T1")  #143.38e-2 - 5.67e-2/2
        t1_TOF_distance = det_module.distance_m("T1", "T5")   #215.13e-2
       
        
        #WCTE tank window 
        beam_window_thickness = 1.2e-3 #https://wcte.hyperk.ca/wg/simulation-and-analysis/meetings/2024/20241122/meeting/simulation-update-beam-pipe-and-camera-housing/pipe_housing_wcsim_20241122.pdf/view


        #simplification, consider all the air as a single blob split in half before and after the aerogels
        #The additional length is the air gap inside the ACTs * nACTs
#         L_air_T4_to_T1 = L_t4t1 - ACT02_thickness - ACT35_thickness + additional_length #- 0.06 - 0.003  
        L_air_T4_to_T1 = L_t4t1 - ACT02_thickness - ACT35_thickness - T4_total_thickness/2 - T1_total_thickness/2 #- 0.06 - 0.003  
        #0.06 is the default value, 0.003 is the fine tune based on momentum measurements, see Alie's slides: https://wcte.hyperk.ca/wg/beam/meetings/2025/20250922/meeting/momentum-estimation-from-tof/acraplet_20250922_wctebeam.pdf


        #check what particle we have
        masses = { #MeV/c
            "Electrons": 0.511,
            "Muons": 105.658,
            "Pions": 139.57,
            "Protons": 938.272,
            "Deuteron": 1876.123,
            "Helium3": 2808.392,
            "Lithium6": 5601.52,
            "Tritium": 2808.921,
        }

        #Check run polarity
        if self.run_momentum < 0:
            if particle == "Electrons":
                p_name = "electron"
            if particle == "Muons":
                p_name = "muMinus"
            if particle == "Pions":
                p_name = "piMinus"
            if particle == "Proton":
                return 0

        elif self.run_momentum > 0:
            if particle == "Electrons":
                p_name = "positron"
            if particle == "Muons":
                p_name = "muPlus"
            if particle == "Pions":
                p_name = "piPlus"
            if particle == "Protons":
                p_name = "proton"
            if particle == "Deuteron":
                p_name = "deuteron"
                
            # to first order, scale up the proton dEdx 
            if particle == "Helium3":
                p_name = "proton"
            if particle == "Tritium":
                p_name = "proton"
            if particle == "Lithium6":
                p_name = "proton"

        # Read in the theoretical losses from G4 tables provided by Arturo
        losses_dataset_air = f"../include/{p_name}StoppingPowerAirGeant4.csv"
        losses_dataset_plasticScintillator = f"../include/{p_name}StoppingPowerPlasticScintillatorGeant4.csv"
        losses_dataset_mylar = f"../include/{p_name}StoppingPowerMylarGeant4.csv"
        #ACT tables
        losses_dataset_upstream = f"../include/{p_name}StoppingPowerAerogel{ACT02_material}Geant4.csv"
        losses_dataset_downstream = f"../include/{p_name}StoppingPowerAerogel{ACT35_material}Geant4.csv"
   
        particle_name = particle

        #Open all the files
        with open(losses_dataset_air, mode = 'r') as file:
            psp_air = pd.read_csv(file) #psp = particle stopping power

        with open(losses_dataset_plasticScintillator, mode = 'r') as file:
            psp_plasticScintillator = pd.read_csv(file) 

        with open(losses_dataset_upstream, mode = 'r') as file:
            psp_upstreamACT = pd.read_csv(file)

        with open(losses_dataset_downstream, mode = 'r') as file:
            psp_downstreamACT = pd.read_csv(file)

        with open(losses_dataset_mylar, mode = 'r') as file:
            psp_mylar = pd.read_csv(file)



        if verbose: print(f"\n The initial momenta considered for the {p_name} are {momentum}") 

            
        #reset the time of flight and the total length, will do again laterbut necessary for the Mylar to have
        total_tof = np.zeros(len(momentum))
        total_length = 0
        
        
        
        #re-write this code a bit cleaner, account for the dEdx at each step
        #To be cleaned up/expanded on, when ready
        
        if self.there_is_ACT5:
            #all of the materials that are in the beam 
            names_materials = ["Mylar_beam_window",
                              "air_beam_window_to_T0",
                              "T0_vinyl",
                              "T0_Mylar",
                              "T0_scintillator",
                              "T0_Mylar",
                              "T0_vinyl",
                              "air_T0_to_T4",
                              "T4_vinyl",
                              "T4_Mylar",
                              "T4_scintillator",
                              "T4_Mylar",
                              "T4_vinyl",
                              "air_T4_to_ACT0",
                              "ACT0_vinyl",
                              "ACT0_Mylar",
                              "ACT0_aerogel",
                              "ACT0_Mylar",
                              "ACT0_air",
                              "ACT0_vinyl",
                              "air_ACT0_to_ACT1",
                              "ACT1_vinyl",
                              "ACT1_Mylar",
                              "ACT1_aerogel",
                              "ACT1_Mylar",
                              "ACT1_air",
                              "ACT1_vinyl",
                              "air_ACT1_to_ACT2",
                              "ACT2_vinyl",
                              "ACT2_Mylar",
                              "ACT2_aerogel",
                              "ACT2_Mylar",
                              "ACT2_air",
                              "ACT2_vinyl",
                              "air_ACT2_to_ACT3",
                              "ACT3_vinyl",
                              "ACT3_Mylar",
                              "ACT3_aerogel",
                              "ACT3_Mylar",
                              "ACT3_air",
                              "ACT3_vinyl",
                              "air_ACT3_to_ACT4",
                              "ACT4_vinyl",
                              "ACT4_Mylar",
                              "ACT4_aerogel",
                              "ACT4_Mylar",
                              "ACT4_air",
                              "ACT4_vinyl",
                              "air_ACT4_to_ACT5",
                              "ACT5_vinyl",
                              "ACT5_Mylar",
                              "ACT5_aerogel",
                              "ACT5_Mylar",
                              "ACT5_air",
                              "ACT5_vinyl",
                              "air_ACT5_to_T1",
                              "T1_vinyl",
                              "T1_Mylar",
                              "T1_scintillator",
                              "T1_Mylar",
                              "T1_vinyl",
                              "air_T1_to_T5",
                              "T5_vinyl",
                              "T5_Mylar",
                              "T5_scintillator",
                              "T5_Mylar",
                              "T5_vinyl",
                              "air_T5_to_window", 
                              "WCTE_window"]
        else:
                names_materials = ["Mylar_beam_window",
                              "air_beam_window_to_T0",
                              "T0_vinyl",
                              "T0_Mylar",
                              "T0_scintillator",
                              "T0_Mylar",
                              "T0_vinyl",
                              "air_T0_to_T4",
                              "T4_vinyl",
                              "T4_Mylar",
                              "T4_scintillator",
                              "T4_Mylar",
                              "T4_vinyl",
                              "air_T4_to_ACT0",
                              "ACT0_vinyl",
                              "ACT0_Mylar",
                              "ACT0_aerogel",
                              "ACT0_Mylar",
                              "ACT0_air",
                              "ACT0_vinyl",
                              "air_ACT0_to_ACT1",
                              "ACT1_vinyl",
                              "ACT1_Mylar",
                              "ACT1_aerogel",
                              "ACT1_Mylar",
                              "ACT1_air",
                              "ACT1_vinyl",
                              "air_ACT1_to_ACT2",
                              "ACT2_vinyl",
                              "ACT2_Mylar",
                              "ACT2_aerogel",
                              "ACT2_Mylar",
                              "ACT2_air",
                              "ACT2_vinyl",
                              "air_ACT2_to_ACT3",
                              "ACT3_vinyl",
                              "ACT3_Mylar",
                              "ACT3_aerogel",
                              "ACT3_Mylar",
                              "ACT3_air",
                              "ACT3_vinyl",
                              "air_ACT3_to_ACT4",
                              "ACT4_vinyl",
                              "ACT4_Mylar",
                              "ACT4_aerogel",
                              "ACT4_Mylar",
                              "ACT4_air",
                              "ACT4_vinyl",
                              "air_ACT4_to_T1",
                              "T1_vinyl",
                              "T1_Mylar",
                              "T1_scintillator",
                              "T1_Mylar",
                              "T1_vinyl",
                              "air_T1_to_T5",
                              "T5_vinyl",
                              "T5_Mylar",
                              "T5_scintillator",
                              "T5_Mylar",
                              "T5_vinyl",
                              "air_T5_to_window", 
                              "WCTE_window"]
        
        
        
        
        
        #We look into the initial momenta as the particle exits the beam pipe 
        #need to take into account the energy lost to the Mylar window

        #Note that the TOF only starts as we start crossing T0, 
        #we however still need to account for the momentum lost 

        ###################  Exit Beam Pipe  ############################################# 
        #1 step is enough in the Mylar
        momentum, _, total_length = self.return_losses(5, 0.25e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        if verbose: print(f"After crossing the mylar beam pipe window, the momentum is {momentum} and the TOF is {total_tof}")

        ##################    Cross T0    ###############################################
        #First half, 10 steps
        #Actually, say the TOF starts at the very beginning of T0
        
        momentum, total_tof, total_length = self.return_losses(3, 0.18e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)


        ################## beginning of TOF calculation
        #reset the time of flight and the total length
        total_tof = np.zeros(len(momentum))
        total_length = 0

        #second half, 25 steps
        
        momentum, _, total_length = self.return_losses(25, t0_thickness, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)
        momentum, total_tof, total_length = self.return_losses(3, 0.18e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)
        #At the moment we do not add to the tof the half-crossing of T0 we might have to ? Have that 

        if verbose: print(f"After crossing the T0, the momentum is {momentum} and the TOF is {total_tof}")


    ###################    Air  between T0 and T4  ######################################### 
        momentum, total_tof, total_length = self.return_losses(100, L_t0t4, particle_name, momentum, total_tof, total_length, psp_air, verbose = verbose)
        if verbose: print(f"After crossing the air between T0 and T4, the {particle_name} momentum is {momentum} and the TOF is {total_tof}")

    ########################    Cross T4  #####################################
        momentum, total_tof, total_length = self.return_losses(3, 0.18e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)
        momentum, total_tof, total_length = self.return_losses(35, t4_thickness, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)
        momentum, total_tof, total_length = self.return_losses(3, 0.18e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)
        if verbose: print(f"After crossing the T4, the {particle_name} momentum is {momentum} and the TOF is {total_tof}")

    ####################    Half of the total air between T4 and T1  ########################
        momentum, total_tof, total_length = self.return_losses(25, L_air_T4_to_T1/7, particle_name, momentum, total_tof, total_length, psp_air, verbose = verbose)
        if verbose: print(f"After crossing a seventh the air gap between T4 and T1, the {particle_name} momentum is {momentum} and the TOF is {total_tof}")

    ###################    Cross ACT0  ####################
        #one layer tape (taken as a layer of black sheet at 0.06mm - assumed plastic scintillator) + 2 layers = 0.18mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.18e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)

        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        #aerogel
        momentum, total_tof, total_length = self.return_losses(25, ACT02_thick_per_box, particle_name, momentum, total_tof, total_length, psp_upstreamACT, verbose = verbose)

        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        # 2 layers pf black sheet = 0.12mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.12e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)
        if verbose: print(f"After ACT0, the  momentum is {momentum} and the TOF is {total_tof}")

        momentum, total_tof, total_length = self.return_losses(n_step, L_air_T4_to_T1/7, particle_name, momentum, total_tof, total_length, psp_air, verbose = verbose)

    #########################################    Cross ACT1  ################################################################## 

        #one layer tape (taken as a layer of black sheet at 0.06mm - assumed plastic scintillator) + 2 layers = 0.18mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.18e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)

        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        #aerogel
        momentum, total_tof, total_length = self.return_losses(25, ACT02_thick_per_box, particle_name, momentum, total_tof, total_length, psp_upstreamACT, verbose = verbose)

        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        # 2 layers pf black sheet = 0.12mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.12e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)

        if verbose: print(f"After ACT1, the {particle_name} momentum is {momentum} and the TOF is {total_tof}")

        momentum, total_tof, total_length = self.return_losses(5, L_air_T4_to_T1/7, particle_name, momentum, total_tof, total_length, psp_air, verbose = verbose)

    #########################################    Cross ACT2  ################################################################## 

         #one layer tape (taken as a layer of black sheet at 0.06mm - assumed plastic scintillator) + 2 layers = 0.18mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.18e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)

        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        #aerogel
        momentum, total_tof, total_length = self.return_losses(25, ACT02_thick_per_box, particle_name, momentum, total_tof, total_length, psp_upstreamACT, verbose = verbose)

        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        # 2 layers pf black sheet = 0.12mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.12e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)
        if verbose: print(f"After ACT2, the {particle_name} momentum is {momentum} and the TOF is {total_tof}")

        momentum, total_tof, total_length = self.return_losses(5, L_air_T4_to_T1/7, particle_name, momentum, total_tof, total_length, psp_air, verbose = verbose)

    #########################################    Cross ACT3  ################################################################## 

         #one layer tape (taken as a layer of black sheet at 0.06mm - assumed plastic scintillator) + 2 layers = 0.18mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.18e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)

        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        #aerogel
        momentum, total_tof, total_length = self.return_losses(25, ACT35_thick_per_box, particle_name, momentum, total_tof, total_length, psp_downstreamACT, verbose = verbose)

        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        # 2 layers pf black sheet = 0.12mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.12e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)
        if verbose: print(f"After ACT3, the {particle_name} momentum is {momentum} and the TOF is {total_tof}")

        momentum, total_tof, total_length = self.return_losses(5, L_air_T4_to_T1/7, particle_name, momentum, total_tof, total_length, psp_air, verbose = verbose)
    #########################################    Cross ACT4  ################################################################## 

         #one layer tape (taken as a layer of black sheet at 0.06mm - assumed plastic scintillator) + 2 layers = 0.18mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.18e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)

        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        #aerogel
        momentum, total_tof, total_length = self.return_losses(25, ACT35_thick_per_box, particle_name, momentum, total_tof, total_length, psp_downstreamACT, verbose = verbose)
        #reflector of 0.02mm
        momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

        # 2 layers pf black sheet = 0.12mm 
        momentum, total_tof, total_length = self.return_losses(3, 0.12e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)
        if verbose: print(f"After ACT4, the {particle_name} momentum is {momentum} and the TOF is {total_tof}")

        momentum, total_tof, total_length = self.return_losses(5, L_air_T4_to_T1/7, particle_name, momentum, total_tof, total_length, psp_air, verbose = verbose)
    #########################################    Cross ACT5  ################################################################## 


        if there_is_ACT5:
             #one layer tape (taken as a layer of black sheet at 0.06mm - assumed plastic scintillator) + 2 layers = 0.18mm 
            momentum, total_tof, total_length = self.return_losses(3, 0.18e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)

            #reflector of 0.02mm
            momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

            #aerogel
            momentum, total_tof, total_length = self.return_losses(25, ACT35_thick_per_box, particle_name, momentum, total_tof, total_length, psp_downstreamACT, verbose = verbose)
            if verbose: print(f"After ACT5, the {particle_name} momentum is {momentum} and the TOF is {total_tof}")    
            #reflector of 0.02mm
            momentum, total_tof, total_length = self.return_losses(1, 0.02e-3, particle_name, momentum, total_tof, total_length, psp_mylar, verbose = verbose)

            # 2 layers pf black sheet = 0.12mm 
            momentum, total_tof, total_length = self.return_losses(3, 0.12e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)


        momentum, total_tof, total_length = self.return_losses(5, L_air_T4_to_T1/7, particle_name, momentum, total_tof, total_length, psp_air, verbose = verbose)
        momentum, total_tof, total_length = self.return_losses(3, 0.18e-3, particle_name, momentum, total_tof, total_length, psp_plasticScintillator, verbose = verbose)
    #     momentum, total_tof, total_length = return_losses(n_step, L_air_T4_to_T1/2, particle_name, momentum, total_tof, total_length, psp_air, verbose = verbose)
        if verbose: print(f"After crossing a seventh the air gap between T4 and T1, the {particle_name} momentum is {momentum} and the TOF is {total_tof}")

        if verbose: print(f"The total travel length is {total_length}")
        if verbose: print(f"Initial momenta for {particle_name}: {initial_momentum}\n Final momenta: {momentum} \n TOF between T0 and T1: {total_tof}\n")
            
        new_tof = np.zeros(len(momentum))
            
        #continue the propagation, we are almost there to the beam window, useful to have for later extrapolation
        ############################## cross T1 #####################
        momentum, _, total_length = self.return_losses(20, t0_thickness, particle_name, momentum, new_tof, total_length, psp_plasticScintillator, verbose = verbose)
        
        
        ############################## cross air gap between T1 and TOF 
        momentum, _, total_length = self.return_losses(20, t1_TOF_distance, particle_name, momentum, new_tof, total_length, psp_air, verbose = verbose)
        
        
        ######################### cross TOF #############################
        momentum, _, total_length = self.return_losses(20, TOF_thickness, particle_name, momentum, new_tof, total_length, psp_plasticScintillator, verbose = verbose)
        
        ################ cross beam window into the tank
        momentum, _, total_length = self.return_losses(5, beam_window_thickness, particle_name, momentum, new_tof, total_length, psp_mylar, verbose = verbose)
        
        
        return initial_momentum, momentum, total_tof
     
        
        
    def extrapolate_momentum(self, initial_momentum, theoretical_tof, measured_tof, err_measured_tof):
        '''From the theoretical TOF and the measaured tof, extrapolate the value of the momentum with the associated error ''' 
        diff_m_exp = list(abs(theoretical_tof-measured_tof)) 
        A = diff_m_exp.index(min(diff_m_exp)) 
       
        if A == len(diff_m_exp):
            B = diff_m_exp.index(diff_m_exp[A-1])
        elif A == 0:
            B = diff_m_exp.index(diff_m_exp[A+1])
        else:  
            try:
                #if we cannot find an intrercept, we do not include it
                B = diff_m_exp.index(min(diff_m_exp[A+1], diff_m_exp[A-1])) 
            except: 
                print("The measured TOF is ", measured_tof, "the theoretical TOF is", theoretical_tof, " We did not find an intercept, returning 0, 0 for the momentum guess and error")
                return 0, 0
        #simple linear extrapolation: m = ( y_b x_a - y_a x_b ) / (x_a-x_b) 
        x_a, y_a = initial_momentum[A], theoretical_tof[A] 
        x_b, y_b = initial_momentum[B], theoretical_tof[B] 
        intercept = (y_b*x_a - y_a*x_b) / (x_a-x_b) 
        gradient = (y_a - intercept)/x_a 
        momentum_guess = (measured_tof - intercept)/gradient 
        momentum_minus = (measured_tof - err_measured_tof - intercept)/gradient 
        momentum_plus = (measured_tof + err_measured_tof - intercept)/gradient 
        err_mom = momentum_guess-momentum_plus 
        return momentum_guess, err_mom
    
    
    def extrapolate_trigger_momentum_coarse(self, initial_momentum, theoretical_tof, measured_tof, err_measured_tof):
        '''From the theoretical TOF and the measaured tof, extrapolate the value of the momentum with the associated error, same version as abopve but working with arrays instead of single values, re-written with help from generative AI'''
        
        # Make sure they're numpy arrays
        initial_momentum = np.asarray(initial_momentum)
        theoretical_tof = np.asarray(theoretical_tof)

        # For each measured TOF, find closest index in theoretical_tof
        idx_closest = np.abs(theoretical_tof[:, None] - measured_tof).argmin(axis=0)

        # Pick neighbor for interpolation (next or previous)
        # use np.clip to avoid going out of bounds
        idx_neighbor = np.clip(idx_closest + 1, 0, len(theoretical_tof)-1)

        x_a = initial_momentum[idx_closest]
        y_a = theoretical_tof[idx_closest]
        x_b = initial_momentum[idx_neighbor]
        y_b = theoretical_tof[idx_neighbor]

        # linear interpolation parameters
        intercept = (y_b * x_a - y_a * x_b) / (x_a - x_b)
        gradient = (y_a - intercept) / x_a

        momentum_guess = (measured_tof - intercept) / gradient
        momentum_minus = (measured_tof - err_measured_tof - intercept) / gradient
        momentum_plus = (measured_tof + err_measured_tof - intercept) / gradient
        err_mom = np.abs(momentum_guess - momentum_plus)

        return momentum_guess, err_mom
    
    
    def extrapolate_trigger_momentum(self, initial_momentum, theoretical_tof,
                                 measured_tof, err_measured_tof):
        """
        Vectorized: invert tof(p) by interpolation tof->p, compute propagated error
        sigma_p = sigma_t / |dt/dp|.
        initial_momentum and theoretical_tof must describe the same-length grid.
        """

        p_grid = np.asarray(initial_momentum)
        t_grid = np.asarray(theoretical_tof)

        # Ensure monotonic t_grid for interpolation: sort by t_grid
        sort_idx = np.argsort(t_grid)
        t_sorted = t_grid[sort_idx]
        p_sorted = p_grid[sort_idx]

        # Optionally remove duplicates in t_sorted (np.interp requires strictly increasing x)
        # We'll compress duplicates by keeping the first occurrence
        dif = np.diff(t_sorted)
        keep = np.concatenate(([True], dif != 0))
        t_unique = t_sorted[keep]
        p_unique = p_sorted[keep]

        # Invert by interpolation: p(t) via np.interp
        momentum_guess = np.interp(measured_tof, t_unique, p_unique,
                                   left=np.nan, right=np.nan)

        # Numerical derivative dt/dp on original grid (use p_grid order)
        # compute dt/dp as gradient(t_grid) / gradient(p_grid)
        dp = np.gradient(p_grid)
        dt = np.gradient(t_grid)
        dtdp = dt / dp  # same length as grid

        # Now map derivative to the momentum_guess values by interpolation
        # but we need dtdp as function of p: sort by p_grid and use p_sorted (already sorted by t earlier)
        # For safety, sort p_grid increasing:
        p_sort_idx = np.argsort(p_grid)
        p_for_deriv = p_grid[p_sort_idx]
        dtdp_for_deriv = dtdp[p_sort_idx]

        # interpolate dtdp at momentum_guess
        dtdp_at_guess = np.interp(momentum_guess, p_for_deriv, dtdp_for_deriv,
                                  left=np.nan, right=np.nan)

        # propagate TOF error to momentum error
        err_mom = np.abs(err_measured_tof / dtdp_at_guess)

        return momentum_guess, err_mom

            
    def estimate_particle_momentum(self):    
        

        particles_tof_names = {"Muons": "muon",
                               "Pions": "pion",
                               "Protons": "proton",
                               "Deuteron": "deuteron",
                               "Tritium": "tritium"
        }
        
        #dictionary to keep the initial and final mean particle momenta, as estimated from the T0T1 tof
        self.particle_mom_mean = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0,"tritium":0,"lithium6":0}
        self.particle_mom_mean_err = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0,"tritium":0,"lithium6":0}
        
        self.particle_mom_final_mean = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0,"tritium":0,"lithium6":0}
        self.particle_mom_final_mean_err = {"electron": 0,"muon": 0,"pion": 0,"proton": 0,"deuteron":0,"helium3":0,"tritium":0,"lithium6":0}
        
        
        for particle in ["Muons", "Pions", "Protons", "Deuteron", "Tritium"]:
            measured_tof_mean = self.particle_tof_mean[particles_tof_names[particle]]
            measured_tof_error = self.particle_tof_eom[particles_tof_names[particle]]
            
            
            measured_tof_t0t4_mean = self.particle_tof_t0t4_mean[particles_tof_names[particle]]
            measured_tof_t0t4_error = self.particle_tof_t0t4_eom[particles_tof_names[particle]]
            
            if measured_tof_mean >= 0.2:
                if particle == "Muons" or particle == "Pions":
                    momentum_guess=np.linspace(170, 1900, 56)
                elif particle == "Protons":
                    #heavier particles have to have more energy to reach T1 (and even more so, T5) 
                    momentum_guess=np.linspace(300, 1900, 46)

                elif particle == "Deuteron":
                    #heavier particles have to have more energy to reach T1 (and even more so, T5) 
                    momentum_guess=np.linspace(650, 1900, 36)

                elif particle == "Tritium":
                    #heavier particles have to have more energy to reach T1 (and even more so, T5) 
                    momentum_guess=np.linspace(300, 1900, 36)

                initial_momentum_th, final_momentum_th, T0T1_TOF_th, T0T4_TOF_th, T4T1_TOF_th = self.give_theoretical_TOF(particle, momentum_guess)

                fig, ax = plt.subplots(figsize = (8, 6))
                ax.plot(initial_momentum_th, T0T1_TOF_th, color = "g", marker = "+", label = f"Predicted T0T1 TOF")
                ax.plot(initial_momentum_th, T0T4_TOF_th, color = "b", marker = "x", label = f"Predicted T0T4 TOF")
                #ax.plot(initial_momentum_th, T4T1_TOF_th, color = "k", marker = "o", label = f"{particle}: theoretical T4T1 TOF")


                

               
                
                ax.axhline(measured_tof_mean, color = "g", linestyle = "--", label = f"Measured T0T1 TOF:\n{measured_tof_mean:.2f} +/- {measured_tof_error:.1e} ns")
                ax.axhspan(measured_tof_mean - measured_tof_error, measured_tof_mean + measured_tof_error, color = "g", alpha = 0.2)

                ax.axhline(measured_tof_t0t4_mean, color = "b", linestyle = "--", label = f"Measured T0T4 TOF:\n{measured_tof_t0t4_mean:.2f} +/- {measured_tof_t0t4_error:.1e} ns")
                ax.axhspan(measured_tof_t0t4_mean - measured_tof_t0t4_error, measured_tof_t0t4_mean + measured_tof_t0t4_error, color = "g", alpha = 0.2)


                extrapolated_mean_mom, extrapolated_err_mom = self.extrapolate_momentum(initial_momentum_th, T0T1_TOF_th, measured_tof_mean, measured_tof_error)
                extrapolated_mean_final_mom, extrapolated_err_final_mom = self.extrapolate_momentum(final_momentum_th, T0T1_TOF_th, measured_tof_mean, measured_tof_error)
                
                extrapolated_mean_mom_t0t4, extrapolated_err_mom_t0t4 = self.extrapolate_momentum(initial_momentum_th, T0T4_TOF_th, measured_tof_t0t4_mean, measured_tof_t0t4_error)
                extrapolated_mean_final_mom_t0t4, extrapolated_err_final_mom_t0t4 = self.extrapolate_momentum(final_momentum_th, T0T4_TOF_th, measured_tof_t0t4_mean, measured_tof_t0t4_error)

                ax.axvline(extrapolated_mean_mom, color = "green", linestyle = "-.", label = f"T0T1 TOF-est. momentum \n Initial: {extrapolated_mean_mom:.1f} +/- {extrapolated_err_mom:.1f} MeV/c \n Final: {extrapolated_mean_final_mom:.1f} +/- {extrapolated_err_final_mom:.1f} MeV/c")
                
                ax.axvline(extrapolated_mean_mom_t0t4, color = "blue", linestyle = "-.", label = f"T0T4 TOF-est. momentum \n Initial: {extrapolated_mean_mom_t0t4:.1f} +/- {extrapolated_err_mom_t0t4:.1f} MeV/c \n Final: {extrapolated_mean_final_mom_t0t4:.1f} +/- {extrapolated_err_final_mom_t0t4:.1f} MeV/c")
                

                ax.set_ylabel("TOF (ns)", fontsize = 18)
                ax.set_xlabel("Initial momentum (MeV/c)", fontsize = 18)
                ax.legend(fontsize = 12)
                ax.grid()
                ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) - {particle}", fontsize = 20)

                self.pdf_global.savefig(fig)
                plt.close()

                self.particle_mom_mean[particles_tof_names[particle]] = extrapolated_mean_mom
                self.particle_mom_mean_err[particles_tof_names[particle]] = extrapolated_err_mom
                self.particle_mom_final_mean[particles_tof_names[particle]] = extrapolated_mean_final_mom
                self.particle_mom_final_mean_err[particles_tof_names[particle]] = extrapolated_err_final_mom
          
        print("Initial momentum reconstructed: ", self.particle_mom_mean)

            
        
    def estimate_momentum(self, verbose = False):
        '''This function is estimating the momentum for each trigger based on the pre-computed PID and the material budget in the beam line. It attemps at calculating errors as well, this function is deprecated, see estimate_particle_momentum instead (cleaner and using an updated TOF estimation)'''
        
        momentum_points = np.linspace(170, 1900, 56)
        
        #give_tof gives the nominal time of flight for particles of this type in the given momentum range
        #We then compare the measured TOF with those theoretical values to retrieve the momentum
        #The muon momentum is those after the beam window
        init_mom, pion_mom, pion_tof = self.give_tof( "Pions", momentum_points,  self.n_eveto, self.n_tagger, self.there_is_ACT5, self.run_momentum, "T0T1", verbose = verbose)
        
        init_mom, muon_mom, muon_tof = self.give_tof( "Muons", momentum_points,  self.n_eveto, self.n_tagger, self.there_is_ACT5, self.run_momentum, "T0T1", verbose = False)
        
  
        ##### Now extrapolate the mean momentum of muons and pion
        #Step 1 plot
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.plot(momentum_points, pion_tof, color = "g", marker = "+", label = "Expected pion TOF")
        ax.plot(momentum_points, muon_tof, color = "orange", marker = "x", label = "Expected muon TOF")
        
        
        #read in the measured TOF
        pion_tof_m = self.particle_tof_mean["pion"]
        pion_tof_error = self.particle_tof_eom["pion"]
        pion_tof_std = self.particle_tof_std["pion"]
        muon_tof_m = self.particle_tof_mean["muon"]
        muon_tof_error = self.particle_tof_eom["muon"]
        muon_tof_std = self.particle_tof_std["muon"]
        
        #Plot the measured TOF
        ax.axhline(pion_tof_m, color = "g", linestyle = "--", label = f"Measured pions TOF: {pion_tof_m:.2f} +/- {pion_tof_error:.1e} ns (EOM)  +/- {pion_tof_std:.1e} ns (std)")
        ax.axhspan(pion_tof_m - pion_tof_error, pion_tof_m + pion_tof_error, color = "g", alpha = 0.2)
        
        ax.axhspan(pion_tof_m - pion_tof_std, pion_tof_m + pion_tof_std, color = "g", alpha = 0.1)
        
        ax.axhline(muon_tof_m, color = "orange", linestyle = "--", label = f"Measured muons TOF: {muon_tof_m:.2f} +/- {muon_tof_error:.1e} ns (EOM) +/- {muon_tof_std:.1e} ns (std)")
        ax.axhspan(muon_tof_m - muon_tof_error, muon_tof_m + muon_tof_error, color = "orange", alpha = 0.2)
        ax.axhspan(muon_tof_m - muon_tof_std, muon_tof_m + muon_tof_std, color = "orange", alpha = 0.1)

        #Extrapolate each momenta at the start !
        mean_mom_pion, err_mom_pion = self.extrapolate_momentum(momentum_points, pion_tof, pion_tof_m, pion_tof_error)
#         _, std_mom_pion = self.extrapolate_momentum(momentum_points, pion_tof, pion_tof_std, pion_tof_error)
        mean_mom_muon, err_mom_muon = self.extrapolate_momentum(momentum_points, muon_tof, muon_tof_m, muon_tof_error)
#         _, std_mom_muon = self.extrapolate_momentum(momentum_points, muon_tof, muon_tof_m, muon_tof_std)
        
        
        mean_mom_pion_final, err_mom_pion_final = self.extrapolate_momentum(pion_mom, pion_tof, pion_tof_m, pion_tof_error)
        mean_mom_muon_final, err_mom_muon_final = self.extrapolate_momentum(muon_mom, muon_tof, muon_tof_m, muon_tof_error)

        
         #plot the extrapolated momenta
        ax.axvline(mean_mom_pion, color = "green", linestyle = "-.", label = f"Estimated mean pion momentum: {mean_mom_pion:.2f} +/- {err_mom_pion:.1f} MeV/c (EOM)")
        ax.axvspan(mean_mom_pion - err_mom_pion, mean_mom_pion + err_mom_pion, color = "green", alpha = 0.2)
#         ax.axvspan(std_mom_pion - std_mom_pion, mean_mom_pion + std_mom_pion, color = "green", alpha = 0.1)
        ax.axvline(mean_mom_muon, color = "orange", linestyle = "-.", label = f"Estimated mean muon momentum: {mean_mom_muon:.2f} +/- {err_mom_muon:.1f} MeV/c  (EOM)")
        ax.axvspan(mean_mom_muon - err_mom_muon, mean_mom_muon + err_mom_muon, color = "orange", alpha = 0.2)
#         ax.axvspan(mean_mom_muon - std_mom_muon, mean_mom_muon + std_mom_muon, color = "orange", alpha = 0.1)
        
        #make a clean plot
        ax.set_ylabel("TOF (ns)", fontsize = 18)
        ax.set_xlabel("Initial momentum", fontsize = 18)
        ax.legend(fontsize = 12)
        ax.grid()
        try:
            ax.set_ylim(min(muon_tof) * 0.98,max(pion_tof) * 1.04)
        except:
            print("")
                
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)

        self.pdf_global.savefig(fig)
        plt.close()
        
        
        self.particle_mom_mean = {
            "electron": 0,
            "muon": mean_mom_muon,
            "pion": mean_mom_pion,
            "proton": 0,
            "deuteron":0,
            "helium3":0,
            "tritium":0,
            "lithium6":0
            
        }
        
        self.particle_mom_mean_err = {
            "electron": 0,
            "muon": err_mom_muon,
            "pion": err_mom_pion,
            "proton": 0,
            "deuteron":0,
            "helium3":0,
            "tritium":0,
            "lithium6":0
        }
        
        
        self.particle_mom_final_mean = {
            "electron": 0,
            "muon": mean_mom_muon_final,
            "pion": mean_mom_pion_final,
            "proton": 0,
            "deuteron":0,
            "helium3":0,
            "tritium":0,
            "lithium6":0
            
        }
        
        self.particle_mom_final_mean_err = {
            "electron": 0,
            "muon": err_mom_muon_final,
            "pion": err_mom_pion_final,
            "proton": 0,
            "deuteron":0,
            "helium3":0,
            "tritium":0,
            "lithium6":0
        }
     
    

        if sum(self.df["is_proton"]==True) > 20 :
            
            momentum_points_proton = np.linspace(self.run_momentum * 0.7, 1900,48)
            #get the nominal TOF for protons
            init_mom, proton_mom, proton_tof = self.give_tof( "Protons", momentum_points_proton,  self.n_eveto, self.n_tagger, self.there_is_ACT5, self.run_momentum, "T0T1", verbose = False)
            
        

            fig, ax = plt.subplots(figsize = (8, 6))
            ax.plot(momentum_points_proton, proton_tof, color = "r", marker = "+", label = "Expected proton TOF")

            #read in the measured TOF
            proton_tof_m = self.particle_tof_mean["proton"]
            proton_tof_error = self.particle_tof_eom["proton"]


            #Plot the measured TOF
            ax.axhline(proton_tof_m, color = "r", linestyle = "--", label = f"Measured proton TOF: {proton_tof_m:.2f} +/- {proton_tof_error:.1e} ns (EOM)")
            ax.axhspan(proton_tof_m - proton_tof_error, proton_tof_m + proton_tof_error, color = "g", alpha = 0.2)


            #Extrapolate each momenta
            mean_mom_proton, err_mom_proton = self.extrapolate_momentum(momentum_points_proton, proton_tof, proton_tof_m, proton_tof_error)
            
            #we consider that we have the same error at the end than at the begining
            mean_mom_proton_final, err_mom_proton_final = self.extrapolate_momentum(proton_mom, proton_tof, proton_tof_m, proton_tof_error)

             #plot the extrapolated momenta
            ax.axvline(mean_mom_proton, color = "red", linestyle = "-.", label = f"Estimated mean proton momentum: {mean_mom_proton:.2f} +/- {err_mom_proton:.1f} MeV/c")
            ax.axvspan(mean_mom_proton - err_mom_proton, mean_mom_proton + err_mom_proton, color = "red", alpha = 0.2)

            #make a clean plot
            ax.set_ylabel("TOF (ns)", fontsize = 18)
            ax.set_xlabel("Initial momentum", fontsize = 18)
            ax.legend(fontsize = 12)
            ax.grid()
            try:
                ax.set_ylim(min(proton_tof) * 0.98,max(proton_tof) * 1.04)
            except:
                print("")
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
            
            self.pdf_global.savefig(fig)
            plt.close()

            self.particle_mom_mean["proton"] = mean_mom_proton
            self.particle_mom_mean_err["proton"] = err_mom_proton
            
            self.particle_mom_final_mean["proton"] = mean_mom_proton_final
            self.particle_mom_final_mean_err["proton"] = err_mom_proton_final
            
            
            ################## if we have helium3, it will be around the same TOF
            if  sum(self.df["is_helium3"]) > 20:
            
                #we cannot go too low momentum, otherwise unphysical... 
                momentum_points_helium3 = np.linspace(1500, 1900, 12)

                init_mom, helium3_mom, helium3_tof = self.give_tof( "Helium3", momentum_points_helium3,  self.n_eveto, self.n_tagger, self.there_is_ACT5, self.run_momentum, "T0T1", verbose = False)

                fig, ax = plt.subplots(figsize = (8, 6))
                ax.plot(momentum_points_helium3, helium3_tof, color = "k", marker = "+", label = "Expected helium3 TOF")

                #read in the measured TOF
                helium3_tof_m = self.particle_tof_mean["helium3"]
                helium3_tof_error = self.particle_tof_eom["helium3"]


                #Plot the measured TOF
                ax.axhline(helium3_tof_m, color = "k", linestyle = "--", label = f"Measured helium3 TOF: {helium3_tof_m:.2f} +/- {helium3_tof_error:.1e} ns (EOM)")

                ax.axhspan(helium3_tof_m - helium3_tof_error, helium3_tof_m + helium3_tof_error, color = "g", alpha = 0.2)

                #Extrapolate each momenta
                mean_mom_helium3, err_mom_helium3 = self.extrapolate_momentum(momentum_points_helium3, helium3_tof, helium3_tof_m, helium3_tof_error)
                mean_mom_helium3_final, err_mom_helium3_final = self.extrapolate_momentum(helium3_mom, helium3_tof, helium3_tof_m, helium3_tof_error)

                 #plot the extrapolated momenta
                ax.axvline(mean_mom_helium3, color = "black", linestyle = "-.", label = f"Estimated mean helium3 momentum: {mean_mom_helium3:.2f} +/- {err_mom_helium3:.1f} MeV/c")
                ax.axvspan(mean_mom_helium3 - err_mom_helium3, mean_mom_helium3 + err_mom_helium3, color = "black", alpha = 0.2)

              

                self.particle_mom_mean["helium3"] = mean_mom_helium3
                self.particle_mom_mean_err["helium3"] = err_mom_helium3

                self.particle_mom_final_mean["helium3"] = mean_mom_helium3_final
                self.particle_mom_final_mean_err["helium3"] = err_mom_helium3_final
                
                
                #make a clean plot
                ax.set_ylabel("TOF (ns)", fontsize = 18)
                ax.set_xlabel("Initial momentum", fontsize = 18)
                ax.legend(fontsize = 12)
                ax.grid()
#                 ax.set_ylim(min(helium3_tof_m) * 0.98,max(momentum_points_helium3) * 1.04)
                ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)

                self.pdf_global.savefig(fig)
                plt.close()



        
        if sum(self.df["is_deuteron"]==True) > 20 :
            
            #we cannot go too low momentum, otherwise unphysical... 
            momentum_points_deuteron = np.linspace(0.75 * abs(self.run_momentum), 1.4 * abs(self.run_momentum),12)
            
            init_mom, deuteron_mom, deuteron_tof = self.give_tof( "Deuteron", momentum_points_deuteron,  self.n_eveto, self.n_tagger, self.there_is_ACT5, self.run_momentum, "T0T1", verbose = False)
        
            fig, ax = plt.subplots(figsize = (8, 6))
            ax.plot(momentum_points_deuteron, deuteron_tof, color = "k", marker = "+", label = "Expected deuteron TOF")

            #read in the measured TOF
            deuteron_tof_m = self.particle_tof_mean["deuteron"]
            deuteron_tof_error = self.particle_tof_eom["deuteron"]


            #Plot the measured TOF
            ax.axhline(deuteron_tof_m, color = "k", linestyle = "--", label = f"Measured deuteron TOF: {deuteron_tof_m:.2f} +/- {deuteron_tof_error:.1e} ns (EOM)")
            
            ax.axhspan(deuteron_tof_m - deuteron_tof_error, deuteron_tof_m + deuteron_tof_error, color = "g", alpha = 0.2)


            #Extrapolate each momenta
            mean_mom_deuteron, err_mom_deuteron = self.extrapolate_momentum(momentum_points_deuteron, deuteron_tof, deuteron_tof_m, deuteron_tof_error)
            mean_mom_deuteron_final, err_mom_deuteron_final = self.extrapolate_momentum(deuteron_mom, deuteron_tof, deuteron_tof_m, deuteron_tof_error)

             #plot the extrapolated momenta
            ax.axvline(mean_mom_deuteron, color = "black", linestyle = "-.", label = f"Estimated mean deuteron momentum: {mean_mom_deuteron:.2f} +/- {err_mom_deuteron:.1f} MeV/c")
            ax.axvspan(mean_mom_deuteron - err_mom_deuteron, mean_mom_deuteron + err_mom_deuteron, color = "black", alpha = 0.2)

            #make a clean plot
            ax.set_ylabel("TOF (ns)", fontsize = 18)
            ax.set_xlabel("Initial momentum", fontsize = 18)
            ax.legend(fontsize = 12)
            ax.grid()
            try:
                ax.set_ylim(min(deuteron_tof) * 0.98,max(deuteron_tof) * 1.04)
            except:
                print("")
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)

            self.pdf_global.savefig(fig)
            plt.close()

            self.particle_mom_mean["deuteron"] = mean_mom_deuteron
            self.particle_mom_mean_err["deuteron"] = err_mom_deuteron
            
            self.particle_mom_final_mean["deuteron"] = mean_mom_deuteron_final
            self.particle_mom_final_mean_err["deuteron"] = err_mom_deuteron_final

        print(f"The estimated mean particle momenta are {self.particle_mom_mean} MeV/c with an error {self.particle_mom_mean_err} MeV/c")
        
        
        if  self.particle_tof_mean["tritium"] != 0:
            
            #we cannot go too low momentum, otherwise unphysical... 
            momentum_points_tritium = np.linspace(0.4 * abs(self.run_momentum), 1.4 * abs(self.run_momentum),12)

            init_mom, tritium_mom, tritium_tof = self.give_tof( "Tritium", momentum_points_tritium,  self.n_eveto, self.n_tagger, self.there_is_ACT5, self.run_momentum, "T0T1", verbose = False)

            fig, ax = plt.subplots(figsize = (8, 6))
            ax.plot(momentum_points_tritium, tritium_tof, color = "k", marker = "+", label = "Expected tritium TOF")

            #read in the measured TOF
            tritium_tof_m = self.particle_tof_mean["tritium"]
            tritium_tof_error = self.particle_tof_eom["tritium"]


            #Plot the measured TOF
            ax.axhline(tritium_tof_m, color = "k", linestyle = "--", label = f"Measured tritium TOF: {tritium_tof_m:.2f} +/- {tritium_tof_error:.1e} ns (EOM)")

            ax.axhspan(tritium_tof_m - tritium_tof_error, tritium_tof_m + tritium_tof_error, color = "g", alpha = 0.2)

            #Extrapolate each momenta
            mean_mom_tritium, err_mom_tritium = self.extrapolate_momentum(momentum_points_tritium, tritium_tof, tritium_tof_m, tritium_tof_error)
            mean_mom_tritium_final, err_mom_tritium_final = self.extrapolate_momentum(tritium_mom, tritium_tof, tritium_tof_m, tritium_tof_error)

             #plot the extrapolated momenta
            ax.axvline(mean_mom_tritium, color = "black", linestyle = "-.", label = f"Estimated mean tritium momentum: {mean_mom_tritium:.2f} +/- {err_mom_tritium:.1f} MeV/c")
            ax.axvspan(mean_mom_tritium - err_mom_tritium, mean_mom_tritium + err_mom_tritium, color = "black", alpha = 0.2)



            self.particle_mom_mean["tritium"] = mean_mom_tritium
            self.particle_mom_mean_err["tritium"] = err_mom_tritium

            self.particle_mom_final_mean["tritium"] = mean_mom_tritium_final
            self.particle_mom_final_mean_err["tritium"] = err_mom_tritium_final


            #make a clean plot
            ax.set_ylabel("TOF (ns)", fontsize = 18)
            ax.set_xlabel("Initial momentum", fontsize = 18)
            ax.legend(fontsize = 12)
            ax.grid()
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)

            self.pdf_global.savefig(fig)
            plt.close()
            
            
        ################## if we have helium3, it will be around the same TOF
        if  self.particle_tof_mean["lithium6"] != 0:
            #we cannot go too low momentum, otherwise unphysical... 
            momentum_points_lithium6 = np.linspace(2000, 5000, 12)

            init_mom, lithium6_mom, lithium6_tof = self.give_tof( "Lithium6", momentum_points_lithium6,  self.n_eveto, self.n_tagger, self.there_is_ACT5, self.run_momentum, "T0T1", verbose = False)

            fig, ax = plt.subplots(figsize = (8, 6))
            ax.plot(momentum_points_lithium6, lithium6_tof, color = "k", marker = "+", label = "Expected lithium6 TOF")

            #read in the measured TOF
            lithium6_tof_m = self.particle_tof_mean["lithium6"]
            lithium6_tof_error = self.particle_tof_eom["lithium6"]


            #Plot the measured TOF
            ax.axhline(lithium6_tof_m, color = "k", linestyle = "--", label = f"Measured lithium6 TOF: {lithium6_tof_m:.2f} +/- {lithium6_tof_error:.1e} ns (EOM)")

            ax.axhspan(lithium6_tof_m - lithium6_tof_error, lithium6_tof_m + lithium6_tof_error, color = "g", alpha = 0.2)

            #Extrapolate each momenta
            mean_mom_lithium6, err_mom_lithium6 = self.extrapolate_momentum(momentum_points_lithium6, lithium6_tof, lithium6_tof_m, lithium6_tof_error)
            mean_mom_lithium6_final, err_mom_lithium6_final = self.extrapolate_momentum(lithium6_mom, lithium6_tof, lithium6_tof_m, lithium6_tof_error)

             #plot the extrapolated momenta
            ax.axvline(mean_mom_lithium6, color = "black", linestyle = "-.", label = f"Estimated mean lithium6 momentum: {mean_mom_lithium6:.2f} +/- {err_mom_lithium6:.1f} MeV/c")
            ax.axvspan(mean_mom_lithium6 - err_mom_lithium6, mean_mom_lithium6 + err_mom_lithium6, color = "black", alpha = 0.2)



            self.particle_mom_mean["lithium6"] = mean_mom_lithium6
            self.particle_mom_mean_err["lithium6"] = err_mom_lithium6

            self.particle_mom_final_mean["lithium6"] = mean_mom_lithium6_final
            self.particle_mom_final_mean_err["lithium6"] = err_mom_lithium6_final


            #make a clean plot
            ax.set_ylabel("TOF (ns)", fontsize = 18)
            ax.set_xlabel("Initial momentum", fontsize = 18)
            ax.legend(fontsize = 12)
            ax.grid()
#                 ax.set_ylim(min(helium3_tof_m) * 0.98,max(momentum_points_helium3) * 1.04)
            ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)

            self.pdf_global.savefig(fig)
            plt.close()
        
        
        
        
        ######### Now calculate the momentum for each trigger individually
        #The error on the tof is taken as the width of the electron tof distribution 
              
        self.df["initial_momentum"] = np.nan  # initialize column
        self.df["initial_momentum_error"] = np.nan  # initialize column

        mask_muon = self.df["is_muon"] == 1
        mask_pion = self.df["is_pion"] == 1
        mask_proton = self.df["is_proton"] == 1
        mask_deuteron = self.df["is_deuteron"] == 1
        mask_helium3 = self.df["is_helium3"] == 1
        
        #this is the error assumed on each TOF
        electron_tof_std = self.particle_tof_std["electron"]

        #Store the correct trigger TOFs, don't forget to use the corrected one, 
        #otherwise non-physical results due to offset of electron TOF with L/c
        trigger_muon_tof = self.df.loc[mask_muon, "tof_corr"].to_numpy()
        trigger_pion_tof = self.df.loc[mask_pion, "tof_corr"].to_numpy()
        trigger_proton_tof = self.df.loc[mask_proton, "tof_corr"].to_numpy()
        trigger_deuteron_tof = self.df.loc[mask_deuteron, "tof_corr"].to_numpy()
        trigger_helium3_tof = self.df.loc[mask_helium3, "tof_corr"].to_numpy()
        
        #now compute the momentum for each trigger based on the TOF that is measured for each of them 
        mom, mom_err = self.extrapolate_trigger_momentum(momentum_points, muon_tof, trigger_muon_tof, electron_tof_std)
        self.df.loc[mask_muon, "initial_momentum"] = mom
        self.df.loc[mask_muon, "initial_momentum_error"] = mom_err
        
        #now extrapolating the momentum after the WCTE beam pipe
        mom, mom_err = self.extrapolate_trigger_momentum(muon_mom, muon_tof, trigger_muon_tof, electron_tof_std)
        self.df.loc[mask_muon, "final_momentum"] = mom
        self.df.loc[mask_muon, "final_momentum_error"] = mom_err
        
        
        
        mom, mom_err = self.extrapolate_trigger_momentum(momentum_points, pion_tof, trigger_pion_tof, electron_tof_std)
        self.df.loc[mask_pion, "initial_momentum"] = mom
        self.df.loc[mask_pion, "initial_momentum_error"] = mom_err
        
        
        mom, mom_err = self.extrapolate_trigger_momentum(pion_mom, pion_tof, trigger_pion_tof, electron_tof_std)
        self.df.loc[mask_pion, "final_momentum"] = mom
        self.df.loc[mask_pion, "final_momentum_error"] = mom_err

        
        if sum(self.df["is_proton"]) > 20:
            mom, mom_err = self.extrapolate_trigger_momentum(momentum_points_proton, proton_tof, trigger_proton_tof, electron_tof_std)

            self.df.loc[mask_proton, "initial_momentum"] = mom
            self.df.loc[mask_proton, "initial_momentum_error"] = mom_err
            
            mom, mom_err = self.extrapolate_trigger_momentum(proton_mom, proton_tof, trigger_proton_tof, electron_tof_std)

            self.df.loc[mask_proton, "final_momentum"] = mom
            self.df.loc[mask_proton, "final_momentum_error"] = mom_err
            
        if sum(self.df["is_deuteron"]) > 20:
            mom, mom_err = self.extrapolate_trigger_momentum(momentum_points_deuteron, deuteron_tof, trigger_deuteron_tof, electron_tof_std)

            self.df.loc[mask_deuteron, "initial_momentum"] = mom
            self.df.loc[mask_deuteron, "initial_momentum_error"] = mom_err
            
            
            mom, mom_err = self.extrapolate_trigger_momentum(deuteron_mom, deuteron_tof, trigger_deuteron_tof, electron_tof_std)

            self.df.loc[mask_deuteron, "final_momentum"] = mom
            self.df.loc[mask_deuteron, "final_momentum_error"] = mom_err
            
            
            
        if sum(self.df["is_helium3"]) > 20:
            mom, mom_err = self.extrapolate_trigger_momentum(momentum_points_helium3, helium3_tof, trigger_helium3_tof, electron_tof_std)

            self.df.loc[mask_helium3, "initial_momentum"] = mom
            self.df.loc[mask_helium3, "initial_momentum_error"] = mom_err
            
            
            mom, mom_err = self.extrapolate_trigger_momentum(helium3_mom, helium3_tof, trigger_helium3_tof, electron_tof_std)

            self.df.loc[mask_helium3, "final_momentum"] = mom
            self.df.loc[mask_helium3, "final_momentum_error"] = mom_err
            
            
            
        print("\n \n", self.df.head())
        
        #### Next, plot the reconstructed momenta (without error for each of the triggers, this will help chekch the overlap)
        
        fig, ax = plt.subplots(figsize = (8, 6))
        momentum_bins = np.linspace(min(momentum_points) * 0.6, max(momentum_points) * 1.4, 100)
        
        n_mu = sum(self.df["is_muon"])
        n_pi = sum(self.df["is_pion"])
        n_p = sum(self.df["is_proton"])
        n_D = sum(self.df["is_deuteron"])
        n_He3 = sum(self.df["is_helium3"])
        
        
        f_mu = n_mu/len(self.df["is_muon"])
        f_pi = n_pi/len(self.df["is_pion"])
        
        
        #muons
        ax.hist(self.df["initial_momentum"][self.df["is_muon"] == 1], 
                bins = momentum_bins, color = "orange", 
                histtype = "step",
                label = f"Muons {n_mu} triggers ({f_mu * 100:.1f}% of particles) p = {mean_mom_muon:.2f} +/- {err_mom_muon:.1f} MeV/c")
        
        ax.axvline(self.particle_mom_mean["muon"], color = "orange", linestyle = "-.")
        ax.axvspan(self.particle_mom_mean["muon"] - self.particle_mom_mean_err["muon"], self.particle_mom_mean["muon"] + self.particle_mom_mean_err["muon"], color = "orange", alpha = 0.2)
        
        
        #Pions
        ax.hist(
            self.df["initial_momentum"][self.df["is_pion"] == 1],
            bins=momentum_bins,
            color="blue",
            histtype="step",
            label=f"Pions {n_pi} triggers ({f_pi * 100:.1f}% of particles) p = {self.particle_mom_mean['pion']:.2f} +/- {self.particle_mom_mean_err['pion']:.1f} MeV/c"
        )
        ax.axvline(
            self.particle_mom_mean["pion"],
            color="blue",
            linestyle="-.",
           
        )
        ax.axvspan(
            self.particle_mom_mean["pion"] - self.particle_mom_mean_err["pion"],
            self.particle_mom_mean["pion"] + self.particle_mom_mean_err["pion"],
            color="blue",
            alpha=0.2
        )
        
        
        #Protons:
        if sum(self.df["is_proton"]) > 20:
            f_p = n_p/len(self.df["is_proton"])
            ax.hist(
                self.df["initial_momentum"][self.df["is_proton"] == 1],
                bins=momentum_bins,
                color="red",
                histtype="step",
                label=f"Protons: {n_p} triggers ({f_p * 100:.1f}% of particles) p = {self.particle_mom_mean['proton']:.2f} +/- {self.particle_mom_mean_err['proton']:.1f} MeV/c"
            )
            ax.axvline(
                self.particle_mom_mean["proton"],
                color="red",
                linestyle="-.",
            )
            ax.axvspan(
                self.particle_mom_mean["proton"] - self.particle_mom_mean_err["proton"],
                self.particle_mom_mean["proton"] + self.particle_mom_mean_err["proton"],
                color="red",
                alpha=0.2
            )
            
            
        #Deuterons:
        if sum(self.df["is_deuteron"]) > 20:
            f_D = n_D/len(self.df["is_deuteron"])
            ax.hist(
                self.df["initial_momentum"][self.df["is_deuteron"] == 1],
                bins=momentum_bins,
                color="black",
                histtype="step",
                label=f"Deuterons: {n_D} triggers ({f_D * 100:.1f}% of particles) p = {self.particle_mom_mean['deuteron']:.2f} +/- {self.particle_mom_mean_err['deuteron']:.1f} MeV/c"
            )
            
#             print("Reconstructed momenta for deuterons", self.df["initial_momentum"][self.df["is_deuteron"] == 1])
            ax.axvline(
                self.particle_mom_mean["deuteron"],
                color="black",
                linestyle="-.",
            )
            ax.axvspan(
                self.particle_mom_mean["deuteron"] - self.particle_mom_mean_err["deuteron"],
                self.particle_mom_mean["deuteron"] + self.particle_mom_mean_err["deuteron"],
                color="black",
                alpha=0.2
            )
            
            
        #helium3:
        if sum(self.df["is_helium3"]) > 20:
            f_He3 = n_He3/len(self.df["is_helium3"])
            ax.hist(
                self.df["initial_momentum"][self.df["is_helium3"] == 1],
                bins=momentum_bins,
                color="green",
                histtype="step",
                label=f"Helium 3 nuclei: {n_He3} triggers ({f_He3 * 100:.2f}% of particles) p = {self.particle_mom_mean['helium3']:.2f} +/- {self.particle_mom_mean_err['helium3']:.1f} MeV/c"
            )
            
#             print("Reconstructed momenta for deuterons", self.df["initial_momentum"][self.df["is_helium3"] == 1])
            ax.axvline(
                self.particle_mom_mean["helium3"],
                color="green",
                linestyle="-.",
            )
            ax.axvspan(
                self.particle_mom_mean["helium3"] - self.particle_mom_mean_err["helium3"],
                self.particle_mom_mean["helium3"] + self.particle_mom_mean_err["helium3"],
                color="green",
                alpha=0.2
            )
        
        #Add a caption to the figure
        plt.figtext(0.5, -0.05, f"Mean momentum for each particle type is obtained from mean TOF and error on mean (EOM).", 
                    ha="center", va="top", fontsize=10)
       
        
        #make a clean plot
        ax.set_ylabel("Number of events", fontsize = 18)
        ax.set_xlabel("Initial momentum (MeV/c)", fontsize = 18)
        ax.legend(fontsize = 10)
        ax.grid()
        ax.set_yscale("log")
        ax.set_ylim(0.5, len(self.df["is_electron"]) * 20)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
        
        self.pdf_global.savefig(fig)
        
        try:
            ax.set_xlim(self.particle_mom_mean["muon"] * 0.4, self.particle_mom_mean["pion"] * 1.6)
        except:
            print("")
        
        
        if sum(self.df["is_proton"])>20:
            try:
                ax.set_xlim(self.particle_mom_mean["muon"] * 0.4, self.particle_mom_mean["proton"] * 1.6)
            except:
                print("")
            
        if sum(self.df["is_deuteron"])>20:
            try:
                ax.set_xlim(self.particle_mom_mean["muon"] * 0.4, self.particle_mom_mean["deuteron"] * 1.6)
            except:
                print("")
            
        

        self.pdf_global.savefig(fig)
        plt.close()
        
        
        
        ######## Momentum after beam window
        
        fig, ax = plt.subplots(figsize = (8, 6))
        #muons
        ax.hist(self.df["final_momentum"][self.df["is_muon"] == 1], 
                bins = momentum_bins, color = "orange", 
                histtype = "step",
                label = f"Muons {n_mu} triggers ({f_mu * 100:.1f}% of particles) p = {mean_mom_muon_final:.2f} +/- {err_mom_muon_final:.1f} MeV/c")
        
        ax.axvline(self.particle_mom_final_mean["muon"], color = "orange", linestyle = "-.")
        ax.axvspan(self.particle_mom_final_mean["muon"] - self.particle_mom_final_mean_err["muon"], self.particle_mom_final_mean["muon"] + self.particle_mom_final_mean_err["muon"], color = "orange", alpha = 0.2)
        
        
        #Pions
        ax.hist(
            self.df["final_momentum"][self.df["is_pion"] == 1],
            bins=momentum_bins,
            color="blue",
            histtype="step",
            label=f"Pions {n_pi} triggers ({f_pi * 100:.1f}% of particles) p = {self.particle_mom_final_mean['pion']:.2f} +/- {self.particle_mom_final_mean_err['pion']:.1f} MeV/c"
        )
        ax.axvline(
            self.particle_mom_final_mean["pion"],
            color="blue",
            linestyle="-.",
           
        )
        ax.axvspan(
            self.particle_mom_final_mean["pion"] - self.particle_mom_final_mean_err["pion"],
            self.particle_mom_final_mean["pion"] + self.particle_mom_final_mean_err["pion"],
            color="blue",
            alpha=0.2
        )
        
        
        #Protons:
        if sum(self.df["is_proton"]) > 20:
            f_p = n_p/len(self.df["is_proton"])
            ax.hist(
                self.df["final_momentum"][self.df["is_proton"] == 1],
                bins=momentum_bins,
                color="red",
                histtype="step",
                label=f"Protons: {n_p} triggers ({f_p * 100:.1f}% of particles) p = {self.particle_mom_final_mean['proton']:.2f} +/- {self.particle_mom_final_mean_err['proton']:.1f} MeV/c"
            )
            ax.axvline(
                self.particle_mom_final_mean["proton"],
                color="red",
                linestyle="-.",
            )
            ax.axvspan(
                self.particle_mom_final_mean["proton"] - self.particle_mom_final_mean_err["proton"],
                self.particle_mom_final_mean["proton"] + self.particle_mom_final_mean_err["proton"],
                color="red",
                alpha=0.2
            )
            
            
        #Deuterons:
        if sum(self.df["is_deuteron"]) > 20:
            f_D = n_D/len(self.df["is_deuteron"])
            ax.hist(
                self.df["final_momentum"][self.df["is_deuteron"] == 1],
                bins=momentum_bins,
                color="black",
                histtype="step",
                label=f"Deuterons: {n_D} triggers ({f_D * 100:.1f}% of particles) p = {self.particle_mom_final_mean['deuteron']:.2f} +/- {self.particle_mom_final_mean_err['deuteron']:.1f} MeV/c"
            )
            
            #print("Reconstructed momenta for deuterons", self.df["initial_momentum"][self.df["is_deuteron"] == 1])
            ax.axvline(
                self.particle_mom_final_mean["deuteron"],
                color="black",
                linestyle="-.",
            )
            ax.axvspan(
                self.particle_mom_final_mean["deuteron"] - self.particle_mom_final_mean_err["deuteron"],
                self.particle_mom_final_mean["deuteron"] + self.particle_mom_final_mean_err["deuteron"],
                color="black",
                alpha=0.2
            )
            
            
            #Helium3:
        if sum(self.df["is_helium3"]) > 20:
            f_He3 = n_He3/len(self.df["is_helium3"])
            ax.hist(
                self.df["final_momentum"][self.df["is_helium3"] == 1],
                bins=momentum_bins,
                color="green",
                histtype="step",
                label=f"Helium3 nuclei: {n_He3} triggers ({f_He3 * 100:.2f}% of particles) p = {self.particle_mom_final_mean['helium3']:.2f} +/- {self.particle_mom_final_mean_err['helium3']:.1f} MeV/c"
            )
            
            #print("Reconstructed momenta for deuterons", self.df["initial_momentum"][self.df["is_deuteron"] == 1])
            ax.axvline(
                self.particle_mom_final_mean["helium3"],
                color="green",
                linestyle="-.",
            )
            ax.axvspan(
                self.particle_mom_final_mean["helium3"] - self.particle_mom_final_mean_err["helium3"],
                self.particle_mom_final_mean["helium3"] + self.particle_mom_final_mean_err["helium3"],
                color="green",
                alpha=0.2
            )
            
        
        #Add a caption to the figure
        plt.figtext(0.5, -0.05, f"Mean momentum for each particle type is obtained from mean TOF and error on mean (EOM).", 
                    ha="center", va="top", fontsize=10)
       
        
        #make a clean plot
        ax.set_ylabel("Number of events", fontsize = 18)
        ax.set_xlabel("Momentum after beam window (MeV/c)", fontsize = 18)
        ax.legend(fontsize = 10)
        ax.grid()
        ax.set_yscale("log")
        ax.set_ylim(0.5, len(self.df["is_electron"]) * 20)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)

        self.pdf_global.savefig(fig)
        try:
                
            ax.set_xlim(self.particle_mom_final_mean["muon"] * 0.4, self.particle_mom_final_mean["pion"] * 1.6)
        except:
            print("")
        
        
        if  sum(self.df["is_proton"])>20:
            try:
                ax.set_xlim(self.particle_mom_final_mean["muon"] * 0.4, self.particle_mom_final_mean["proton"] * 1.6)
            except:
                print("")
            
        if sum(self.df["is_deuteron"])>20:
            try:
                ax.set_xlim(self.particle_mom_final_mean["muon"] * 0.4, self.particle_mom_final_mean["deuteron"] * 1.6)
            except:
                print("")
            
        self.pdf_global.savefig(fig)
        
        
        plt.close()
        
        
            
    def measure_particle_TOF(self):
        '''Measure the TOF for each of the particles accounting for any offsets between the electron TOF and L/c'''
        
        
        there_is_proton = False
        if sum(self.df["is_proton"]) > 20:
            there_is_proton = True
        #Define the bounds inside which we will attempt the fits 
        if self.run_momentum > 600:
            times_of_flight_min = [ 8, 5, -70]
            times_of_flight_max = [120, 50, 70 ]

        elif there_is_proton:
            times_of_flight_min = [ 8, 5, -70]
            times_of_flight_max = [60, 50, 70 ]

        else:
            
            times_of_flight_min = [ 8, 5, -70]
            times_of_flight_max = [25, 20, 70 ]
            
            
            
        ##### First do T0-T1 ###########
        time_of_flight = self.df["tof"]
        
        #Define the bins
        bins_tof = np.arange(times_of_flight_min[0], times_of_flight_max[0], 0.2)

        bin_centers = (bins_tof[1:] + bins_tof[:-1])/2
        
        #Fit the electron TOF
        electron_tof = time_of_flight[self.df["is_electron"] == 1]
        h, _ = np.histogram(electron_tof, bins = bins_tof)
        
    
        det_module = db.from_yaml("../include/wcte_beam_detectors.yaml")
        popt, pcov = fit_gaussian(h, bin_centers)
        L = det_module.distance_m("T0", "T1")*100 #need to be cm
        #L is in cm, c is in m.ns^-1
        t0 = L/(c * 10**2) - popt[1] #convert m.ns^-1 into cm.ns^-1
        
        
        
         ##### Second do T0-T4 ###########
        time_of_flight_t0t4 = self.df["tof_t0t4"]
        
       
        bins_tof_t0t4 = np.arange(times_of_flight_min[0]+10, times_of_flight_max[0]+10, 0.2)
        bin_centers_t0t4 = (bins_tof_t0t4[1:] + bins_tof_t0t4[:-1])/2
        
        #Fit the electron TOF
        electron_tof_t0t4 = time_of_flight_t0t4[self.df["is_electron"] == 1]
        h_t0t4, _ = np.histogram(electron_tof_t0t4, bins = bins_tof_t0t4)
        
        
        
        
        popt_t0t4, pcov_t0t4 = fit_gaussian(h_t0t4, bin_centers_t0t4)
        L_t0t4 = det_module.distance_m("T0", "T4")*100 #need to be cm
        t0_t0t4 = L_t0t4/(c * 10**2) - popt_t0t4[1] #convert m.ns^-1 into cm.ns^-1 
        
        fig, ax = plt.subplots()
        ax.hist(electron_tof_t0t4, bins = bins_tof_t0t4, histtype = "step")
        ax.grid
        ax.plot(bins_tof_t0t4, gaussian(bins_tof_t0t4, popt_t0t4[0], popt_t0t4[1], popt_t0t4[2]), "--", color = "k")
        ax.set_title("Electron T0-T4 TOF", weight = "bold")
        ax.set_xlabel("T0-T4 TOF (ns)")
        self.pdf_global.savefig(fig)
        

        
        bins_tof_t0t4 += t0_t0t4
        bin_centers_t0t4 += t0_t0t4
        
        
        print(f"The time difference between the reconstructed electron TOF and L/c = {L/(c * 10**2):.2f} is {t0:.2f} ns")
        print(f"The time difference between the reconstructed electron TOF and L/c (T0T4) = {L_t0t4/(c * 10**2):.2f} is {t0_t0t4:.2f} ns")
        
        #Correct the TOF by this offset, decide to make a new column, cleaner
        self.df["tof_corr"] = self.df["tof"] + t0
        self.df["tof_t0t4_corr"] = self.df["tof_t0t4"] + t0_t0t4
        
        #Check TOF for each particle type
        h_mu, _ = np.histogram(self.df["tof_corr"][self.df["is_muon"]==1], bins = bins_tof)
        popt_mu, pcov = fit_gaussian(h_mu, bin_centers)
        
        h_pi, _ = np.histogram(self.df["tof_corr"][self.df["is_pion"]==1], bins = bins_tof)
        popt_pi, pcov = fit_gaussian(h_pi, bin_centers)
        
        h_mu_t0t4, _ = np.histogram(self.df["tof_t0t4_corr"][self.df["is_muon"]==1], bins = bins_tof_t0t4)
        
       
        popt_mu_t0t4, pcov_t0t4 = fit_gaussian(h_mu_t0t4, bin_centers_t0t4)
        
        h_pi_t0t4, _ = np.histogram(self.df["tof_t0t4_corr"][self.df["is_pion"]==1], bins = bins_tof_t0t4)
        popt_pi_t0t4, pcov_t0t4 = fit_gaussian(h_pi_t0t4, bin_centers_t0t4)
        
        if there_is_proton:
            h_p, _ = np.histogram(self.df["tof_corr"][self.df["is_proton"]==1], bins = bins_tof)
            popt_p, pcov = fit_gaussian(h_p, bin_centers)
            
            h_p_t0t4, _ = np.histogram(self.df["tof_t0t4_corr"][self.df["is_proton"]==1], bins = bins_tof_t0t4)
            try:
                popt_p_t0t4, pcov_t0t4 = fit_three_gaussians(h_p_t0t4, bin_centers_t0t4)
            except: 
                popt_p_t0t4 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            
        if sum(self.df["is_deuteron"])>20:
            h_D, _ = np.histogram(self.df["tof_corr"][self.df["is_deuteron"]==1], bins = bins_tof)
            popt_D, pcov = fit_gaussian(h_D, bin_centers)
            
            h_D_t0t4, _ = np.histogram(self.df["tof_t0t4_corr"][self.df["is_deuteron"]==1], bins = bins_tof_t0t4)
            popt_D_t0t4, pcov_t0t4 = fit_gaussian(h_D_t0t4, bin_centers_t0t4)
            
            
        #Here, plot the TOF 
        fig, ax = plt.subplots(figsize = (8, 6))
        
        #plot the distributions
        ax.hist(self.df["tof_corr"][self.df["is_electron"]==1], bins = bins_tof, histtype = "step", label = f"Electrons: tof = {popt[1]+t0:.2f} "+ r"$\pm$"+ f" {popt[2]:.2f} ns")  
        ax.hist(self.df["tof_corr"][self.df["is_muon"]==1], bins = bins_tof, histtype = "step", label = f"Muons: tof = {popt_mu[1]:.2f} "+ r"$\pm$"+ f" {popt_mu[2]:.2f} ns")
        ax.hist(self.df["tof_corr"][self.df["is_pion"]==1], bins = bins_tof, histtype = "step", label = f"Pions: tof = {popt_pi[1]:.2f} "+ r"$\pm$"+ f" {popt_pi[2]:.2f} ns")
        
        
        if there_is_proton:
            ax.hist(self.df["tof_corr"][self.df["is_proton"]==1], bins = bins_tof, histtype = "step", label = f"Protons: tof = {popt_p[1]:.2f} "+ r"$\pm$"+ f" {popt_p[2]:.2f} ns")
            
        if sum(self.df["is_deuteron"])>20:
            ax.hist(self.df["tof_corr"][self.df["is_deuteron"]==1], bins = bins_tof, histtype = "step", label = f"Deuterons: tof = {popt_D[1]:.2f} "+ r"$\pm$"+ f" {popt_D[2]:.2f} ns")
            
            
            
        
            
        if sum(self.df["is_helium3"])>20:
            
            try:
                h_He3, _ = np.histogram(self.df["tof_corr"][self.df["is_helium3"]==1], bins = bins_tof)
                popt_He3, pcov = fit_gaussian(h_He3, bin_centers)
                ax.hist(self.df["tof_corr"][self.df["is_helium3"]==1], bins = bins_tof, histtype = "step", label = f"Helium3 nuclei: tof = {popt_He3[1]:.2f} "+ r"$\pm$"+ f" {popt_He3[2]:.2f} ns")
                ax.plot(bins_tof, gaussian(bins_tof, popt_He3[0], popt_He3[1], popt_He3[2]), "--", color = "k")
                
                
            except:
                popt_He3 = [0, 0, 0]
                mean = self.df["tof_corr"][self.df["is_helium3"]==1].mean()
                std = self.df["tof_corr"][self.df["is_helium3"]==1].std()
                ax.hist(self.df["tof_corr"][self.df["is_helium3"]==1], bins = bins_tof, histtype = "step", label = f"Helium3 nuclei: tof = {mean:.2f} "+ r"$\pm$"+ f" {std:.2f} ns")

            
        if sum(self.df["is_tritium"])>20:
            try:
                h_tritium, _ = np.histogram(self.df["tof_corr"][self.df["is_tritium"]==1], bins = bins_tof)
                popt_tritium, pcov = fit_gaussian(h_tritium, bin_centers)
                ax.hist(self.df["tof_corr"][self.df["is_tritium"]==1], bins = bins_tof, histtype = "step", label = f"Tritium or Lithium6 nuclei: tof = {popt_tritium[1]:.2f} "+ r"$\pm$"+ f" {popt_tritium[2]:.2f} ns")
                ax.plot(bins_tof, gaussian(bins_tof, popt_tritium[0], popt_tritium[1], popt_tritium[2]), "--", color = "k")
            except:
                popt_tritium = [0, 0, 0]
                mean = self.df["tof_corr"][self.df["is_tritium"]==1].mean()
                std = self.df["tof_corr"][self.df["is_tritium"]==1].std()
                ax.hist(self.df["tof_corr"][self.df["is_tritium"]==1], bins = bins_tof, histtype = "step", label = f"Tritium or Lithium6 nuclei: tof = {mean:.2f} "+ r"$\pm$"+ f" {std:.2f} ns")
            
            
        
        if sum(self.df["is_lithium6"])>20:
            try:
                h_Li6, _ = np.histogram(self.df["tof_corr"][self.df["is_lithium6"]==1], bins = bins_tof)
                popt_Li6, pcov = fit_gaussian(h_Li6, bin_centers)
                ax.hist(self.df["tof_corr"][self.df["is_lithium6"]==1], bins = bins_tof, histtype = "step", label = f"Lithium6 or Tritium nuclei: tof = {popt_Li6[1]:.2f} "+ r"$\pm$"+ f" {popt_Li6[2]:.2f} ns")
                ax.plot(bins_tof, gaussian(bins_tof, popt_Li6[0], popt_Li6[1], popt_Li6[2]), "--", color = "k")
            except:
                popt_Li6 = [0, 0, 0]
                mean = self.df["tof_corr"][self.df["is_lithium6"]==1].mean()
                std = self.df["tof_corr"][self.df["is_lithium6"]==1].std() 
                ax.hist(self.df["tof_corr"][self.df["is_lithium6"]==1], bins = bins_tof, histtype = "step", label = f"Lithium6 or Tritium  nuclei: tof = {mean:.2f} "+ r"$\pm$"+ f" {std:.2f} ns")
                
            
          
            
            
        
        
        #plot the fits, for visual inspection
        
        ax.plot(bins_tof, gaussian(bins_tof, popt[0], popt[1]+t0, popt[2]), "--", color = "k")
        
        for p in [popt_mu, popt_pi]:
            ax.plot(bins_tof, gaussian(bins_tof, p[0], p[1], p[2]), "--", color = "k")
        
        
        if there_is_proton:
            ax.plot(bins_tof, gaussian(bins_tof, popt_p[0], popt_p[1], popt_p[2]), "--", color = "k")
            
        if sum(self.df["is_deuteron"])>20:
            ax.plot(bins_tof, gaussian(bins_tof, popt_D[0], popt_D[1], popt_D[2]), "--", color = "k")
            
            
        ax.set_ylabel("Number of events", fontsize = 18)
        ax.set_xlabel("Time of flight (ns)", fontsize = 18)
        ax.legend(fontsize = 10)
        ax.grid()
        ax.set_yscale("log")
        ax.set_ylim(0.5, 5e5)
        ax.set_title(f"Run {self.run_number} T0-T1 TOF ({self.run_momentum} MeV/c)", fontsize = 20)
        self.pdf_global.savefig(fig)
        
        
        if there_is_proton:
            ax.set_xlim(None, popt_p[1] * 1.2)
            
        if sum(self.df["is_deuteron"])>20:
            ax.set_xlim(None, popt_D[1] * 1.2)
            
        self.pdf_global.savefig(fig)
        
        plt.close()
        
        ################################### Make a second clean plot for T0T4 TOFs ################################ 
        fig, ax = plt.subplots(figsize = (8, 6))
        
        #plot the distributions
        ax.hist(self.df["tof_t0t4_corr"][self.df["is_electron"]==1], bins = bins_tof_t0t4, histtype = "step", label = f"Electrons: tof = {popt_t0t4[1]+t0_t0t4:.2f} "+ r"$\pm$"+ f" {popt_t0t4[2]:.2f} ns")  
        ax.hist(self.df["tof_t0t4_corr"][self.df["is_muon"]==1], bins = bins_tof_t0t4, histtype = "step", label = f"Muons: tof = {popt_mu_t0t4[1]:.2f} "+ r"$\pm$"+ f" {popt_mu_t0t4[2]:.2f} ns")
        ax.hist(self.df["tof_t0t4_corr"][self.df["is_pion"]==1], bins = bins_tof_t0t4, histtype = "step", label = f"Pions: tof = {popt_pi_t0t4[1]:.2f} "+ r"$\pm$"+ f" {popt_pi_t0t4[2]:.2f} ns")
        
        
        if there_is_proton:
            ax.hist(self.df["tof_t0t4_corr"][self.df["is_proton"]==1], bins = bins_tof_t0t4, histtype = "step", label = f"Protons: tof = {popt_p_t0t4[1]:.2f} "+ r"$\pm$"+ f" {popt_p_t0t4[2]:.2f} ns")
            
            
        if sum(self.df["is_deuteron"])>20:
            ax.hist(self.df["tof_t0t4_corr"][self.df["is_deuteron"]==1], bins = bins_tof_t0t4, histtype = "step", label = f"Deuterons: tof = {popt_D_t0t4[1]:.2f} "+ r"$\pm$"+ f" {popt_D_t0t4[2]:.2f} ns")
            
            
            
        
            
        if sum(self.df["is_helium3"])>20:
            
            try:
                h_He3_t0t4, _ = np.histogram(self.df["tof_t0t4_corr"][self.df["is_helium3"]==1], bins = bins_tof_t0t4)
                popt_He3_t0t4, pcov_t0t4 = fit_gaussian(h_He3_t0t4, bin_centers_t0t4)
                ax.hist(self.df["tof_t0t4_corr"][self.df["is_helium3"]==1], bins = bins_tof_t0t4, histtype = "step", label = f"Helium3 nuclei: tof = {popt_He3_t0t4[1]:.2f} "+ r"$\pm$"+ f" {popt_He3_t0t4[2]:.2f} ns")
                ax.plot(bins_tof_t0t4, gaussian(bins_tof_t0t4, popt_He3_t0t4[0], popt_He3_t0t4[1], popt_He3_t0t4[2]), "--", color = "k")
                
                
            except:
                popt_He3 = [0, 0, 0]
                mean = self.df["tof_t0t4_corr"][self.df["is_helium3"]==1].mean()
                std = self.df["tof_t0t4_corr"][self.df["is_helium3"]==1].std()
                ax.hist(self.df["tof_t0t4_corr"][self.df["is_helium3"]==1], bins = bins_tof, histtype = "step", label = f"Helium3 nuclei: tof = {mean:.2f} "+ r"$\pm$"+ f" {std:.2f} ns")

            
        if sum(self.df["is_tritium"])>20:
            try:
                h_tritium_t0t4, _ = np.histogram(self.df["tof_t0t4_corr"][self.df["is_tritium"]==1], bins = bins_tof_t0t4)
                popt_tritium_t0t4, pcov_t0t4 = fit_gaussian(h_tritium_t0t4, bin_centers_t0t4)
                ax.hist(self.df["tof_t0t4_corr"][self.df["is_tritium"]==1], bins = bins_tof_t0t4, histtype = "step", label = f"Tritium nuclei: tof = {popt_tritium_t0t4[1]:.2f} "+ r"$\pm$"+ f" {popt_tritium_t0t4[2]:.2f} ns")
                ax.plot(bins_tof_t0t4, gaussian(bins_tof_t0t4, popt_tritium_t0t4[0], popt_tritium_t0t4[1], popt_tritium_t0t4[2]), "--", color = "k")
            except:
                popt_tritium_t0t4 = [0, 0, 0]
                mean = self.df["tof_t0t4_corr"][self.df["is_tritium"]==1].mean()
                std = self.df["tof_t0t4_corr"][self.df["is_tritium"]==1].std()
                ax.hist(self.df["tof_t0t4_corr"][self.df["is_tritium"]==1], bins = bins_tof_t0t4, histtype = "step", label = f"Tritium nuclei: tof = {mean:.2f} "+ r"$\pm$"+ f" {std:.2f} ns")
            
            
        
        if sum(self.df["is_lithium6"])>20:
            try:
                h_Li6_t0t4, _ = np.histogram(self.df["tof_t0t4_corr"][self.df["is_lithium6"]==1], bins = bins_tof_t0t4)
                popt_Li6_t0t4, pcov_t0t4 = fit_gaussian(h_Li6_t0t4, bin_centers_t0t4)
                ax.hist(self.df["tof_t0t4_corr"][self.df["is_lithium6"]==1], bins = bins_tof_t0t4, histtype = "step", label = f"Lithium6 nuclei: tof = {popt_Li6_t0t4[1]:.2f} "+ r"$\pm$"+ f" {popt_Li6_t0t4[2]:.2f} ns")
                ax.plot(bins_tof_t0t4, gaussian(bins_tof_t0t4, popt_Li6_t0t4[0], popt_Li6_t0t4[1], popt_Li6_t0t4[2]), "--", color = "k")
            except:
                popt_Li6 = [0, 0, 0]
                mean = self.df["tof_t0t4_corr"][self.df["is_lithium6"]==1].mean()
                std = self.df["tof_t0t4_corr"][self.df["is_lithium6"]==1].std() 
                ax.hist(self.df["tof_t0t4_corr"][self.df["is_lithium6"]==1], bins = bins_tof_t0t4, histtype = "step", label = f"Lithium6 nuclei: tof = {mean:.2f} "+ r"$\pm$"+ f" {std:.2f} ns")
                
           
        
        #plot the fits, for visual inspection
        
        ax.plot(bins_tof_t0t4, gaussian(bins_tof_t0t4, popt_t0t4[0], popt_t0t4[1]+t0_t0t4, popt_t0t4[2]), "--", color = "k")
        
        for p in [popt_mu_t0t4, popt_pi_t0t4]:
            ax.plot(bins_tof_t0t4, gaussian(bins_tof_t0t4, p[0], p[1], p[2]), "--", color = "k")
        
        
        if there_is_proton:
            ax.plot(bins_tof_t0t4, three_gaussians(bins_tof_t0t4, popt_p_t0t4[0], popt_p_t0t4[1], popt_p_t0t4[2], popt_p_t0t4[3], popt_p_t0t4[4], popt_p_t0t4[5], popt_p_t0t4[6], popt_p_t0t4[7], popt_p_t0t4[8]), "--", color = "k")
            
            #choose the gaussian corresponding to the highest amplitude
            if popt_p_t0t4[0] > popt_p_t0t4[3] and popt_p_t0t4[0] > popt_p_t0t4[6]:
               
                mean_proton_T0T4_tof = popt_p_t0t4[1]
                std_proton_T0T4_tof = popt_p_t0t4[2]
                
            elif  popt_p_t0t4[3]>popt_p_t0t4[6]:
                mean_proton_T0T4_tof = popt_p_t0t4[4]
                std_proton_T0T4_tof = popt_p_t0t4[5]
                
                
            else:
                mean_proton_T0T4_tof = popt_p_t0t4[7]
                std_proton_T0T4_tof = popt_p_t0t4[8]
                
            
        if sum(self.df["is_deuteron"])>20:
            ax.plot(bins_tof_t0t4, gaussian(bins_tof_t0t4, popt_D_t0t4[0], popt_D_t0t4[1], popt_D_t0t4[2]), "--", color = "k")

        if sum(self.df["is_tritium"])>20:
            ax.plot(bins_tof_t0t4, gaussian(bins_tof_t0t4, popt_tritium_t0t4[0], popt_tritium_t0t4[1], popt_tritium_t0t4[2]), "--", color = "k")
            
            
        ax.set_ylabel("Number of events", fontsize = 18)
        ax.set_xlabel("Time of flight (ns)", fontsize = 18)
        ax.legend(fontsize = 10)
        ax.grid()
        ax.set_yscale("log")
        ax.set_ylim(0.5, 5e5)
        ax.set_title(f"Run {self.run_number} T0-T4 TOF ({self.run_momentum} MeV/c)", fontsize = 20)
        self.pdf_global.savefig(fig)
        
        
        if there_is_proton:
            ax.set_xlim(None, popt_p[1] * 1.2)
            
        if sum(self.df["is_deuteron"])>20:
            ax.set_xlim(None, popt_D[1] * 1.2)
            
        self.pdf_global.savefig(fig)
        
        plt.close()
        
        
        #Here save the mean TOF and std for each particle population
        self.particle_tof_mean = {
            "electron": popt[1]+t0,
            "muon": popt_mu[1],
            "pion": popt_pi[1],
            "proton": popt_p[1] if there_is_proton else 0,
            "deuteron": popt_D[1] if sum(self.df["is_deuteron"])>20 else 0,
            "helium3": popt_He3[1] if sum(self.df["is_helium3"])>20 else 0,
            "tritium": popt_tritium[1] if sum(self.df["is_tritium"])>20 else 0,
            "lithium6": popt_Li6[1] if sum(self.df["is_lithium6"])>20 else 0,
        }
        
        self.particle_tof_std = {
            "electron": popt[2],
            "muon": popt_mu[2],
            "pion": popt_pi[2],
            "proton": popt_p[2] if there_is_proton else 0,
            "deuteron": popt_D[2] if sum(self.df["is_deuteron"])>20 else 0,
            "helium3": popt_He3[2] if sum(self.df["is_helium3"])>20 else 0,
            "tritium": popt_tritium[2] if sum(self.df["is_tritium"])>20 else 0,
            "lithium6": popt_Li6[2] if sum(self.df["is_lithium6"])>20 else 0,    
        }
        
        self.particle_tof_eom = {
            "electron": popt[2]/np.sqrt(sum(self.df["is_electron"])),
            "muon": popt_mu[2]/np.sqrt(sum(self.df["is_muon"])),
            "pion": popt_pi[2]/np.sqrt(sum(self.df["is_pion"])),
            "proton": popt_p[2]/np.sqrt(sum(self.df["is_proton"])) if there_is_proton else 0,
            "deuteron": popt_D[2]/np.sqrt(sum(self.df["is_deuteron"])) if sum(self.df["is_deuteron"])>20 else 0,
            "helium3": popt_He3[2]/np.sqrt(sum(self.df["is_helium3"])) if sum(self.df["is_helium3"])>20 else 0,
            "tritium": popt_tritium[2]/np.sqrt(sum(self.df["is_tritium"])) if sum(self.df["is_tritium"])>20 else 0,
            "lithium6": popt_Li6[2]/np.sqrt(sum(self.df["is_lithium6"])) if sum(self.df["is_lithium6"])>20 else 0,
 
            
            
        }
        
        #################### same for T0T4 #####################
        self.particle_tof_t0t4_mean = {
            "electron": popt_t0t4[1]+t0_t0t4,
            "muon": popt_mu_t0t4[1],
            "pion": popt_pi_t0t4[1],
            "proton": mean_proton_T0T4_tof if there_is_proton else 0,
            "deuteron": popt_D_t0t4[1] if sum(self.df["is_deuteron"])>20 else 0,
            "helium3": popt_He3[1] if sum(self.df["is_helium3"])>20 else 0,
            "tritium": popt_tritium_t0t4[1] if sum(self.df["is_tritium"])>20 else 0,
            "lithium6": popt_Li6[1] if sum(self.df["is_lithium6"])>20 else 0,
        }
        
        self.particle_tof_t0t4_std = {
            "electron": popt_t0t4[2],
            "muon": popt_mu_t0t4[2],
            "pion": popt_pi_t0t4[2],
            "proton": std_proton_T0T4_tof if there_is_proton else 0,
            "deuteron": popt_D_t0t4[2] if sum(self.df["is_deuteron"])>20 else 0,
            "helium3": popt_He3[2] if sum(self.df["is_helium3"])>20 else 0,
            "tritium": popt_tritium_t0t4[2] if sum(self.df["is_tritium"])>20 else 0,
            "lithium6": popt_Li6[2] if sum(self.df["is_lithium6"])>20 else 0,    
        }
        
        self.particle_tof_t0t4_eom = {
            "electron": popt_t0t4[2]/np.sqrt(sum(self.df["is_electron"])),
            "muon": popt_mu_t0t4[2]/np.sqrt(sum(self.df["is_muon"])),
            "pion": popt_pi_t0t4[2]/np.sqrt(sum(self.df["is_pion"])),
            "proton": std_proton_T0T4_tof/np.sqrt(sum(self.df["is_proton"])) if there_is_proton else 0,
            "deuteron": popt_D_t0t4[2]/np.sqrt(sum(self.df["is_deuteron"])) if sum(self.df["is_deuteron"])>20 else 0,
            "helium3": popt_He3[2]/np.sqrt(sum(self.df["is_helium3"])) if sum(self.df["is_helium3"])>20 else 0,
            "tritium": popt_tritium_t0t4[2]/np.sqrt(sum(self.df["is_tritium"])) if sum(self.df["is_tritium"])>20 else 0,
            "lithium6": popt_Li6[2]/np.sqrt(sum(self.df["is_lithium6"])) if sum(self.df["is_lithium6"])>20 else 0,
 
            
            
        }
        
        
    def plot_all_TOFs(self):
        "quick function to plot all the TOFs for checking"
        bins = [np.linspace(10, 55, 150), np.linspace(0, 25, 150), np.linspace(-40, 40, 150), np.linspace(20, 70, 150), np.linspace(10, 55, 150), np.linspace(10, 55, 150), np.linspace(-220, 220, 300), np.linspace(-220, 220, 300), np.linspace(-220, 220, 300), np.linspace(-220, 220, 300)]
        tof_var = ["tof_corr", "tof_t0t4_corr", "tof_t4t1", "tof_t0t5", "tof_t1t5", "tof_t4t5", "t0_time", "t1_time", "t4_time", "t5_time"]
        tof_names = ["T0-T1 (etof corrected)","T0-T4(etof corrected)", "T4-T1", "T0-T5", "T1-T5", "T4-T5", "T0", "T1", "T4", "T5"]
        
        for i in range(len(tof_var)):
            fig, ax = plt.subplots(figsize = (8, 6))
            for population in ["is_electron", "is_muon", "is_pion", "is_proton"]:
                ax.hist(self.df[self.df[population]==1][tof_var[i]], bins = bins[i], label = f"{population} total: {sum(self.df[population])}", histtype = "step")

            ax.set_ylabel("Number of events", fontsize = 18)
            ax.set_xlabel("Time of flight (ns)", fontsize = 18)
            ax.legend(fontsize = 10)
            ax.grid()
            ax.set_yscale("log")
            ax.set_ylim(0.5, 5e5)
            ax.set_title(f"Run {self.run_number} {tof_names[i]} Time ({self.run_momentum} MeV/c) \n require T5 = {self.require_t5_hit}", fontsize = 20)
            self.pdf_global.savefig(fig)
            
        
        
        
    def plot_TOF_charge_distribution(self):
        '''Check visually the total charge deposited in the TOF detector, can be handy to identify events that do not actually cross the TOF'''
        
        
        fig, ax = plt.subplots(figsize = (8, 6))
        
        bins = np.linspace( min(self.df_all["total_TOF_charge"]), max(self.df_all["total_TOF_charge"]), 100)
        _ = ax.hist(self.df["total_TOF_charge"][self.df["is_electron"] == 1], bins = bins, label = 'electron', histtype = "step")
        _ = ax.hist(self.df["total_TOF_charge"][self.df["is_muon"] == 1], bins = bins, label = 'muon', histtype = "step")
        _ = ax.hist(self.df["total_TOF_charge"][self.df["is_pion"] == 1], bins = bins, label = 'pion', histtype = "step")

        if sum(self.df["is_proton"] == 1)>100:
            _ = ax.hist(self.df["total_TOF_charge"][self.df["is_proton"] == 1], bins = bins, label = 'proton', histtype = "step")
            
        if sum(self.df["is_helium3"] == 1) >1:
            _ = ax.hist(self.df["total_TOF_charge"][self.df["is_helium3"] == 1], bins = bins, label = 'helium3', histtype = "step")
            
        if sum(self.df["is_deuteron"] == 1) >20:
            _ = ax.hist(self.df["total_TOF_charge"][self.df["is_deuteron"] == 1], bins = bins, label = 'deuteron', histtype = "step")
            
        if sum(self.df["is_tritium"] == 1) >1:
            _ = ax.hist(self.df["total_TOF_charge"][self.df["is_tritium"] == 1], bins = bins, label = 'tritium', histtype = "step")
            
        if sum(self.df["is_lithium6"] == 1) >1:
            _ = ax.hist(self.df["total_TOF_charge"][self.df["is_lithium6"] == 1], bins = bins, label = 'lithium6', histtype = "step")
        _ = ax.hist(self.df_all["total_TOF_charge"][self.df_all["is_kept"] == 0], bins = bins, label = 'Triggers not kept for analysis', color = "k", histtype = "step")

        ax.legend()
        ax.set_yscale("log")
        ax.set_xlabel("Total charge in TOF detector (a.u.)")
        self.pdf_global.savefig(fig)
        plt.close()

        fig, ax = plt.subplots(figsize = (8, 6))
        bins = np.linspace( min(self.df["total_TOF_charge"]), max(self.df["total_TOF_charge"]), 100)
        _ = ax.hist(self.df["total_TOF_charge"][self.df["is_electron"] == 1], bins = bins, label = 'electron', histtype = "step")
        _ = ax.hist(self.df["total_TOF_charge"][self.df["is_muon"] == 1], bins = bins, label = 'muon', histtype = "step")
        _ = ax.hist(self.df["total_TOF_charge"][self.df["is_pion"] == 1], bins = bins, label = 'pion', histtype = "step")

        if sum(self.df["is_proton"] == 1)>100:
            _ = ax.hist(self.df["total_TOF_charge"][self.df["is_proton"] == 1], bins = bins, label = 'proton', histtype = "step")
        if sum(self.df["is_deuteron"] == 1) >20:
            _ = ax.hist(self.df["total_TOF_charge"][self.df["is_deuteron"] == 1], bins = bins, label = 'deuteron', histtype = "step")

        ax.legend()
        ax.set_yscale("log")
        ax.set_xlabel("Total charge in TOF detector (a.u.)")
        self.pdf_global.savefig(fig)
        plt.close()
        
        
                   
    def output_beam_ana_to_root(self, output_name = None):
        ''''Output the results of the beam analysis as a root file with three branches, the 1D run information (number, nominal momentum, refractive index, whether ACT5 is in the beam line), the 1D results (mean measured [T0T1] TOF and momentum for each particle type with EOM and std for the TOF only), number of triggers kept, total number of triggers'''
        if output_name == None:
            output_name = f"beam_analysis_output_R{self.run_number}.root"
            
            
        for col in self.df.columns:
            if col not in self.df_all.columns:
                self.df_all[col] = np.nan  # create empty column first if you want aligned length
                self.df_all.loc[self.df_all["is_kept"], col] = self.df[col].values

                
                
        for is_particle in ["is_muon", "is_electron", "is_pion", "is_proton", "is_deuteron", "is_helium3", "is_tritium", "is_lithium6"]:
            self.df_all[is_particle] = self.df_all[is_particle].fillna(0)
            self.df_all[is_particle] = self.df_all[is_particle].astype(np.int32)

        
        
        self.df_all["is_kept"] = self.df_all["is_kept"].astype(np.int32)
        
        

       # --- Convert DataFrame to dictionary of numpy arrays ---
        branches = {col: self.df_all[col].to_numpy() for col in self.df_all.columns}

        # --- Create ROOT file and save the tree ---
        with uproot.recreate(output_name) as f:
            # Write the main TTree
            f["beam_analysis"] = branches

            # Write scalar metadata as a separate tree (recommended)
            f["run_info"] = {
                "run_number": np.array([self.run_number], dtype=np.int32),
                "run_momentum": np.array([self.run_momentum], dtype=np.float64),          
                "n_eveto": np.array([self.n_eveto], dtype=np.float64),
                "n_tagger": np.array([self.n_tagger], dtype=np.float64),
                "there_is_ACT5":np.array([self.there_is_ACT5], dtype = np.int32),

            }


            #save as a separate branch the 1d results of interest

            results = {
                "act_eveto_cut":np.array([self.eveto_cut], dtype=np.float64),
                "act_tagger_cut":np.array([self.act35_cut_pi_mu], dtype=np.float64),
                "proton_tof_cut":np.array([proton_tof_cut], dtype=np.float64),
                "deuteron_tof_cut":np.array([deuteron_tof_cut], dtype=np.float64),
                "mu_tag_cut": np.array([self.mu_tag_cut], dtype=np.float64),
                "using_mu_tag_cut": np.array([self.using_mu_tag_cut], dtype=np.float64),
                
                "pion_purity":np.array([self.pion_purity], dtype=np.float64),
                "pion_efficiency":np.array([self.pion_efficiency], dtype=np.float64),
                "muon_purity":np.array([self.muon_purity], dtype=np.float64),
                "muon_efficiency":np.array([self.muon_efficiency], dtype=np.float64),
            }
            for prefix, d in [("tof_mean", self.particle_tof_mean),
                  ("tof_std", self.particle_tof_std),
                  ("tof_eom", self.particle_tof_eom),
                  ("momentum_mean", self.particle_mom_mean),
                  ("momentum_eom", self.particle_mom_mean_err),
                  ("momentum_after_beam_window_mean", self.particle_mom_final_mean),
                  ("momentum_after_beam_window_eom", self.particle_mom_final_mean_err)]:
                for key, value in d.items():
                    results[f"{prefix}_{key}"] = np.array([value], dtype=np.float64)

                    
            results["n_electrons"] = np.array([sum(self.df_all["is_electron"]) ], dtype=np.float64) 
            results["n_muons"] =  np.array([sum(self.df_all["is_muon"])], dtype=np.float64) 
            results["n_pions"] =  np.array([sum(self.df_all["is_pion"])], dtype=np.float64)   
            results["n_protons"] =  np.array([sum(self.df_all["is_proton"])], dtype=np.float64)   
            results["n_deuterons"] =   np.array([sum(self.df_all["is_deuteron"])], dtype=np.float64)   
            results["n_helium3"] =   np.array([sum(self.df_all["is_helium3"])], dtype=np.float64)   
            results["n_lithium6"] =   np.array([sum(self.df_all["is_lithium6"])], dtype=np.float64)   
            results["n_tritium"] =   np.array([sum(self.df_all["is_tritium"])], dtype=np.float64)   
            results["n_triggers_kept"] =  np.array([sum(self.df_all["is_kept"])], dtype=np.float64) 
            
            results["n_triggers_total"] =  np.array([len(self.df_all["is_proton"]) ], dtype=np.float64)  
            
            
            f["scalar_results"] = results 
            
            print(f"Saved output file to {output_name}")
            
            
            
            
    def output_to_root(self, output_name = None):
        ''''Output the results of the beam analysis as a root file with three branches, the 1D run information (number, nominal momentum, refractive index, whether ACT5 is in the beam line), the 1D results (cut lines, number of triggers kept, total number of triggers) and the relevant varaibles '''
        if output_name == None:
            output_name = f"beam_analysis_output_R{self.run_number}.root"
            
           
        for col in self.df.columns:
            if col not in self.df_all.columns:
                if col not in ["is_muon", "is_electron", "is_pion", "is_proton", "is_deuteron", "is_helium3", "is_tritium", "is_lithium6", "final_momentum", "final_momentum_error", "initial_momentum", "initial_momentum_error"]:
                    # create empty column first so the DataFrame lengths match
                    self.df_all[col] = np.nan
                    # copy values from df; cast bools to a numeric type to avoid
                    # FutureWarning about incompatible dtype
                    values = self.df[col]
                    if values.dtype == bool:
                        print(f"Converting boolean column {col} to int for ROOT compatibility.")
                        # converting to float keeps compatibility with NaNs
                        values = values.astype(np.float64)
                    self.df_all.loc[self.df.index, col] = values

        
        self.df_all["is_kept"] = self.df_all["is_kept"].astype(np.int32) 
        
        
        #changing the names of the variables so they are more clear 
        rename_map = {
            "tof": "tof_t0t1",
            "t4_l": "t4_l_time",
            "t4_r": "t4_r_time",
            "act0_time_l": "act0_l_time",
            "act0_time_r": "act0_r_time",
            "act0_l": "act0_l_charge",
            "act1_l": "act1_l_charge",
            "act2_l": "act2_l_charge",
            "act3_l": "act3_l_charge",
            "act4_l": "act4_l_charge",
            "act5_l": "act5_l_charge",
            "act0_r": "act0_r_charge",
            "act1_r": "act1_r_charge",
            "act2_r": "act2_r_charge",
            "act3_r": "act3_r_charge",
            "act4_r": "act4_r_charge",
            "act5_r": "act5_r_charge",
            "mu_tag_l": "mu_tag_l_charge",
            "mu_tag_r": "mu_tag_r_charge",
         }
        
        self.df_all = self.df_all.rename(columns=rename_map)
        nTriggers = np.array([len(self.df_all["is_kept"]) ], dtype=np.float64)
        self.df_all = self.df_all.drop("is_kept", axis = 1)

        

       # --- Convert DataFrame to dictionary of numpy arrays ---
        branches = {col: self.df_all[col].to_numpy() for col in self.df_all.columns}

        # --- Create ROOT file and save the tree ---
        with uproot.recreate(output_name) as f:
            # Write the main TTree
            f["beam_analysis"] = branches

            # Write scalar metadata as a separate tree (recommended)
            f["run_info"] = {
                "run_number": np.array([self.run_number], dtype=np.int32),
                "run_momentum": np.array([self.run_momentum], dtype=np.float64),          
                "n_eveto": np.array([self.n_eveto], dtype=np.float64),
                "n_tagger": np.array([self.n_tagger], dtype=np.float64),
                "there_is_ACT5":np.array([self.there_is_ACT5], dtype = np.int32),

            }


            #save as a separate branch the 1d results of interest

            results = {
                "act_eveto_cut":np.array([self.eveto_cut], dtype=np.float64),
                "act_tagger_cut":np.array([self.act35_cut_pi_mu], dtype=np.float64),
                "proton_tof_cut":np.array([self.proton_tof_cut], dtype=np.float64),
                "deuteron_tof_cut":np.array([self.deuteron_tof_cut], dtype=np.float64),
                "mu_tag_cut": np.array([self.mu_tag_cut], dtype=np.float64),
                "using_mu_tag_cut": np.array([self.using_mu_tag_cut], dtype=np.float64),
                
                #output the number of triggers identified as each particle, for reference 
                #here we need to sum from the df dataframe and not the df_all which doesn't have the PID info
                "n_electrons": np.array([sum(self.df["is_electron"])], dtype=np.float64),
                "n_muons": np.array([sum(self.df["is_muon"])], dtype=np.float64),
                "n_pions": np.array([sum(self.df["is_pion"])], dtype=np.float64),
                "n_protons": np.array([sum(self.df["is_proton"])], dtype=np.float64),
                "n_deuterium": np.array([sum(self.df["is_deuteron"])], dtype=np.float64),
                
#                 "pion_purity":np.array([self.pion_purity], dtype=np.float64),
#                 "pion_efficiency":np.array([self.pion_efficiency], dtype=np.float64),
#                 "muon_purity":np.array([self.muon_purity], dtype=np.float64),
#                 "muon_efficiency":np.array([self.muon_efficiency], dtype=np.float64),
            }
            for prefix, d in [("tof_mean", self.particle_tof_mean),
                  ("tof_std", self.particle_tof_std),
                  ("tof_eom", self.particle_tof_eom),
                  ("momentum_mean", self.particle_mom_mean),
                  ("momentum_eom", self.particle_mom_mean_err),
                  ("momentum_after_beam_window_mean", self.particle_mom_final_mean),
                  ("momentum_after_beam_window_eom", self.particle_mom_final_mean_err)]:
                for key, value in d.items():
                    results[f"{prefix}_{key}"] = np.array([value], dtype=np.float64)

                    
#             results["n_electrons"] = np.array([sum(self.df_all["is_electron"]) ], dtype=np.float64) 
#             results["n_muons"] =  np.array([sum(self.df_all["is_muon"])], dtype=np.float64) 
#             results["n_pions"] =  np.array([sum(self.df_all["is_pion"])], dtype=np.float64)   
#             results["n_protons"] =  np.array([sum(self.df_all["is_proton"])], dtype=np.float64)   
#             results["n_deuterons"] =   np.array([sum(self.df_all["is_deuteron"])], dtype=np.float64)   
#             results["n_helium3"] =   np.array([sum(self.df_all["is_helium3"])], dtype=np.float64)   
#             results["n_lithium6"] =   np.array([sum(self.df_all["is_lithium6"])], dtype=np.float64)   
#             results["n_tritium"] =   np.array([sum(self.df_all["is_tritium"])], dtype=np.float64)   

            results["n_triggers_total"] = nTriggers  
            
            
            f["scalar_results"] = results 
            
            print(f"Saved output file to {output_name}")
            
            
            
            
    def study_electrons(self, cut_line):
        '''This function is dedicated to understanding why some electrons are missed by the ACT02 tag and check why some muons and pions are tagged as electrons'''
        #step 1: make a selection of the particles that are not tagged as electrons by ACT02 but deposit a lot of light in the ACT35
        mask = (self.df["is_electron"] == 0) & (self.df["act_tagger"] >= cut_line) & (self.df["tof"] < proton_tof_cut) 
        df_e = self.df[mask]
        
        self.plot_ACT35_left_vs_right(cut_line, "non-tagged e-like triggers")
        
        bins = np.linspace(0, self.eveto_cut, 50)
        
        
        fig, axs = plt.subplots(3, 3, figsize = (14, 10), sharex = False)
        
        for i, ax_row in enumerate(axs):
            for j, ax in enumerate(ax_row):
                ax.hist2d(df_e[f"act{i}_l"], df_e[f"act{j}_r"], bins = (bins, bins), norm=LogNorm())
                ax.set_xlabel(f"ACT{i} left (PE)", fontsize = 12)
                ax.set_ylabel(f"ACT{j} right (PE)", fontsize = 12)
        
        fig.suptitle(f"Triggers not tagged by ACT20 with ACT35 > {cut_line} PE", weight = "bold", fontsize = 18)
        self.pdf_global.savefig(fig)
        plt.close()
        
        bins = np.linspace(0, 25, 50)
        fig, axs = plt.subplots(2+int(self.there_is_ACT5 == True), 2+int(self.there_is_ACT5 == True), figsize = (14, 10), sharex = False)
        
        for i, ax_row in enumerate(axs):
            i = i+3
            for j, ax in enumerate(ax_row):
                
                j = j+3
                ax.hist2d(df_e[f"act{i}_l"], df_e[f"act{j}_r"], bins = (bins, bins), norm=LogNorm())
                ax.set_xlabel(f"ACT{i} left (PE)", fontsize = 12)
                ax.set_ylabel(f"ACT{j} right (PE)", fontsize = 12)
        
        fig.suptitle(f"Triggers not tagged by ACT20 with ACT35 > {cut_line} PE", weight = "bold", fontsize = 18)
        self.pdf_global.savefig(fig)
        plt.close()
        

        
        fig, axs = plt.subplots(2+int(self.there_is_ACT5 == True), 2+int(self.there_is_ACT5 == True), figsize = (14, 10), sharex = False)
        for i, ax_row in enumerate(axs):
            i = i+3     
            for j, ax in enumerate(ax_row):
                j = j+3 
                ax.hist2d(df_e[f"act{i}_l"], df_e[f"act{j}_l"], bins = (bins, bins), norm=LogNorm())
                
                ax.set_xlabel(f"ACT{i} left (PE)", fontsize = 12)

                ax.set_ylabel(f"ACT{j} left (PE)", fontsize = 12)
                    
        
        fig.suptitle(f"Triggers not tagged by ACT20 with ACT35 > {cut_line} PE", weight = "bold", fontsize = 18)
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        fig, axs = plt.subplots(2+int(self.there_is_ACT5 == True), 2+int(self.there_is_ACT5 == True), figsize = (14, 10), sharex = False)
        for i, ax_row in enumerate(axs):
            i = i+3     
            for j, ax in enumerate(ax_row):
                j = j+3 
                ax.hist2d(df_e[f"act{i}_r"], df_e[f"act{j}_r"], bins = (bins, bins), norm=LogNorm())
                
                ax.set_xlabel(f"ACT{i} right (PE)", fontsize = 12)

                ax.set_ylabel(f"ACT{j} right (PE)", fontsize = 12)
                    
        
        fig.suptitle(f"Triggers not tagged by ACT20 with ACT35 > {cut_line} PE", weight = "bold", fontsize = 18)
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        
     ################################################
     ### Here check using the tof which ones are the muons and which are the other
   
        print(f"The difference between the muon TOF ({self.particle_tof_mean['electron']} +/- {self.particle_tof_std['electron']} ns) and the electron TOF ({self.particle_tof_mean['muon']} +/- {self.particle_tof_std['muon']} ns)")
    
        
        mid_tof_e_mu = self.particle_tof_mean["electron"]+ 3 * self.particle_tof_std["electron"]
        
        #(self.particle_tof_mean["muon"]-self.particle_tof_mean["electron"])/2
        df_e_true = df_e[df_e["tof"]<mid_tof_e_mu]
        df_mu_true = df_e[df_e["tof"]>mid_tof_e_mu]
    
        
        
        ### check against the tof
        tof_bins = np.linspace(12, proton_tof_cut, 100)
        fig, ax = plt.subplots(figsize = (8, 6))
        
        ax.hist(df_e_true["tof"], bins = tof_bins, color = "red", label = f"e-like triggers not tagged by ACT20 with ACT35 > {cut_line}")
        ax.hist(df_mu_true["tof"], bins = tof_bins, color = "black", label = f"mu-like triggers not tagged by ACT20 with ACT35 > {cut_line}")
        ax.hist(self.df["tof"][self.df["is_electron"]], bins = tof_bins, label = "Triggers tagged as electrons by ACT20", histtype = "step")
        ax.hist(self.df["tof"][self.df["is_muon"]], bins = tof_bins, label = "Muons", histtype = "step")
        ax.hist(self.df["tof"][self.df["is_pion"]], bins = tof_bins, label = "Pions", histtype = "step")
        ax.axvline(mid_tof_e_mu, linestyle = "--", color = "k", label = f"e/mu tof cut: {mid_tof_e_mu:.2f}")
        fig.suptitle(f"Study of triggers \n not tagged by ACT20 with ACT35 > {cut_line} PE", weight = "bold", fontsize = 18)
        ax.legend()
        ax.set_yscale("log")
        ax.grid()
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        ###### check the distribution of muons and electrons
        
        fig, axs = plt.subplots(3, 3, figsize = (14, 10), sharex = False)
        
        for i, ax_row in enumerate(axs):
            for j, ax in enumerate(ax_row):
                ax.grid()
                
                ax.scatter(df_mu_true[f"act{i}_l"], df_mu_true[f"act{j}_r"], color = "black", label = "muon-like", s = 2)
                ax.scatter(df_e_true[f"act{i}_l"], df_e_true[f"act{j}_r"], color = "red", label = "electron-like", s = 1)
                
                ax.set_xlabel(f"ACT{i} left (PE)", fontsize = 12)
                ax.set_ylabel(f"ACT{j} right (PE)", fontsize = 12)
                ax.legend()
        
        fig.suptitle(f"Triggers not tagged by ACT20 with ACT35 > {cut_line} PE", weight = "bold", fontsize = 18)
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        
        #################
        fig, axs = plt.subplots(2+int(self.there_is_ACT5 == True), 2+int(self.there_is_ACT5 == True), figsize = (14, 10), sharex = False)
        
        for i, ax_row in enumerate(axs):
            i = i+3
            for j, ax in enumerate(ax_row):
                
                j = j+3
                ax.grid()
                
                ax.scatter(df_mu_true[f"act{i}_l"], df_mu_true[f"act{j}_r"], color = "black", label = "muon-like", s = 2)
                ax.scatter(df_e_true[f"act{i}_l"], df_e_true[f"act{j}_r"], color = "red", label = "electron-like", s = 1)
                
                ax.set_xlabel(f"ACT{i} left (PE)", fontsize = 12)
                ax.set_ylabel(f"ACT{j} right (PE)", fontsize = 12)
                ax.legend()
        
        fig.suptitle(f"Triggers not tagged by ACT20 with ACT35 > {cut_line} PE", weight = "bold", fontsize = 18)
        self.pdf_global.savefig(fig)
        plt.close()
        
        fig, axs = plt.subplots(2+int(self.there_is_ACT5 == True), 2+int(self.there_is_ACT5 == True), figsize = (14, 10), sharex = False)
        for i, ax_row in enumerate(axs):
            i = i+3     
            for j, ax in enumerate(ax_row):
                j = j+3 
                ax.grid()
                
                ax.scatter(df_mu_true[f"act{i}_l"], df_mu_true[f"act{j}_l"], color = "black", label = "muon-like", s = 2)
                ax.scatter(df_e_true[f"act{i}_l"], df_e_true[f"act{j}_l"], color = "red", label = "electron-like", s = 1)
                
                ax.set_xlabel(f"ACT{i} left (PE)", fontsize = 12)

                ax.set_ylabel(f"ACT{j} left (PE)", fontsize = 12)
                ax.legend()
                    
        
        fig.suptitle(f"Triggers not tagged by ACT20 with ACT35 > {cut_line} PE", weight = "bold", fontsize = 18)
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        fig, axs = plt.subplots(2+int(self.there_is_ACT5 == True), 2+int(self.there_is_ACT5 == True), figsize = (14, 10), sharex = False)
        for i, ax_row in enumerate(axs):
            i = i+3     
            for j, ax in enumerate(ax_row):
                j = j+3 
                ax.grid()
                
                ax.scatter(df_mu_true[f"act{i}_r"], df_mu_true[f"act{j}_r"], color = "black", label = "muon-like", s = 2)
                ax.scatter(df_e_true[f"act{i}_r"], df_e_true[f"act{j}_r"], color = "red", label = "electron-like", s = 1)
                ax.set_xlabel(f"ACT{i} right (PE)", fontsize = 12)

                ax.set_ylabel(f"ACT{j} right (PE)", fontsize = 12)
                ax.legend()
                    
        
        fig.suptitle(f"Triggers not tagged by ACT20 with ACT35 > {cut_line} PE", weight = "bold", fontsize = 18)
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        
    def plot_number_particles_per_POT(self):
        '''This function plots the number of particles of each type recorded per spill and then per POT, required for the beam flux paper and represent an example of how to read POT information from Arturo's readings of the nxcals CERN database'''
        
#         #making a complete dataframe with all of the entries, including the rejected ones 
#         df_comp = self.df_all.copy()
        
#         for col in self.df.columns:
#             if col not in df_comp.columns:
#                 df_comp[col] = np.nan  # create empty column first if you want aligned length
#                 df_comp.loc[df_comp["is_kept"], col] = self.df[col].values
        
        spill_index = [s for s in self.df["spill_number"].unique()]
        number_e_per_spill = np.array([sum(self.df[self.df["spill_number"]==s]["is_electron"]) for s in self.df["spill_number"].unique()])
        number_mu_per_spill = np.array([sum(self.df[self.df["spill_number"]==s]["is_muon"]) for s in self.df["spill_number"].unique()])
        number_pi_per_spill = np.array([sum(self.df[self.df["spill_number"]==s]["is_pion"]) for s in self.df["spill_number"].unique()])
        number_p_per_spill = np.array([sum(self.df[self.df["spill_number"]==s]["is_proton"]) for s in self.df["spill_number"].unique()])
        number_D_per_spill = np.array([sum(self.df[self.df["spill_number"]==s]["is_deuteron"]) for s in self.df["spill_number"].unique()])
        
        number_rejected_per_spill = np.array([len(self.df_all[(self.df_all["spill_number"]==s) & (self.df_all["is_kept"]==0)]) for s in self.df_all["spill_number"].unique()])
        spill_index_all = [s for s in self.df_all["spill_number"].unique()]
        
        
        
        
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.plot(spill_index, number_e_per_spill, "x", label = f"Electrons ({sum(self.df['is_electron'])})")
        ax.plot(spill_index, number_mu_per_spill, "x", label = f"Muons  ({sum(self.df['is_muon'])})")
        ax.plot(spill_index, number_pi_per_spill, "x", label = f"Pions  ({sum(self.df['is_pion'])})")
        ax.plot(spill_index, number_p_per_spill, "x", label = f"Protons  ({sum(self.df['is_proton'])})")
        ax.plot(spill_index, number_D_per_spill, "x", label = f"Deuterons  ({sum(self.df['is_deuteron'])})")
        ax.plot(spill_index_all, number_rejected_per_spill, "x", label = "Rejected triggers", color =  "darkgray")
        ax.set_ylabel("Number of particles", fontsize = 20)
        ax.set_xlabel("Spill index", fontsize = 20)
        ax.legend(fontsize = 16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)\n ({np.array(spill_index).max()} spills)", fontsize = 20)
        self.pdf_global.savefig(fig)
        plt.close()
        
        ## Here read the number of POT per spill from the nxcals data 
        #The spills are always in order, we do not need to re-arrange them, in principle
        try:
            df_pot = pd.read_csv(f"/eos/experiment/wcte/user_data/fiorenti/nxcals/run_{self.run_number}_pot.csv", header = 0)
        except:
            return 0
    
        
        
#         if len(df_pot) != len(spill_index):
#             print(f"There is a discrepancy between the number of recorded spills and the number of spills ({len(spill_index)}) according to the nxcals data ({len(df_pot)}), please check.")
#             raise Error
                  
        n_pot_per_trigger = np.array(df_pot["POT0"])

        n_pot_per_trigger = n_pot_per_trigger[0:len(spill_index)]
        

        

        #decide that there are a bin for each ten spills
#         n_bins = int(max(spill_index)/10)
        n_bins = np.linspace(0, 35, 100)
        n_bins_narrow = np.linspace(0, 35, 300)
        fig, ax = plt.subplots(figsize = (8, 6))
        
        
        bin_centers = (n_bins[1:]+n_bins[:-1])/2
        
        h_e, _, _ = ax.hist(number_e_per_spill/n_pot_per_trigger, bins = n_bins, label = "Electrons", color = "blue", histtype = "step")
        popt, pcov = fit_gaussian(h_e, bin_centers)
        plt.plot(n_bins_narrow, gaussian(n_bins_narrow, *popt), '--', color = "blue", label = f"Gaussian fit: mean {popt[1]:.2f}, std {popt[2]:.2f}")
        
        
        h_mu, _, _ = ax.hist(number_mu_per_spill/n_pot_per_trigger, bins = n_bins, label = "Muons", color = "orange", histtype = "step")
        popt, pcov = fit_gaussian(h_mu, bin_centers)
        plt.plot(n_bins_narrow, gaussian(n_bins_narrow, *popt), '--', color = "orange", label = f"Gaussian fit: mean {popt[1]:.2f}, std {popt[2]:.2f}")
        
        h_pi, _, _ = ax.hist(number_pi_per_spill/n_pot_per_trigger, bins = n_bins, label = "Pions", color = "green", histtype = "step")
        
        popt, pcov = fit_gaussian(h_pi, bin_centers)
        plt.plot(n_bins_narrow, gaussian(n_bins_narrow, *popt), '--', color = "green", label = f"Gaussian fit: mean {popt[1]:.2f}, std {popt[2]:.2f}")
        
        h_p, _, _ = ax.hist(number_p_per_spill/n_pot_per_trigger, bins = n_bins, label = "Protons", color = "red", histtype = "step")
        popt, pcov = fit_gaussian(h_p, bin_centers)
        plt.plot(n_bins_narrow, gaussian(n_bins_narrow, *popt), '--', color = "red", label = f"Gaussian fit: mean {popt[1]:.2f}, std {popt[2]:.2f}")
        
        ax.set_xlabel("Number of particles per 10^10 POT", fontsize = 20)
        ax.set_ylabel("Number of spills", fontsize = 20)
        ax.legend(fontsize = 14)
        ax.grid()
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)", fontsize = 20)
        self.pdf_global.savefig(fig)
        plt.close()
        
        
    def study_beam_structure(self):
        """This function studies the timing difference between consecutive triggers to estimate the probability that a given bunch holds a particle and also study the temporal structure of the beam. It will be useful to correlate this with the scalar information (at some point)"""
        
        #step 1: make a histogram of the timings for a given spill
        #work with all the data so we are not biased by our selection (though we will have for sure the online electron veto impact to keep in mind)  
        
        spill_number = 6
        df_with_diffs = self.df.copy()
        #self.df_all[self.df_all["spill_number"]==spill_number].copy() #.copy()
        df_with_diffs["ref0_dt"] = df_with_diffs.groupby("spill_number")["ref0_time"].transform(lambda x: x - x.min())
        df_with_diffs["ref1_dt"] = df_with_diffs.groupby("spill_number")["ref1_time"].transform(lambda x: x - x.min())


        
        bins = np.linspace (0, 30, 30)
        
        
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.hist(df_with_diffs["ref0_dt"], bins = bins, color = "blue", label = "ref0", histtype="step")
        ax.hist(df_with_diffs["ref1_dt"], bins = bins, color = "black", label = "ref1", histtype="step")
        
        ax.set_xlabel("Time since begining of spill", fontsize=20)
        ax.set_ylabel("Number of triggers", fontsize=20)
        
        ax.legend(fontsize=20)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) \n Ref times distribution, total number of spills {max(df_with_diffs['spill_number'])}")
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        df_with_diffs["t0_times"] = df_with_diffs.groupby("spill_number")["t0_time"].transform(lambda x: x - x.min())
        df_with_diffs["t1_times"] = df_with_diffs.groupby("spill_number")["t1_time"].transform(lambda x: x - x.min())
        df_with_diffs["t5_times"] = df_with_diffs.groupby("spill_number")["t5_time"].transform(lambda x: x - x.min())
        df_with_diffs["t4_times"] = df_with_diffs.groupby("spill_number")["t4_time"].transform(lambda x: x - x.min())


        
        bins = np.linspace (-210, -140, 60)
        
        
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.hist(df_with_diffs["t0_time"], bins = bins, color = "blue", label = "T0 average times", histtype="step")
        ax.hist(df_with_diffs["t1_time"], bins = bins, color = "black", label = "T1 average times", histtype="step")
        ax.hist(df_with_diffs["t4_time"], bins = bins, color = "green", label = "T4 average times", histtype="step")
        ax.hist(df_with_diffs["t5_time"], bins = bins, color = "red", label = "T5 average times", histtype="step")
        
        ax.set_xlabel("Recorded time", fontsize=20)
        ax.set_ylabel("Number of triggers", fontsize=20)
        
        ax.legend(fontsize=16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) \nTS times distribution, Looking at spill {spill_number}") #max(df_with_diffs["spill_number"])}")
        self.pdf_global.savefig(fig)
        plt.close()
        

        
        fig, ax = plt.subplots(figsize = (8, 6))
        ax.plot(df_with_diffs["t0_time"], marker = "x", color = "blue", label = "T0 average times", linestyle = "")
        ax.plot(df_with_diffs["t1_time"], marker = "x", color = "black", label = "T1 average times", linestyle = "")
        ax.plot(df_with_diffs["t4_time"], marker = "x", color = "green", label = "T4 average times", linestyle = "")
        ax.plot(df_with_diffs["t5_time"], marker = "x", color = "red", label = "T5 average times", linestyle = "")
        
        ax.set_ylim(-210, -120)
        
        
        ax.set_ylabel("Time of events in given spill", fontsize=20)
        ax.set_xlabel("Trigger index within spill", fontsize=20)
        
        ax.legend(fontsize=16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) \nTS times distribution, Looking at spill {spill_number}") #max(df_with_diffs["spill_number"])}")
        self.pdf_global.savefig(fig)
        plt.close()
        
        fig, axs = plt.subplots(2, 2, figsize = (18, 16))
        y = np.arange(len(df_with_diffs))
        y_bins = np.linspace(0, len(df_with_diffs), 50)
        
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c) \nTS times distribution, Looking at spill {spill_number}") 
        
        bins_2d = [y_bins, np.linspace(-220, -140, 80)]
        axs[0,0].hist2d(y, df_with_diffs["t0_time"], bins = bins_2d, label = "T0 average times")
        axs[0,0].set_title("T0", fontsize = 20)
        axs[0,1].hist2d(y, df_with_diffs["t1_time"], bins = bins_2d,  label = "T1 average times")
        axs[0,1].set_title("T1", fontsize = 20)
        
        axs[1,0].hist2d(y, df_with_diffs["t4_time"], bins = bins_2d,  label = "T4 average times")
        axs[1,0].set_title("T4", fontsize = 20)
        
        axs[1,1].hist2d(y, df_with_diffs["t5_time"], bins = bins_2d,  label = "T5 average times")
        axs[1,1].set_title("T5", fontsize = 20)
        
        
        for axes in axs:
            for ax in axes:
#                 ax.set_ylim(-210, -120)        
                ax.set_ylabel("Time of events in given spill", fontsize=10)
                ax.set_xlabel("Trigger index within spill", fontsize=10)
#                 ax.legend(fontsize=10)
                #max(df_with_diffs["spill_number"])}")
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        
        
        df_p = self.df[self.df["is_proton"]==True].copy()#
        
        
        fig, ax = plt.subplots(figsize = (8, 6))
        for s in [5, 6,7]:
        #I want to plot the tof of protons as a function of the registered T4 hit 
            ax.plot(df_p[df_p["spill_number"]==s]["t4_time"], df_p[df_p["spill_number"]==s]["tof_t0t4"], marker = "x", label = f"T0-T4 TOF vs T4 time spill {s}", linestyle = "")
        

        
        ax.set_ylabel("T0-T4 TOF", fontsize=20)
        ax.set_xlabel("T4 time of event", fontsize=20)
        ax.set_xlim(-200,-125)
        ax.set_ylim(25, 50)
        ax.legend(fontsize=16)
        
        self.pdf_global.savefig(fig)
        
        plt.close()
        
        
        fig, ax = plt.subplots(figsize = (8, 6))
        for s in [5, 6,7]:
        #I want to plot the tof of protons as a function of the registered T4 hit 
            ax.plot(df_p[df_p["spill_number"]==s]["ref0_time"], df_p[df_p["spill_number"]==s]["tof_t0t4"], marker = "x", label = f"T0-T4 TOF vs ref0 time spill {s}", linestyle = "")
        

        
        ax.set_ylabel("T0-T4 TOF", fontsize=20)
        ax.set_xlabel("ref0 time of event", fontsize=20)
#         ax.set_xlim(-200,-125)
        ax.set_ylim(25, 50)
        ax.legend(fontsize=16)
        
        self.pdf_global.savefig(fig)
        
        plt.close()
        
        
        
        fig, ax = plt.subplots(figsize = (8, 6))
        
        df_p = self.df[self.df["is_proton"]==True].copy()
        #I want to plot the tof of protons as a function of the registered T4 hit 
        ax.plot(df_p["t0_time"], df_p["tof_t0t4"], marker = "x", color = "red", label = "T0-T4 TOF vs T0 time", linestyle = "")
        ax.plot(df_p["t5_time"], df_p["tof_t0t5"], marker = "x", color = "black", label = "T0-T5 TOF vs T5 time", linestyle = "")
        ax.plot(df_p["t4_time"], df_p["tof_t4t5"], marker = "x", color = "green", label = "T4-T5 TOF vs T4 time", linestyle = "")
        
#         ax.plot(df_with_diffs["t0_time"], marker = "x", color = "blue", label = "T0 average times", linestyle = "")
#         ax.plot(df_with_diffs["t1_time"], marker = "x", color = "black", label = "T1 average times", linestyle = "")
#         ax.plot(df_with_diffs["t4_time"], marker = "x", color = "green", label = "T4 average times", linestyle = "")
#         ax.plot(df_with_diffs["t5_time"], marker = "x", color = "red", label = "T5 average times", linestyle = "")
        
        ax.set_ylabel("T0-T4 TOF", fontsize=20)
        ax.set_xlabel("TS time of event", fontsize=20)
#         ax.set_xlim(-200,-125)
#         ax.set_ylim(25, 50)
        ax.set_xlim(-250,-125)
        ax.set_ylim(20, 60)
        
        ax.legend(fontsize=16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)") #max(df_with_diffs["spill_number"])}")
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        bins = np.linspace (0, 60, 60)
        
        fig, ax = plt.subplots(figsize = (8, 6))

        #I want to plot the tof of protons as a function of the registered T4 hit 
        ax.hist(df_p["tof_t0t4"],  bins = bins, color = "blue", label = "T0-T4 TOF (protons)", histtype = "step")
        ax.hist(df_p["tof"], bins = bins, color = "red", label = "T0-T1 TOF(protons)", histtype = "step")
        ax.hist(df_p["tof_t4t5"]+25, bins = bins, color = "green", label = "T4-T5 TOF(protons)", histtype = "step")
        ax.hist(df_p["tof_t0t5"]-30, bins = bins, color = "black", label = "T0-T5 TOF(protons)", histtype = "step")
        ax.hist(df_p["tof_t4t1"]+20, bins = bins, color = "magenta", label = "T1-T4 TOF(protons)", histtype = "step")
        
#         ax.plot(df_with_diffs["t0_time"], marker = "x", color = "blue", label = "T0 average times", linestyle = "")
#         ax.plot(df_with_diffs["t1_time"], marker = "x", color = "black", label = "T1 average times", linestyle = "")
#         ax.plot(df_with_diffs["t4_time"], marker = "x", color = "green", label = "T4 average times", linestyle = "")
#         ax.plot(df_with_diffs["t5_time"], marker = "x", color = "red", label = "T5 average times", linestyle = "")
        
        ax.set_xlabel("TOF", fontsize=20)
        ax.set_ylabel("Number of events", fontsize=20)
        
        ax.legend(fontsize=16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)") #max(df_with_diffs["spill_number"])}")
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        fig, ax = plt.subplots(figsize = (8, 6))
        #I want to plot the tof of protons as a function of the registered T4 hit 
        ax.plot(df_p["t0_time"], df_p["t1_time"], marker = "x", color = "red", label = "T1 vs T0 time", linestyle = "")
        ax.plot(df_p["t4_time"], df_p["t0_time"], marker = "x", color = "black", label = "T4 vs T0 time", linestyle = "")
        ax.plot(df_p["t1_time"], df_p["t4_time"], marker = "x", color = "green", label = "T1 vs T4 time", linestyle = "")
        
#         ax.plot(df_with_diffs["t0_time"], marker = "x", color = "blue", label = "T0 average times", linestyle = "")
#         ax.plot(df_with_diffs["t1_time"], marker = "x", color = "black", label = "T1 average times", linestyle = "")
#         ax.plot(df_with_diffs["t4_time"], marker = "x", color = "green", label = "T4 average times", linestyle = "")
#         ax.plot(df_with_diffs["t5_time"], marker = "x", color = "red", label = "T5 average times", linestyle = "")
        
        ax.set_ylabel("TS time", fontsize=20)
        ax.set_xlabel("TS time", fontsize=20)
        ax.set_xlim(-220,-150)
        ax.set_ylim(-220, -150)
#         ax.set_xlim(-250,-125)
#         ax.set_ylim(20, 60)
        ax.legend(fontsize=16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)") #max(df_with_diffs["spill_number"])}")
        self.pdf_global.savefig(fig)
        plt.close()
        
        #How the T4 tof as a function of the difference between T4 l and T4 r
        fig, ax = plt.subplots(figsize = (8, 6))
        #I want to plot the tof of protons as a function of the registered T4 hit 
        ax.plot(df_p["t4_l"]-df_p["t4_r"], df_p["tof_t0t4"], marker = "x", color = "red", label = "T0-T4 TOF vs T4L-T4R time", linestyle = "")
        ax.plot(df_p["t4_l"]-df_p["t4_r"], df_p["tof_t4t1"]+20, marker = "x", color = "black", label = "T1-T4 TOF vs T4L-T4R time", linestyle = "")

        
        
        ax.set_ylabel("TOF", fontsize=20)
        ax.set_xlabel("T4L-T4R", fontsize=20)
        ax.set_xlim(-15,15)
        ax.set_ylim(0,50)
#         ax.set_ylim(-220, -150)
#         ax.set_xlim(-250,-125)
#         ax.set_ylim(20, 60)
        ax.legend(fontsize=16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)") #max(df_with_diffs["spill_number"])}")
        self.pdf_global.savefig(fig)
        plt.close()
        
         #How the T4 tof as a function of the difference between T4 l and T4 r
        fig, ax = plt.subplots(figsize = (8, 6))
        bins = np.linspace(-15, 15, 30)
        #I want to plot the tof of protons as a function of the registered T4 hit 
        ax.hist(df_p["t4_l"]-df_p["t4_r"], color = "red", bins = bins, label = "T0-T4 TOF vs T4L-T4R time", histtype = "step")              
        
        ax.set_ylabel("Number of events", fontsize=20)
        ax.set_xlabel("T4L-T4R", fontsize=20)
        ax.set_xlim(-15,15)
        ax.legend(fontsize=16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)") #max(df_with_diffs["spill_number"])}")
        self.pdf_global.savefig(fig)
        plt.close()
        
        fig, ax = plt.subplots(figsize = (8, 6))
        #I want to plot the tof of protons as a function of the registered T4 hit 
        ax.scatter(df_p["t4_l"]+40, df_p["t4_l"]-df_p["t0_time"], marker = "x", color = "red", label = "T4L-T0 vs T4L+40ns")
        ax.scatter(df_p["t4_r"], df_p["t4_r"]-df_p["t0_time"], marker = "x", color = "blue", label = "T4R-T0 vs T4R")
        
        ax.scatter(df_p["t4_l"]+20, df_p["t4_r"]-df_p["t0_time"], marker = "x", color = "green", label = "T4R-T0 vs T4L+20ns")
        
        ax.set_xlabel("T4 L or R", fontsize=20)
        ax.set_ylabel("T4L-T0 or TR-T0", fontsize=20)
        ax.set_xlim(-200,-100)
        ax.set_ylim(0,50)
#         ax.set_ylim(-220, -150)
#         ax.set_xlim(-250,-125)
#         ax.set_ylim(20, 60)
        ax.legend(fontsize=16)
        ax.set_title(f"Run {self.run_number} ({self.run_momentum} MeV/c)") #max(df_with_diffs["spill_number"])}")
        self.pdf_global.savefig(fig)
        plt.close()
        
        
        
        
        
        
        
        

        
        

            
           


        
            
            
            
            
            
            
            
            
           
    
    
 


        
            
            
       
        





        
        
        
        
