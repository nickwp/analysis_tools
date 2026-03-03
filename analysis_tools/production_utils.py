"""
production_utils.py

Shared utilities for WCTE data production scripts.
"""

import os
import subprocess
import hashlib
import json
import numpy as np
from .wcte_pmt_mapping import PMTMapping



# ── Status sidecar ────────────────────────────────────────────────────────────
def write_status_json(output_root_path, metrics, warnings=None, errors=None):
    """Write a JSON sidecar file alongside an output ROOT file.

    The sidecar has the same basename as the ROOT file with _status.json suffix.
    status is "ok", "warning", or "error" depending on warnings/errors lists.

    Parameters
    ----------
    output_root_path : str  — absolute path to the output ROOT file
    metrics          : dict — key QC numbers (n_triggers, bad_trig_pct, etc.)
    warnings         : list of str — quality threshold violations (non-fatal)
    errors           : list of str — hard errors (file should not be used)
    """
    warnings = warnings or []
    errors   = errors   or []

    if errors:
        status = "error"
    elif warnings:
        status = "warning"
    else:
        status = "ok"

    payload = {
        "status":   status,
        "metrics":  metrics,
        "warnings": warnings,
        "errors":   errors,
    }

    base = os.path.splitext(output_root_path)[0]
    json_path = f"{base}_status.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    return json_path, status


def read_status_json(output_root_path):
    """Read the status sidecar for a given output ROOT file.
    Returns the parsed dict, or None if no sidecar exists."""
    base = os.path.splitext(output_root_path)[0]
    json_path = f"{base}_status.json"
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        return json.load(f)


# ── Git provenance ─────────────────────────────────────────────────────────────
def get_git_descriptor(debug=False):
    """Return the git describe string for the current repo.
    Raises an exception if the repo has uncommitted changes (unless debug=True)."""
    try:
        desc = subprocess.check_output(
            ["git", "describe", "--always", "--tags"],
            stderr=subprocess.STDOUT
        ).decode().strip()

        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.STDOUT
        ).decode().strip()
        if status:
            if debug:
                print("Warning: Repository has uncommitted changes, but continuing due to debug mode.")
            else:
                raise Exception("Repository has uncommitted changes")
        return desc
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Git command failed") from e


# ── File utilities ─────────────────────────────────────────────────────────────
def file_sha256(path, chunk_size=1024 * 1024):
    """Return the SHA-256 hex digest of a file (used to tag which slow control
    file was used in a processing run)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Slow control / run database ────────────────────────────────────────────────
def get_run_database_data(json_path, run_number):
    """Return the slow control run database dict for a given run number.
    Raises ValueError if the run is not found."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    run_key = str(run_number)
    if run_key not in data:
        raise ValueError(f"Run number {run_number} not found in run list: {json_path}")
    return data[run_key]


def get_stable_mpmt_list_slow_control(run_data):
    """Return (enabled_channels, channel_mask) as sets in card-channel format.
    Stable channels = enabled_channels - channel_mask."""
    enabled_channels = set(run_data["enabled_channels"])
    channel_mask     = set(run_data["channel_mask"])
    return enabled_channels, channel_mask


# ── PMT channel mapping ────────────────────────────────────────────────────────
def slot_pos_from_card_chan_list(card_chan_list):
    """Convert a collection of card-channel identifiers to slot-position identifiers.

    card-channel format : card*100 + channel  (e.g. 201 → card 2, channel 1)
    slot-position format: slot*100 + position (e.g. 203 → slot 2, position 3)

    Returns a numpy array of slot-position integers."""
    mapping = PMTMapping()
    slot_pos_list = []
    for ch in card_chan_list:
        card     = ch // 100
        pmt_chan = ch % 100
        slot, pmt_pos = mapping.get_slot_pmt_pos_from_card_pmt_chan(card, pmt_chan)
        slot_pos_list.append(100 * slot + pmt_pos)
    return np.array(slot_pos_list)


# ── Trigger-level data quality masks ──────────────────────────────────────────
def get_slow_control_trigger_mask(run_number_str, trigger_times, run_data):
    """Apply slow control data quality flags to a vector of trigger times.

    Parameters
    ----------
    run_number_str : str
    trigger_times  : np.ndarray  (nanoseconds)
    run_data       : dict  — slow control entry for this run

    Returns
    -------
    np.ndarray of bool — True means keep, False means discard.
    """
    BUFFER = 15  # seconds added either side of each problem window
    bad_mask = np.zeros(len(trigger_times), dtype=bool)
    for problem in run_data["problems"]:
        start = (problem[0] - run_data["start"] - BUFFER) * 1e9,
        end   = (problem[1] - run_data["start"] + BUFFER) * 1e9,
        prob  = problem[2]
        if "dropped" in prob:
            bad_mask = np.logical_or(bad_mask,
                                     np.logical_and(trigger_times > start, trigger_times < end))
        elif "no_data" in prob or "Status." in prob or "bad_flow" in prob:
            pass  # channel-level issues — handled separately via the channel mask
        elif "crashed" in prob:
            bad_mask = np.logical_or(bad_mask, trigger_times > (run_data["end"] - 30))
        else:
            raise ValueError(f"Unhandled slow control problem type: '{prob}'")
    return np.logical_not(bad_mask)


def get_67ms_mask(run_number_str, trigger_times):
    """Apply the 67 ms missing-trigger mask for runs before R1841.

    Returns a boolean array — True means keep, False means discard."""
    if int(run_number_str) < 1841:
        return trigger_times % 67108864 > 1e7
    else:
        return np.ones(len(trigger_times), dtype=bool)
