"""store_bad_channels.py

Script for uploading manually-identified problematic PMT channels to the
WCTE calibration database as 'pmt_state' entries.

Once an upload has been performed, wrap the call in `if False:` so it is
preserved for reference but cannot be re-run accidentally.

Usage
-----
Run directly:
    python store_bad_channels.py

Channel IDs use the slot*100 + position convention (e.g. slot 2, pos 3 -> 203),
consistent with the rest of the analysis_tools codebase.
"""

import json
from analysis_tools import CalibrationDBInterface
from analysis_tools.wcte_pmt_mapping import PMTMapping

SLOW_CONTROL_JSON = '/eos/experiment/wcte/configuration/slow_control_summary/all_run_list_v7.json'


def get_hw_trigger_runs_from(min_run_number):
    """Return a sorted list of run numbers that are hardware trigger runs >= min_run_number."""
    with open(SLOW_CONTROL_JSON) as f:
        data = json.load(f)
    hw_runs = []
    for run_str, run_data in data.items():
        run_num = int(run_str)
        if run_num < min_run_number:
            continue
        trigger_name = run_data.get('trigger_name', '')
        if 'hardware' in trigger_name.lower():
            hw_runs.append(run_num)
    return sorted(hw_runs)


def mpmt_card89_all_channels_bad(caldb):
    """Upload all channels (0-18) of mPMT card 89 as bad PMTs for every
    hardware trigger run from R2286 onwards.

    Channels are mapped from card/channel -> slot*100+pos using PMTMapping
    before being submitted to the database.

    Identified 2026-03-12: card 89 offline in all HW trigger runs >= R2286.
    """
    mapping = PMTMapping()

    # Map card 89, channels 0-18 to global PMT IDs (slot*100 + position)
    bad_pmts = []
    for chan in range(19):  # channels 0 to 18 inclusive
        slot, pmt_pos = mapping.get_slot_pmt_pos_from_card_pmt_chan(89, chan)
        glb_pmt_id = 100 * slot + pmt_pos
        bad_pmts.append(glb_pmt_id)
    print(f"Mapped card 89 channels 0-18 -> global PMT IDs: {bad_pmts}")

    hw_runs = get_hw_trigger_runs_from(min_run_number=2286)
    print(f"Found {len(hw_runs)} hardware trigger runs >= R2286: {hw_runs}")

    for run_number in hw_runs:
        print(f"Uploading {len(bad_pmts)} bad PMTs for run {run_number}...")
        caldb.post_bad_pmts(bad_pmts=bad_pmts, run_number=run_number)


def mpmt_card17_all_channels_bad(caldb):
    """Upload all channels (0-18) of mPMT card 17 as bad PMTs for
    runs 1825, 1827, 1829, 1831.

    Channels are mapped from card/channel -> slot*100+pos using PMTMapping
    before being submitted to the database.
    """
    mapping = PMTMapping()

    # Map card 17, channels 0-18 to global PMT IDs (slot*100 + position)
    bad_pmts = []
    for chan in range(19):  # channels 0 to 18 inclusive
        slot, pmt_pos = mapping.get_slot_pmt_pos_from_card_pmt_chan(17, chan)
        glb_pmt_id = 100 * slot + pmt_pos
        bad_pmts.append(glb_pmt_id)
    print(f"Mapped card 17 channels 0-18 -> global PMT IDs: {bad_pmts}")

    target_runs = [1825, 1827, 1829, 1831]
    
    for run_number in target_runs:
        print(f"Uploading {len(bad_pmts)} bad PMTs for run {run_number}...")
        caldb.post_bad_pmts(bad_pmts=bad_pmts, run_number=run_number)


if __name__ == "__main__":
    caldb = CalibrationDBInterface(credential_path="./.wctecaldb.analysistcredential")

    # ---- Uploaded 2026-03-12: card 89 all channels, HW runs >= R2286 ----
    if False:
        # already uploaded
        mpmt_card89_all_channels_bad(caldb)

    # ---- Upload: card 17 all channels, runs [1825, 1827, 1829, 1831] ----
    if False:
        # already uploaded
        mpmt_card17_all_channels_bad(caldb)

    # ---- Query: verify uploaded bad PMTs for each HW run >= R2286 ----
    if True:
        hw_runs = get_hw_trigger_runs_from(min_run_number=2286)
        print(f"Checking {len(hw_runs)} hardware trigger runs >= R2286")
        for run_number in hw_runs:
            pmt_state_data, revision_id, insert_time = caldb.get_calibration_constants(
                run_number=run_number, time=0, calibration_name="pmt_state", official=1
            )
            print("Run ", run_number, "pmt_state_data", pmt_state_data)
            # bad_pmt_ids = [entry["glb_pmt_id"] for entry in pmt_state_data]
            # status = "OK" if len(bad_pmt_ids) == 19 else f"UNEXPECTED COUNT ({len(bad_pmt_ids)})"
            # print(f"  Run {run_number}: {len(bad_pmt_ids)} bad PMTs [{status}] -> {sorted(bad_pmt_ids)}")

    # ---- Query: verify uploaded bad PMTs for runs [1825, 1827, 1829, 1831] ----
    if True:
        target_runs = [1825, 1827, 1829, 1831]
        print(f"Checking {len(target_runs)} runs for card 17: {target_runs}")
        for run_number in target_runs:
            pmt_state_data, revision_id, insert_time = caldb.get_calibration_constants(
                run_number=run_number, time=0, calibration_name="pmt_state", official=1
            )
            print("Run ", run_number, "pmt_state_data", pmt_state_data)
