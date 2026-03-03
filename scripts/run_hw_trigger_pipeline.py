#!/usr/bin/env python3
"""
Hardware-trigger processing pipeline.

  Step 1 (wf)        : hw_trigger_wf_processing.py  -> processed_waveforms/
  Step 2 (calibrate) : calibrate_hits.py                     -> calibrated_hits/
  Step 3 (dq)        : hw_trigger_dq_flags.py          -> dq_flags/

Use --from-step to skip earlier steps and reuse their existing output, e.g.
  --from-step calibrate   skips waveform processing
  --from-step dq          skips waveform processing and calibration
"""

import os
import sys
import subprocess
import argparse
from analysis_tools.production_utils import read_status_json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FROM_STEP = {"wf": 1, "calibrate": 2, "dq": 3}

# ── QC thresholds ─────────────────────────────────────────────────────────────
# Files with metrics exceeding these will not be eligible for merging.
QC_BAD_TRIG_PCT_WARN = 5.0
QC_BAD_HIT_PCT_WARN  = 10.0


def main():
    parser = argparse.ArgumentParser(description="Hardware-trigger pipeline: wf -> calibrate -> dq")
    parser.add_argument("-i", "--input_files", required=True, nargs="+",
                        help="Raw WCTEReadoutWindows ROOT input files")
    parser.add_argument("-r", "--run_number", required=True)
    parser.add_argument("-o", "--output_base", required=True,
                        help="Base output directory; <run_number>/ subdir is created automatically")
    parser.add_argument("--from-step", choices=FROM_STEP.keys(), default=None, dest="from_step",
                        help="Start from this step, reusing outputs of all earlier steps")
    parser.add_argument("--not_official_const", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    start_step = FROM_STEP[args.from_step] if args.from_step else 1

    # Output directories
    run_dir       = os.path.join(args.output_base, args.run_number)
    wf_dir        = os.path.join(run_dir, "processed_waveforms")
    cal_dir       = os.path.join(run_dir, "calibrated_hits")
    dq_dir        = os.path.join(run_dir, "dq_flags")
    for d in [wf_dir, cal_dir, dq_dir, os.path.join(run_dir, "merged")]:
        os.makedirs(d, exist_ok=True)

    # Common flags forwarded to sub-scripts
    extra   = ["--debug"] if args.debug else []
    not_off = ["--not_official_const"] if args.not_official_const else []

    # Validate run number appears in every input filename
    for f in args.input_files:
        if f"R{args.run_number}" not in os.path.basename(f):
            print(f"[ERROR] '{f}' does not match run number R{args.run_number}")
            sys.exit(1)

    failed = []

    for input_file in args.input_files:
        base           = os.path.splitext(os.path.basename(input_file))[0]
        wf_file        = os.path.join(wf_dir, f"{base}_processed_waveforms.root")
        cal_file       = os.path.join(cal_dir, f"{base}_processed_waveforms_calibrated_hits.root")

        print(f"\n{'#'*60}\n  {os.path.basename(input_file)}\n{'#'*60}")

        # ── Step 1: Waveform processing ───────────────────────────────────
        if start_step <= 1:
            result = subprocess.run(
                [sys.executable, os.path.join(SCRIPT_DIR, "hw_trigger_wf_processing.py"),
                 "-i", input_file, "-o", wf_dir] + extra
            )
            if result.returncode != 0:
                print(f"[ERROR] Waveform processing failed — skipping remaining steps for this file")
                failed.append(input_file)
                continue
        else:
            if not os.path.exists(wf_file):
                print(f"[ERROR] --from-step {args.from_step} requires wf output but not found:\n  {wf_file}")
                failed.append(input_file)
                continue
            print(f"[SKIP] wf — using {os.path.basename(wf_file)}")

        # ── Step 2: Calibrate hits ────────────────────────────────────────
        if start_step <= 2:
            result = subprocess.run(
                [sys.executable, os.path.join(SCRIPT_DIR, "calibrate_hits.py"),
                 "-i", wf_file, "-r", args.run_number, "-o", cal_dir] + not_off + extra
            )
            if result.returncode != 0:
                print(f"[ERROR] Calibration failed — skipping remaining steps for this file")
                failed.append(input_file)
                continue
        else:
            if not os.path.exists(cal_file):
                print(f"[ERROR] --from-step {args.from_step} requires calibrated output but not found:\n  {cal_file}")
                failed.append(input_file)
                continue
            print(f"[SKIP] calibrate — using {os.path.basename(cal_file)}")

        # ── Step 3: Data quality flags ────────────────────────────────────
        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "hw_trigger_dq_flags.py"),
             "-i", input_file,
             "-c", cal_dir,
             "-hw", wf_dir,
             "-r", args.run_number,
             "-o", dq_dir] + extra
        )
        if result.returncode != 0:
            print(f"[ERROR] DQ flags failed for this file")
            failed.append(input_file)
        else:
            dq_file = os.path.join(dq_dir, f"{base}_hw_trigger_dq_flags.root")
            sidecar = read_status_json(dq_file)
            if sidecar:
                m = sidecar["metrics"]
                warnings = []
                if m.get("bad_trig_pct", 0) > QC_BAD_TRIG_PCT_WARN:
                    warnings.append(f"bad triggers {m['bad_trig_pct']:.1f}% > {QC_BAD_TRIG_PCT_WARN}%")
                if m.get("bad_hit_pct", 0) > QC_BAD_HIT_PCT_WARN:
                    warnings.append(f"bad hits {m['bad_hit_pct']:.1f}% > {QC_BAD_HIT_PCT_WARN}%")
                if warnings:
                    print(f"  QC WARNING: {'; '.join(warnings)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n_total  = len(args.input_files)
    n_failed = len(failed)
    print(f"\n{'='*60}")
    print(f"  {n_total - n_failed}/{n_total} files completed successfully")
    if failed:
        for f in failed:
            print(f"  ✗  {os.path.basename(f)}")
        sys.exit(1)
    print("*** Hardware trigger pipeline complete ***")


if __name__ == "__main__":
    main()
