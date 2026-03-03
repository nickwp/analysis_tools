#!/usr/bin/env python3
"""
Self-trigger processing pipeline.

  Step 1 (calibrate) : calibrate_hits.py          -> calibrated_hits/
  Step 2 (dq)        : self_trigger_dq_flags.py   -> dq_flags/

Use --from-step to skip earlier steps and reuse their existing output, e.g.
  --from-step dq   skips calibration
"""

import os
import sys
import subprocess
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

FROM_STEP = {"calibrate": 1, "dq": 2}


def main():
    parser = argparse.ArgumentParser(description="Self-trigger pipeline: calibrate -> dq")
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
    run_dir = os.path.join(args.output_base, args.run_number)
    cal_dir = os.path.join(run_dir, "calibrated_hits")
    dq_dir  = os.path.join(run_dir, "dq_flags")
    for d in [cal_dir, dq_dir, os.path.join(run_dir, "merged")]:
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
        base     = os.path.splitext(os.path.basename(input_file))[0]
        cal_file = os.path.join(cal_dir, f"{base}_calibrated_hits.root")

        print(f"\n{'#'*60}\n  {os.path.basename(input_file)}\n{'#'*60}")

        # ── Step 1: Calibrate hits ────────────────────────────────────────
        if start_step <= 1:
            result = subprocess.run(
                [sys.executable, os.path.join(SCRIPT_DIR, "calibrate_hits.py"),
                 "-i", input_file, "-r", args.run_number, "-o", cal_dir] + not_off + extra
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

        # ── Step 2: Data quality flags ────────────────────────────────────
        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "self_trigger_dq_flags.py"),
             "-i", input_file,
             "-c", cal_dir,
             "-r", args.run_number,
             "-o", dq_dir] + extra
        )
        if result.returncode != 0:
            print(f"[ERROR] DQ flags failed for this file")
            failed.append(input_file)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_total  = len(args.input_files)
    n_failed = len(failed)
    print(f"\n{'='*60}")
    print(f"  {n_total - n_failed}/{n_total} files completed successfully")
    if failed:
        for f in failed:
            print(f"  ✗  {os.path.basename(f)}")
        sys.exit(1)
    print("*** Self trigger pipeline complete ***")


if __name__ == "__main__":
    main()
