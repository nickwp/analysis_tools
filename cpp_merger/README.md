# WCTE Production Merger (`merge_production.cpp`)

A cpp script to merge the output of the python analysis scripts into a single ROOT file.

### Options Explained:
* `-r`, `--run`: The run number (Required).
* `-p`, `--dir`: The production directory containing the `[run_number]/` folders (Required).
* `-i`, `--input`: A list of the raw `WCTE_offline_R...` input root files separated by spaces. Must specify all input files before moving on to the next flag (Required).
* `-t`, `--trigger`: The trigger type. Must be either `"hw"` or `"self"` (Required).
* `-b`, `--beam`: Include to process and merge beam analysis data.
* `-d`, `--debug`: Disables strict event length checks and prints debug information.
* `-h`, `--help`: Prints the help manual.

