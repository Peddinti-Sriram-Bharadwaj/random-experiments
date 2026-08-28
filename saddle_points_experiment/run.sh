#!/bin/bash
set -e

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo "Starting Saddle Points in High-Dimensional Spaces Experiment"
echo "Using Conda Environment: rl_sim"
echo "=========================================================="

# Ensure dependencies are installed in the conda environment
echo "Ensuring torch, matplotlib, and numpy are installed in 'rl_sim'..."
conda run -n rl_sim pip install torch matplotlib numpy

# Run the search experiment
echo "Running the experiment to find critical points and calculate eigenvalues..."
conda run -n rl_sim python experiment.py --dimensions 2 3 5 10 20 50 100 --starts 300 --out results.json

# Generate the plots
echo "Generating plots..."
conda run -n rl_sim python plot.py --input results.json --out_dir plots

echo "=========================================================="
echo "Experiment finished successfully!"
echo "Plots generated in: $DIR/plots/"
echo "=========================================================="
