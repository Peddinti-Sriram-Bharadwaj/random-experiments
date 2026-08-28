#!/bin/bash
set -e

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=========================================================="
echo "Starting Escaping Saddle Points Optimization Experiment"
echo "Using Conda Environment: rl_sim"
echo "=========================================================="

# Ensure required packages are installed (numpy, matplotlib)
echo "Ensuring numpy and matplotlib are installed in 'rl_sim'..."
conda run -n rl_sim pip install numpy matplotlib

# Run the optimization simulation
echo "Running the optimization paths..."
conda run -n rl_sim python experiment.py --x0 0.1 0.001 --steps 200 --lr_gd 0.1 --lr_gdm 0.1 --lr_newton 1.0 --lr_sfn 1.0 --out optimization_results.json

# Generate the plots (contour, 3d surface, loss, gradient norm)
echo "Generating plots (2D contour, 3D surface, loss, and gradient norm)..."
conda run -n rl_sim python plot.py --input optimization_results.json --out_dir plots

echo "=========================================================="
echo "Experiment finished successfully!"
echo "Plots generated in: $DIR/plots/"
echo "=========================================================="
