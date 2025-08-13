#!/bin/bash

# Clean old files
rm -rf *.h5

# Generate multiple results in parallel with frequency as continuation parameter
python3 parallel_step.py S

# Create S-curve from the results
arg_list=()
for i in $(seq 10.0 0.2 24.0); do
    arg_list+=("frequency_step_frequency_${i}00.h5")
done

python3 ../../postprocess/Scurve.py "${arg_list[@]}"

# Move to the results directory
mkdir -p results
mv *.h5 results/