#!/bin/bash

# Check if an input argument is given
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_str>"
    exit 1
fi

# Assign the input argument to a variable
input_str=$1

# Define the input and output file names
input_file="${input_str}_lm_profiler_stacks.txt"
output_file="${input_str}_lm-flame-graph.svg"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' not found."
    exit 1
fi

# Run the flamegraph.pl script with the specified parameters
./FlameGraph/flamegraph.pl --title "CUDA time" --countname "us." "$input_file" > "$output_file"

# Inform the user of success
echo "Flame graph generated successfully: $output_file"

