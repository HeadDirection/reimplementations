#!/bin/bash

if [ $# -ne 1 ]; then 
    echo "Usage: ./run_plotting.sh <number of iterations>"
fi

for i in $(seq 1 $1); do
    echo "Running plot iteration $i"
    du -sh hidden_states.npy
    python3 plot_activations.py
done 
