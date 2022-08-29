#!/bin/bash

# Calls run_experiments.py for each VLM, with the correct conda environment
# Usage: ./run_experiments.sh experiment_parameters.json

conda run -n VLM_CLIP --no-capture-output python run_experiments.py CLIP $1 || exit 1
conda run -n VLM_MILES --no-capture-output python run_experiments.py MILES $1 || exit 1
conda run -n videoclip --no-capture-output python run_experiments.py VideoCLIP $1 || exit 1
conda run -n VLM_VTTWINS --no-capture-output python run_experiments.py VT-TWINS $1 || exit 1
conda run -n VLM_UNIVL --no-capture-output python run_experiments.py UniVL $1 || exit 1