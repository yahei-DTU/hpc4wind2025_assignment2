#!/bin/bash
#BSUB -q hpc
#BSUB -J LSTM
#BSUB -n 24
#BSUB -W 03:00
#BSUB -u yahei@dtu.dk
#BSUB -P hpc4wind2025
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[model==XeonE5_2650v4]"
#BSUB -R "span[hosts=1]"

#BSUB -o Output/LSTM_%J.out
#BSUB -e Output/LSTM_%J.err

source /dtu/projects/HPC4Wind_2025/conda/conda_init.sh
conda activate hpc4wind

#script here
python /zhome/25/9/211757/hpc4wind2025_assignment2/LSTM/LSTM.py
