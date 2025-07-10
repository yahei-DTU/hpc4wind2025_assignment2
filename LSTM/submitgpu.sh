#!/bin/sh
#BSUB -q gpuv100
#BSUB -J LSTM_GPU
#BSUB -P hpc4wind2025
#BSUB -n 4
#BSUB -u yahei@dtu.dk
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -R "select[gpu32gb]"
#BSUB -o Output/LSTM_GPU_%J.out
#BSUB -e Output/LSTM_GPU_%J.err

source /dtu/projects/HPC4Wind_2025/conda/conda_init.sh
conda activate hpc4wind

#script here
python /zhome/25/9/211757/hpc4wind2025_assignment2/LSTM/LSTM.py
