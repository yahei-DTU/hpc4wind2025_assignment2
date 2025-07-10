#!/bin/bash
#BSUB -q hpc
#BSUB -J assignment2
#BSUB -n 1
#BSUB -W 03:00
#BSUB -u torhe@dtu.dk
#BSUB -R "rusage[mem=64GB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -o assignment2_1_%J.out
#BSUB -r assignment2_1_%J.err

source /dtu/projects/HPC4Wind_2025/conda/conda_init.sh
conda activate hpc4wind

#script here
python -u MLPS.py