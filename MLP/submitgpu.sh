#!/bin/sh
#BSUB -q gpuv100
#BSUB -J hpc_assignment2_gpu
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30
#BSUB -o hpc_assignment2_gpu_%J.out
#BSUB -e hpc_assignment2_gpu_%J.err

source /dtu/projects/HPC4Wind_2025/conda/conda_init.sh
conda activate hpc4wind

#script here
python -u MLPS.py