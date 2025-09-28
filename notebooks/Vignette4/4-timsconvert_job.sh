#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --job-name=Casestudy_timsconvert
#SBATCH --mem=80G
#SBATCH --partition=short
#SBATCH -o output_%j.txt                    # Standard output file
#SBATCH -e error_%j.txt                     # Standard error file
#SBATCH --mail-user=zhu.yiny@northeastern.edu  # Email
#SBATCH --mail-type=ALL                     # Type of email notifications
eval "$(conda shell.bash hook)"
conda activate timsconvert
cd /work/VitekLab/Data/MS/Melanie_manuscript/
timsconvert --input Kidney_MS1_ITO6.d --outdir mouse_kidney_timsconvert_output --compression none --mode centroid --verbose
conda deactivate
