#!/bin/bash -l
#SBATCH --mem=80000
#SBATCH --ntasks=4
#SBATCH --time=30
#SBATCH --qos=normal
#SBATCH --wckey=combined_dataset
#SBATCH --output=/scratch/ojamil/slurmout/%j.cmbdo
#SBATCH --error=/scratch/ojamil/slurmout/%j.cmbde

module load scitools
export MPLBACKEND='Agg' # set the backend for use on spice
command="python -u /net/home/h06/ojamil/analysis_tools/ML/CRM/caramel/ml_dataprocess/combined_dataset.py"
echo $command
eval $command