#!/bin/bash -l
#SBATCH --mem=60000
#SBATCH --ntasks=24
#SBATCH --time=360
#SBATCH --qos=normal
#SBATCH --wckey=combined_dataset
#SBATCH --output=/scratch/ojamil/slurmout/%j.cmbdo
#SBATCH --error=/scratch/ojamil/slurmout/%j.cmbde

module load scitools
export MPLBACKEND='Agg' # set the backend for use on spice
command="python /net/home/h06/ojamil/analysis_tools/ML/CRM/caramel/ml_dataprocess/combined_dataset.py"
echo $command
eval $command