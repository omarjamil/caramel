#!/bin/bash -l
#SBATCH --mem=12000
#SBATCH --ntasks=4
#SBATCH --time=70
#SBATCH --qos=normal
#SBATCH --wckey=preprocess_data
#SBATCH --output=/scratch/ojamil/slurmout/%j.out
#SBATCH --error=/scratch/ojamil/slurmout/%j.error

module load scitools
export MPLBACKEND='Agg' # set the backend for use on spice
option=$1
region=$2
stash=$3
command="python /net/home/h06/ojamil/analysis_tools/ML/CRM/caramel/ml_dataprocess/preprocess_data.py ${option} ${region} ${stash}"
# command="python /net/home/h06/ojamil/analysis_tools/ML/CRM/caramel/ml_dataprocess/combined_dataset.py"

echo $command
eval $command
unset MPLBACKEND  # unset the environmental variable
