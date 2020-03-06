#!/bin/bash -l
#SBATCH --mem=8000
#SBATCH --ntasks=2
#SBATCH --time=360
#SBATCH --qos=normal
#SBATCH --wckey=preprocess_data
##SBATCH --mail-user=omar.jamil@metoffice.gov.uk
#SBATCH --mail-type=ALL

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
