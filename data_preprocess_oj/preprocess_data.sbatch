#!/bin/bash -l
#SBATCH --mem=6000
#SBATCH --ntasks=2
##SBATCH --output=/scratch/ojamil/preprocess_data_slurmout
#SBATCH --time=2000
#SBATCH --qos=long
#SBATCH --wckey=preprocess_data
#SBATCH --mail-user=omar.jamil@metoffice.gov.uk
#SBATCH --mail-type=ALL
module load scitools
export MPLBACKEND='Agg' # set the backend for use on spice
command="python /net/home/h06/ojamil/analysis_tools/ML/CRM/CRM_code/data_preprocess/preprocess_data.py 3 50S69W $1"
echo $command
eval $command
unset MPLBACKEND  # unset the environmental variable
