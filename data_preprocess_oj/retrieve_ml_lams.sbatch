#!/bin/bash -l
#SBATCH --mem=1000
#SBATCH --ntasks=2
#SBATCH --time=1000
#SBATCH --export=NONE
#SBATCH --qos=long
module load scitools/default-current
command="/home/h06/ojamil/analysis_tools/ML/CRM/CRM_code/data_preprocess/retrieve_ml_lams.py -r $1 -s $2 -d $3"
echo $command
eval $command


