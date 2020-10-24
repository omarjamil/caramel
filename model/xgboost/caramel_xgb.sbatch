#!/bin/bash -l
#SBATCH --mem=12000
#SBATCH --ntasks=4
##SBATCH --output=/scratch/ojamil/caramel_xgb_slurmout
#SBATCH --time=360
#SBATCH --qos=normal
#SBATCH --wckey=caramel_xgb
#SBATCH --mail-user=omar.jamil@metoffice.gov.uk
##SBATCH --mail-type=ALL

command="/net/home/h06/ojamil/software/anaconda3/envs/xgboost/bin/python caramel_gb.py"
echo $command
eval $command
