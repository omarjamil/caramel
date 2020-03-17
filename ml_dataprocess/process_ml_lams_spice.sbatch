#!/bin/bash -l
#SBATCH --mem=6000
#SBATCH --ntasks=4
#SBATCH --time=1000
#SBATCH --export=NONE
#SBATCH --qos=long
# module load scitools
day=$1
region=$2
type=$3
command="/net/home/h06/ojamil/software/anaconda3/envs/juno/bin/python /net/home/h06/ojamil/analysis_tools/ML/CRM/caramel/ml_dataprocess/process_ml_lams.py --start-day ${day} --region ${region} $3"

echo $command
eval $command
