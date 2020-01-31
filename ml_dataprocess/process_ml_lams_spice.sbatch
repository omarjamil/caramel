#!/bin/bash -l
#SBATCH --mem=6000
#SBATCH --ntasks=2
#SBATCH --time=360
#SBATCH --export=NONE
#SBATCH --qos=normal
# module load scitools
day=$1
region=$2
type=$3
command="/net/home/h06/ojamil/software/anaconda3/envs/juno/bin/python /net/home/h06/ojamil/analysis_tools/ML/CRM/caramel/ml_dataprocess/process_ml_lams.py --start-day ${day} --region ${region} $3"

echo $command
eval $command
