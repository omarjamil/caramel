#!/bin/bash -l
#SBATCH --mem=8000
#SBATCH --ntasks=4
#SBATCH --time=360
#SBATCH --export=NONE
#SBATCH --qos=normal
#SBATCH --output=/scratch/ojamil/slurmout/%j.pmlo
#SBATCH --error=/scratch/ojamil/slurmout/%j.pmle
# module load scitools
day=$1
month=$2
region=$3
type=$4
roseid=$5
command="/net/home/h06/ojamil/software/anaconda3/envs/juno/bin/python /net/home/h06/ojamil/analysis_tools/ML/CRM/caramel/ml_dataprocess/process_ml_lams.py --start-day ${day} --start-month ${month} --region ${region} $4 --rose-id ${roseid}"

echo $command
eval $command
