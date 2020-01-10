#!/bin/bash -l
#SBATCH --mem=24000
#SBATCH --ntasks=12
##SBATCH --output=/scratch/ojamil/spice_train_slurmout
#SBATCH --time=360
#SBATCH --qos=normal
#SBATCH --wckey=spice_train
#SBATCH --mail-user=omar.jamil@metoffice.gov.uk
#SBATCH --mail-type=ALL

command="/net/home/h06/ojamil/software/anaconda3/envs/pytorch/bin/python /net/home/h06/ojamil/analysis_tools/ML/CRM/caramel/model/torch/q.py --epochs 40"

echo $command
# eval $command
srun --unbuffered $command
