#!/bin/bash -l
#SBATCH --mem=48000
#SBATCH --ntasks=4
##SBATCH --output=/scratch/ojamil/spice_train_slurmout
#SBATCH --time=4000
#SBATCH --qos=long
#SBATCH --wckey=caramel
##SBATCH --mail-user=omar.jamil@metoffice.gov.uk
##SBATCH --mail-type=ALL

command="/net/home/h06/ojamil/software/anaconda3/envs/pytorch/bin/python /home/h06/ojamil/analysis_tools/ML/CRM/caramel/model/torch/train.py --epochs 30 --identifier 023001AQT --data-region 023001AQT --batch-size 500 --nb-hidden-layers 6 --data-fraction 1.0 --loss mae --normaliser 163001AQT_normalise --nlevs 45"

echo $command
# eval $command
srun --unbuffered $command
