#!/bin/bash -l
#SBATCH --mem=72000
#SBATCH --ntasks=12
##SBATCH --output=/scratch/ojamil/spice_train_slurmout
#SBATCH --time=4200
#SBATCH --qos=long
#SBATCH --wckey=caramel
##SBATCH --mail-user=omar.jamil@metoffice.gov.uk
##SBATCH --mail-type=ALL

# command="/net/home/h06/ojamil/software/anaconda3/envs/pytorch/bin/python /home/h06/ojamil/analysis_tools/ML/CRM/caramel/model/torch/train.py --epochs 10 --identifier 023001AQT --data-region 023001AQT --batch-size 500 --nb-hidden-layers 6 --data-fraction 1.0 --loss mse --normaliser 163001AQT_normalise --nlevs 45"
# command="/net/home/h06/ojamil/software/anaconda3/envs/pytorch/bin/python -u /home/h06/ojamil/analysis_tools/ML/CRM/caramel/model/torch/train.py --epochs 25 --identifier 023001AQT --data-region 023001AQT --batch-size 500 --nb-hidden-layers 7 --data-fraction 1.0 --loss mae --normaliser 023001AQT_normalise --nlevs 55"
command="/net/home/h06/ojamil/software/anaconda3/envs/pytorch/bin/python -u /home/h06/ojamil/analysis_tools/ML/CRM/caramel/model/torch/train_stacked.py --epochs 25 --identifier 023001AQTS --data-region 023001AQTS --batch-size 106 --nb-hidden-layers 7 --samples-fraction 1.0 --data-fraction 1.0 --loss mae --normaliser 023001AQT_normalise --nlevs 55"
# command="/net/home/h06/ojamil/software/anaconda3/envs/pytorch/bin/python /home/h06/ojamil/analysis_tools/ML/CRM/caramel/model/torch/train_cnn.py --epochs 20 --identifier 023001AQT --data-region 023001AQT --batch-size 500 --data-fraction 1.0 --loss mse --normaliser 163001AQT_normalise --nlevs 45 --n-filters 20 --n-nodes 10"

echo $command
# eval $command
srun --unbuffered $command
