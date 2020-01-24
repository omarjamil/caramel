#!/bin/bash -l
#SBATCH --mem=2000
#SBATCH --ntasks=2
#SBATCH --time=2000
#SBATCH --export=NONE
#SBATCH --qos=long
# module load scitools
day=$1
region=$2
echo process_ml_lams.py ${day} ${region}
# /net/home/h06/ojamil/software/anaconda3/envs/juno/bin/python /net/home/h06/ojamil/analysis_tools/ML/CRM/caramel/data_preprocess_oj/process_ml_lams_single_level.py ${day} ${region}
/net/home/h06/ojamil/software/anaconda3/envs/juno/bin/python /net/home/h06/ojamil/analysis_tools/ML/CRM/caramel/data_preprocess_oj/process_ml_lams.py ${day} ${region} 
# /net/home/h06/ojamil/analysis_tools/ML/CRM/CRM_code/data_preprocess/process_ml_lams.py ${day} ${region}
