#!/bin/bash -l
#SBATCH --mem=2000
#SBATCH --ntasks=2
#SBATCH --time=360
#SBATCH --export=NONE
#SBATCH --qos=normal
#SBATCH --output=/scratch/ojamil/slurmout/%j.getf
## SBATCH --error=/scratch/ojamil/slurmout/%j.getf

# April 2017 (u-bu076)
# Oct 2017 (u-bu077)
r=$1
echo "Extracting for $r"
for d in {1..15}
do
    # moo get moose:/devfc/u-bs572/field.pp/201701${d}*_${r}_km1p5_ra1m_pver*.pp /project/spice/radiation/ML/CRM/data/u-bs572/$r
    # moo get moose:/devfc/u-bs573/field.pp/201707${d}*_${r}_km1p5_ra1m_pver*.pp /project/spice/radiation/ML/CRM/data/u-bs573/$r
    moo get moose:/devfc/u-bu076/field.pp/201704${d}*_${r}_km1p5_ra1m_pver*.pp /project/spice/radiation/ML/CRM/data/u-bu076/$r
    # moo get moose:/devfc/u-bu077/field.pp/201710${d}*_${r}_km1p5_ra1m_pver*.pp /project/spice/radiation/ML/CRM/data/u-bu077/$r
done