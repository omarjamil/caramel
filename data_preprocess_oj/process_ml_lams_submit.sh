#!/bin/bash
#region="10N80W" # "80S90W"  "50S69W" "10S160W" "50N144E"

region="80S90W" 
for i in {1..31}
do
    sbatch process_ml_lams_spice.sbatch $i $region
done
