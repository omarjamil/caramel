#!/bin/bash

region="10S40E" #"50N144W" #"50N144E" #"10S160W" "50S69W"
for i in {1..31}
do
    sbatch process_ml_lams_spice.sbatch $i $region
done
