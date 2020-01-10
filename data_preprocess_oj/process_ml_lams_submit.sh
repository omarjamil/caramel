#!/bin/bash

region="50S69W" #"10S40E" #"50N144W" #"50N144E" #"10S160W"
for i in {1..31}
do
    sbatch process_ml_lams_spice.sbatch $i $region
done
