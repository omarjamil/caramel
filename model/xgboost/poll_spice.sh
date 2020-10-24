#!/bin/bash

while true
do
    n_jobs=( $(squeue -u ojamil | wc -l) )
    if [ "$n_jobs" -eq "1" ]
    then
        echo "Spice jobs completed"
        # ./net/home/h06/ojamil/analysis_tools/ML/CRM/caramel/model/torch/fetch_isambard_model.sh
        break
    else
        echo "Jobs running .. " 
        squeue -u ojamil
    fi
    sleep 60
done  