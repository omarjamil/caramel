#!/bin/bash

while true
do
    n_jobs=( $(ssh mo-ojamil@login.isambard 'qstat -u mo-ojamil' | wc -l) )
    if [ "$n_jobs" -eq "0" ]
    then
        echo "Jobs completed ... fetching model"
        # ./net/home/h06/ojamil/analysis_tools/ML/CRM/caramel/model/torch/fetch_isambard_model.sh
        break
    else

        echo "Jobs running .. " 
        ssh mo-ojamil@login.isambard 'qstat -u mo-ojamil'
    fi
    sleep 60
done  