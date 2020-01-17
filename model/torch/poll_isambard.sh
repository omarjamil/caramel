#!/bin/bash

while 1
do
    n_jobs=( $(qstat -u mo-ojamil | wc -l) )
    if [ "$n_jobs" --eq "0" ]
    then
        echo "Job completed"
        break
    fi
    sleep 5
done  