#!/bin/bash

# stashes=(16004 12181 10 12182 254 12183 12 12184 272 12189 273 12190 4 24 1202 1205 1207 1208 1235 2201 2207 2205 3217 3225 3226 3234 3236 3245 4203 4204 9217 30405 30406 30461 16222 99181 99182)
# stashes=(16004 12181 10 12182 254 12183 12 12184 272 12189 273 12190)
stashes=(99181 99182)
option=2
region='10N80W'
for s in ${stashes[@]}
do
    command="sbatch preprocess_data.sbatch $option $region $s"
    echo $command
    eval $command
done