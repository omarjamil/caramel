#!/bin/bash

stashes=(16004 12181 10 12182 254 12183 12 12184 272 12189 273 12190 4 24 1202 1205 1207 1208 1235 2201 2207 2205 3217 3225 3226 3234 3236 3245 4203 4204 9217 30405 30406 30461 16222 99181 99182)
# stashes=(16004 12181 10 12182 254 12183 12 12184 272 12189 273 12190)
# stashes=(99181 99182)
option=5

regions=('50N144W' '10S120W' '10N120W' '20S112W' '0N90W' '30N153W' '80S90E' '40S90W' '10N80E' '0N90E' '10S80W' '50S0E' '70S0E' '0N162W' '30S102W' '40N90E' '70S120E' '60N135W' '70S120W' '50N72E' '40S30E' '10N40W' '20S22E' '10N40E' '30N105E' '20N157E' '40S150W' '40S30W' '30S153W' '0N54E' '50S144W' '20S67E' '60N45E'  '10S40W' '60S45E' '20S112E' '20N67W' '0N126W' '0N126E' '10N0E' '20S157W' '50N0E' '20N118W' '50S144E' '30N51W' '20N22W' '40S90E' '30S153E' '30N0E' '20N157W' '40N150W' '20N67E' '0N162E' '70N120E' '30N51E' '20S67W' '20N22E' '70N0E' '40S150E' '20S22W' '80N90E' '40N150E' '70N120W' '10N160W' '50N72W' '60S135W' '60S39W' '80N90W' '60S135E' '30S102E' '10S0E' '10N160E' '30N102W' '20N112E' '10S160E' '10S80E' '60N135E' '30S51E' '30N153E'  '10N120E' '0N18W' '30S51W' '10S120E' '40N30E' '40N90W' '0N54W' '30S0E' '60N45W' '20S157E' '50S72E' '0N18E' '40N30W')

# regions=('20S157W' '50N0E' '20N118W' '50S144E' '30N51W' '20N22W' '40S90E' '30S153E' '30N0E' '20N157W' '40N150W' '20N67E' '0N162E' '70N120E' '30N51E' '20S67W' '20N22E' '70N0E' '40S150E' '20S22W' '80N90E' '40N150E' '70N120W' '10N160W' '50N72W' '60S135W' '60S39W' '80N90W' '60S135E' '30S102E' '10S0E' '10N160E' '30N102W' '20N112E' '10S160E' '10S80E' '60N135E' '30S51E' '30N153E'  '10N120E' '0N18W' '30S51W' '10S120E' '40N30E' '40N90W' '0N54W' '30S0E' '60N45W' '20S157E' '50S72E' '0N18E' '40N30W')
# regions=('0N162E' '70N120E' '30N51E' '20S67W' '20N22E' '70N0E' '40S150E' '20S22W' '80N90E' '40N150E' '70N120W' '10N160W' '50N72W' '60S135W' '60S39W' '80N90W' '60S135E' '30S102E' '10S0E' '10N160E' '30N102W' '20N112E' '10S160E' '10S80E' '60N135E' '30S51E' '30N153E'  '10N120E' '0N18W' '30S51W' '10S120E' '40N30E' '40N90W' '0N54W' '30S0E' '60N45W' '20S157E' '50S72E' '0N18E' '40N30W')
# for region in ${regions[@]}
# do
#     for s in ${stashes[@]}
#     do
#         command="sbatch preprocess_data.sbatch $option $region $s"
#         echo $command q
#         eval $command
#     done
# done

# poll submit to avoid job numbers limit
# r=0
# while [ $r -lt 52 ]
# do
#     region=${regions[r]}
#     sq=`squeue -u ojamil | wc -l`
#     let qlength=$((sq-1))
#     let gap=$((999-qlength))
#     if (( gap > 40 ))
#     then
#         for s in ${stashes[@]}
#         do
#             command="sbatch preprocess_data.sbatch $option $region $s"
#             echo $command
#             eval $command
#         done
#         r=$((r+1))
#     fi
#     sleep 2.5
# done

# for region in ${regions[@]}
# do
#     for s in ${stashes[@]}
#     do
#         command="bash preprocess_data.sbatch $option $region $s"
#         echo $command
#         eval $command
#     done
# done

for region in ${regions[@]}
do
    command="bash preprocess_data.sbatch $option $region 00000"
    echo $command
    eval $command
done