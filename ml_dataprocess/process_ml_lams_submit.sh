#!/bin/bash
#region="10N80W" # "80S90W"  "50S69W" "10S160W" "50N144E"
# regions=('50S69W' '80S90W' '10N80W' '10S160W' '50N144E' '10S40E' '50N144W' '10S120W' '10N120W' '20S112W' '0N90W' '30N153W' '80S90E' '40S90W' '10N80E' '0N90E' '10S80W' '50S0E' '70S0E' '0N162W' '30S102W' '40N90E' '70S120E' '60N135W' '70S120W' '50N72E' '40S30E' '10N40W' '20S22E' '10N40E' '30N105E' '20N157E' '40S150W' '30N102E' '40S30W' '30S153W' '0N54E' '50S144W' '20S67E' '60N45E'  '10S40W' '60S45E' '20S112E' '20N67W' '0N126W' '0N126E' '10N0E' '20S157W' '50N0E' '20N118W' '50S144E' '30N51W' '20N22W' '40S90E' '30S153E' '30N0E' '20N157W' '40N150W' '20N67E' '0N162E' '70N120E' '30N51E' '20S67W' '20N22E' '70N0E' '40S150E' '20S22W' '80N90E' '40N150E' '70N120W' '10N160W' '50N72W' '60S135W' '60S39W' '80N90W' '60S135E' '30S102E' '10S0E' '10N160E' '30N102W' '20N112E' '10S160E' '10S80E' '60N135E' '30S51E' '30N153E'  '10N120E' '0N18W' '30S51W' '10S120E' '40N30E' '40N90W' '0N54W' '30S0E' '60N45W' '20S157E' '50S72E' '0N18E' '40N30W')

# region="50S69W" 
# for i in {1..31}
# do
#     sbatch process_ml_lams_spice.sbatch $i $region --single
#     sbatch process_ml_lams_spice.sbatch $i $region --multi
#     sbatch process_ml_lams_spice.sbatch $i $region --advect
# done

# regions=('50N144W' '10S120W' '10N120W' '20S112W' '0N90W' '30N153W' '80S90E' '40S90W' '10N80E' '0N90E' '10S80W' '50S0E' '70S0E' '0N162W' '30S102W' '40N90E' '70S120E' '60N135W' '70S120W' '50N72E' '40S30E' '10N40W' '20S22E' '10N40E' '30N105E' '20N157E' '40S150W' '40S30W' '30S153W' '0N54E' '50S144W' '20S67E' '60N45E'  '10S40W' '60S45E' '20S112E' '20N67W' '0N126W' '0N126E' '10N0E' '20S157W' '50N0E' '20N118W' '50S144E' '30N51W' '20N22W' '40S90E' '30S153E' '30N0E' '20N157W' '40N150W' '20N67E' '0N162E' '70N120E' '30N51E' '20S67W' '20N22E' '70N0E' '40S150E' '20S22W' '80N90E' '40N150E' '70N120W' '10N160W' '50N72W' '60S135W' '60S39W' '80N90W' '60S135E' '30S102E' '10S0E' '10N160E' '30N102W' '20N112E' '10S160E' '10S80E' '60N135E' '30S51E' '30N153E'  '10N120E' '0N18W' '30S51W' '10S120E' '40N30E' '40N90W' '0N54W' '30S0E' '60N45W' '20S157E' '50S72E' '0N18E' '40N30W')

# for region in ${regions[@]}
# do
#     for i in 3 15 25
#     do
#         sbatch process_ml_lams_spice.sbatch $i $region --single
#         sbatch process_ml_lams_spice.sbatch $i $region --multi
#         sbatch process_ml_lams_spice.sbatch $i $region --advect
#     done         
# done


# Ocean only from u-bs572 and u-bs573 set aside for validation '0N100W'
# leave out 0N0E as that does not get read into iris properly
regions=('0N100W' '0N130W' '0N15W' '0N160E' '0N160W' '0N30W' '0N50E' '0N70E' '0N88E' '10N100W' '10N120W' '10N140W' '10N145E' '10N160E' '10N170W' '10N30W' '10N50W' '10N60E' '10N88E' '10S120W' '10S140W' '10S15W' '10S170E' '10S170W' '10S30W' '10S5E' '10S60E' '10S88E' '10S90W' '20N135E' '20N145W' '20N170E' '20N170W' '20N30W' '20N55W' '20N65E' '20S0E' '20S100W' '20S105E' '20S130W' '20S160W' '20S30W' '20S55E' '20S80E' '21N115W' '29N65W' '30N130W' '30N145E' '30N150W' '30N170E' '30N170W' '30N25W' '30N45W' '30S100W' '30S10E' '30S130W' '30S15W' '30S160W' '30S40W' '30S60E' '30S88E' '40N140W' '40N150E' '40N160W' '40N170E' '40N25W' '40N45W' '40N65W' '40S0E' '40S100E' '40S100W' '40S130W' '40S160W' '40S50E' '40S50W' '50N140W' '50N149E' '50N160W' '50N170E' '50N25W' '50N45W' '50S150E' '50S150W' '50S30E' '50S30W' '50S88E' '50S90W' '60N15W' '60N35W' '60S0E' '60S140E' '60S140W' '60S70E' '60S70W' '70N0E' '70S160W' '70S40W' '80N150W')
# regions=('0N100W')
# for region in ${regions[@]}
# do
#     for i in 9 14 18 23 30
#     do
#         sbatch process_ml_lams_spice.sbatch $i $region --single
#         sbatch process_ml_lams_spice.sbatch $i $region --multi
#         sbatch process_ml_lams_spice.sbatch $i $region --advect
#     done         
# done

# regions=('0N100W' '0N130W' '0N15W' '0N160W' '0N30W' '0N50E' '0N70E' '0N88E' '10N100W' '10N120W' '10N140W' '10N145E' '10N160E' '10N170W' '10N30W' '10N50W' '10N60E' '10N88E' '10S120W' '10S140W' '10S15W' '10S170E' '10S170W' '10S30W' '10S5E' '10S60E' '10S88E' '10S90W' '20N135E' '20N145W' '20N170E' '20N170W' '20N30W' '20N55W' '20N65E' '20S0E' '20S100W' '20S105E' '20S130W' '20S160W' '20S30W' '20S55E' '20S80E' '21N115W' '29N65W' '30N130W' '30N145E' '30N150W' '30N170E' '30N170W' '30N25W' '30N45W' '30S100W' '30S10E' '30S130W' '30S15W' '30S160W' '30S40W' '30S60E' '30S88E' '40N140W' '40N150E' '40N160W' '40N170E' '40N25W' '40N45W' '40N65W' '40S0E' '40S100E' '40S100W' '40S130W' '40S160W' '40S50E' '40S50W' '50N140W' '50N149E' '50N160W' '50N170E' '50N25W' '50N45W' '50S150E' '50S150W' '50S30E' '50S30W' '50S88E')

# poll submit to avoid job numbers limit
r=0
month=1
while [ $r -lt ${#regions[@]} ]
do
    region=${regions[$r]}
    sq=`squeue -u ojamil | wc -l`
    let qlength=$((sq-1))
    let gap=$((999-qlength))
    if (( gap > 45 ))
    then
        for d in {16..30} # 3 4 5 9 14 18 23 30
        do
            echo region $region day $d
            echo process_ml_lams_spice.sbatch $d $month $region --single
            echo process_ml_lams_spice.sbatch $d $month $region --multi
            echo process_ml_lams_spice.sbatch $d $month $region --advect
            sbatch process_ml_lams_spice.sbatch $d $month $region --single
            sbatch process_ml_lams_spice.sbatch $d $month $region --multi
            sbatch process_ml_lams_spice.sbatch $d $month $region --advect
        done
        r=$((r+1))
    fi
    sleep 2.5
done