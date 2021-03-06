#!/bin/bash
regions=('0N100W' '0N130W' '0N15W' '0N160E' '0N160W' '0N30W' '0N50E' '0N70E' '0N88E' '10N100W' '10N120W' '10N140W' '10N145E' '10N160E' '10N170W' '10N30W' '10N50W' '10N60E' '10N88E' '10S120W' '10S140W' '10S15W' '10S170E' '10S170W' '10S30W' '10S5E' '10S60E' '10S88E' '10S90W' '20N135E' '20N145W' '20N170E' '20N170W' '20N30W' '20N55W' '20N65E' '20S0E' '20S100W' '20S105E' '20S130W' '20S160W' '20S30W' '20S55E' '20S80E' '21N115W' '29N65W' '30N130W' '30N145E' '30N150W' '30N170E' '30N170W' '30N25W' '30N45W' '30S100W' '30S10E' '30S130W' '30S15W' '30S160W' '30S40W' '30S60E' '30S88E' '40N140W' '40N150E' '40N160W' '40N170E' '40N25W' '40N45W' '40N65W' '40S0E' '40S100E' '40S100W' '40S130W' '40S160W' '40S50E' '40S50W' '50N140W' '50N149E' '50N160W' '50N170E' '50N25W' '50N45W' '50S150E' '50S150W' '50S30E' '50S30W' '50S88E' '50S90W' '60N15W' '60N35W' '60S0E' '60S140E' '60S140W' '60S70E' '60S70W' '70N0E' '70S160W' '70S40W' '80N150W')
streams=('a' 'b' 'c' 'd' 'e' 'f')
ana_time=('T0000Z' 'T1200Z')
# 
for r in ${regions[@]}
do
    # i=0
    location='/project/spice/radiation/ML/CRM/data/u-bs572/'$r'/'
    for d in {16..30}
    do
        for a in ${ana_time[@]}
        do
            for s in ${streams[@]}
            do
                for v in {000..011}
                do
                    fname=$location'201701'$d$a'_'$r'_km1p5_ra1m_pver'$s$v'.pp'
                    if [ ! -f "$fname" ]
                    then
                        echo Missing $fname
                    fi
                    # i=$(($i+1))
                done
            done
        done
    done
    echo $r files checked
done