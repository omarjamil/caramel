#!/bin/bash

errorFiles=$(grep -i error *.out | awk -F ":" '{print $1}')


for f in ${errorFiles[@]}
do
    head -n 3 $f
done

echo Regions with error
regions=()
for f in ${errorFiles[@]}
do
    regions+=$(tail -n 1 $f | awk -F "/" '{print $9}')
    regions+=" "
done

IFS=' ' # space is set as delimiter
read -ra tempArr <<< "${regions[@]}" # str is read into an array as tokens separated by IFS

declare -A regions_uniq
for i in "${tempArr[@]}"; do regions_uniq["$i"]=1; done
printf '%s\n' "${!regions_uniq[@]}"

