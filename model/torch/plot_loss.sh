#!/bin/bash

filename=`grep tar $1`.loss
touch $filename
grep "Average loss" $1 | awk '{print $3, $6}' > temp.1 
grep "validation loss" $1 | awk '{print $4}' > temp.2
paste -d' ' temp.1 temp.2 > $filename
rm temp.1 temp.2
echo "plot \"$filename\" using 1:2, \"$filename\" using 1:3" > plot.p
gnuplot -persist plot.p