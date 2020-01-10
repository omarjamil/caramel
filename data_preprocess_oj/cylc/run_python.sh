#!/bin/bash

#
# load the instance of the stack you want, in this case scitools-default.
# also show some info using the help option.
#
module load scitools
module help scitools

#
# need to update the PYTHONPATH to include yor project code?
#
# export PYTHONPATH=~me/python/my-project:${PYTHONPATH}

#
# display useful environment variables
#
echo "LD_LIBRARY_PATH = ${LD_LIBRARY_PATH}"
# echo "PYTHONPATH = ${PYTHONPATH}"
echo "SSS_ENV_DIR = ${SSS_ENV_DIR}"
echo "SSS_TAG_DIR = ${SSS_TAG_DIR}"

#
# call your python program
#
python /home/h06/ojamil/analysis_tools/ML/CRM/caramel/data_preprocess_oj/preprocess_data.py $1 $2
# python /home/h06/ojamil/analysis_tools/ML/CRM/caramel/data_preprocess_oj/cylc/foo.py
