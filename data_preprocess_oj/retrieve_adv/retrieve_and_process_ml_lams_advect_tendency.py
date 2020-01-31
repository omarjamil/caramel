#!/usr/bin/env python	
#
# Cyril Morcrette (2019), Met Office, UK
#
# Use "module load scitools/default-current" at the command line before running this.

# Import some modules
import subprocess
import numpy as np
import datetime
import argparse

# Import some of my own functions
# from cjm_functions import daterange
from cjm_functions import extract_fields_for_advective_tendencies
from cjm_functions import generate_filename_in
from cjm_functions import retrieve_a_file

# Define some functions
def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)
# End of functions

parser = argparse.ArgumentParser(description="Pass the date arguments")
parser.add_argument('-s', required=True, help="Start date YYYYMMDD")
parser.add_argument('-e', required=True, help="End date YYYYMMDD")
parser.add_argument('-r', required=True, help="Region string")
args = parser.parse_args()
sdate = args.s
edate = args.e
region = args.r

########################################################
# Next few lines are things you will need to change
########################################################
roseid='u-bj775'

# Probably worth ignoring the first 24 hours of the simulation
# so set start_date to 1 day after start of the actual runs.
start_date = datetime.datetime.strptime(sdate,"%Y%m%d")
end_date = datetime.datetime.strptime(edate,"%Y%m%d")
# start_date = date(2017, 1, 1)
# end_date   = date(2017, 1, 1)
print("Start date: {0} \n End date: {1}".format(sdate,edate))
# The name of all the domains needs to be added here. May be easiest to just type these out by hand. Or could try linking to a file stored somewhere with the full list.

regions=[region]
#By doing an ls within /home/d04/frme/cylc-run/u-bj775/share/data/ancils you get the list of all the domain names.
# regions=['0N126E','0N54E','10N120W','10N80W','10S40E','20N157E','20S112E','20S67E','30N153W','30S153E','40N30E','40S30E','50N144W','50S69W','60S135E','70N120W','80S90E','0N126W','0N54W','10N160E','10S0E','10S40W','20N157W','20S112W','20S67W','30N51E','30S153W','40N30W','40S30W','50N72E','50S72E','60S135W','70S0E','80S90W','0N162E','0N90E','10N160W','10S120E','10S80E','20N22E','20S157E','30N0E','30N51W','30S51E','40N90E','40S90E','50N72W','60N135E','60S39W','70S120E','0N162W','0N90W','10N40E','10S120W','10S80W','20N22W','20S157W','30N102E','30S0E','30S51W','40N90W','40S90W','50S0E','60N135W','60S45E','70S120W','0N18E','10N0E','10N40W','10S160E','20N112E','20N67E','20S22E','30N102W','30S102E','40N150E','40S150E','50N0E','50S144E','60N45E','70N0E','80N90E','0N18W','10N120E','10N80E','10S160W','20N118W','20N67W','20S22W','30N153E','30S102W','40N150W','40S150W','50N144E','50S144W','60N45W','70N120E','80N90W']
# regions=['0N18E']

###############################################################
# Below here are things you probably will not need to change.
###############################################################
# There is a new output file every hour.
# The timestamps on the files is T000 to T011.
# The simulation has a new analysis coming in every 12 hours
# So for every day there is a T0000Z and a T1200Z and for each there is T000 to T011.
# For each of those times there are 5 files containing:
#   a) lots of 2d fields (surface and radiative fluxes, screen temperature, MSLP etc)
#   b) 2d fields of precip, cloud, and total column moisture
#   c) 3d fields of theta, qv, qcl, qcf, cfl, cff, bcf, qrain, qgraupel
#   d) 3d field of advective increments to c).
#   e) 3d fields of u, v, and w
# Here we retrieve c and e in order to calculate advective tendencies.
list_analysis_time=['T0000Z','T1200Z']
name_str='ra1m'
list_stream=['c','e']

for single_date in daterange(start_date, end_date):
    for ana_time in np.arange(0,2,1):
    # for ana_time in np.arange(0,1,1):
        # Some of the runs start at 00Z, some at 12Z
        analysis_time=list_analysis_time[ana_time]
        for vt in np.arange(0,11+1,1):
        # for vt in np.arange(0,1,1):
            # vt=validitty time (i.e. T+?) range from T+0 to T+11 for sims that have analyses at 00Z and 12Z.
            for region_number in np.arange(0,len(regions),1):
                # Loop over the various LAMs (this could be all 98 of them).
                region=regions[region_number]
                # Retrieve both files (with all variables)
                # for stream_number in np.arange(0,len(list_stream),1):
                #     stream=list_stream[stream_number]
                #     outcome=retrieve_a_file(single_date,vt,roseid,name_str,analysis_time,stream,region,1)
                # Process both files at once
                # (i.e. extract theta, qv, qcl, qcf, qrain, qgraup from file 'c' and u, v, w from file 'e'
                # and write out to netcdf).
                outcome=extract_fields_for_advective_tendencies(single_date,vt,roseid,name_str,analysis_time,region)
                # Remove both pp files.
                # for stream_number in np.arange(0,len(list_stream),1):
                #     stream=list_stream[stream_number]
                #     outcome=retrieve_a_file(single_date,vt,roseid,name_str,analysis_time,stream,region,0)

     
          

