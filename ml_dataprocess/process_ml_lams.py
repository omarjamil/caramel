import warnings
warnings.filterwarnings("ignore")
import subprocess
import numpy as np
from datetime import date, timedelta
import iris
import iris.analysis
import sys
import argparse

# Import some of my own functions
from cjm_functions import process_ml_lam_file
from cjm_functions import extract_fields_for_advective_tendencies
from cjm_functions import generate_filename_in
from cjm_functions import retrieve_a_file

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)   

def process_files(regions: list, start_date: date, end_date: date, list_analysis_time: list, name_str: str, list_stream: list,  list_stash_sec: list, list_stash_code: list, roseid: str, ndims: int):
    for single_date in daterange(start_date, end_date):
        #    for ana_time in np.arange(0,len(list_analysis_time),1):
        for ana_time in np.arange(2):
            # Some of the runs start at 00Z, some at 12Z
            analysis_time=list_analysis_time[ana_time]
            for vt in np.arange(12):
                # vt=validitty time (i.e. T+?) range from T+0 to T+11 for sims that have analyses at 00Z and 12Z.
                for region_number in np.arange(0,len(regions),1):
                    # Loop over the various LAMs (this could be all 98 of them).
                    region=regions[region_number]
                    for stream_number in np.arange(0,len(list_stream),1):
                        stream=list_stream[stream_number]
                        stash_sec=list_stash_sec[stream_number]
                        stash_code=list_stash_code[stream_number]
                        if ndims == 3:
                            outcome=process_ml_lam_file(single_date,vt,roseid,name_str,analysis_time,stream,region,stash_sec,stash_code,3)
                        elif ndims == 2:
                            outcome=process_ml_lam_file(single_date,vt,roseid,name_str,analysis_time,stream,region,stash_sec,stash_code,2)
########################################################
# Next few lines are things you will need to change
########################################################


###############################################################
# Below here are things you probably will not need to change.
###############################################################
# There is a new output file every hour.
# The timestamps on the files is T000 to T011.
# The simulation has a new analysis coming in every 12 hours
# So for every day there is a T0000Z and a T1200Z and for each there is T000 to T011.
# For each of those times there are 4 files containing:
#   a) lots of 2d fields (surface and radiative fluxes, screen temperature, MSLP etc)
#   b) 2d fields of precip, cloud, and total column moisture
#   c) 3d fields of theta, qv, qcl, qcf, cfl, cff, bcf, qrain, qgraupel
#   d) 3d field of increments to c).


def single_level(start_day: int, region: str):
    # roseid='u-bj775'
    roseid='u-br800'
       
    #roseid='u-bj967'

    # Probably worth ignoring the first 24 hours of the simulation
    # so set start_date to 1 day after start of the actual runs.
    # start_day = int(sys.argv[1])
    start_month = 1
    # end_day = start_day + 1 #int(sys.argv[2])
    # end_month = start_month
    start_date = date(2017, start_month, start_day)
    end_date   = start_date+timedelta(days=1) #date(2017, end_month, end_day)

    # The name of all the domains needs to be added here. May be easiest to just type these out by hand. Or could try linking to a file stored somewhere with the full list.
    #regions=['80N90W','0N126E','0N54E']
    # region=sys.argv[2]
    #regions=['50S72E']
    regions=[region]
    #By doing an ls within /home/d04/frme/cylc-run/u-bj775/share/data/ancils you get the list of all the domain names.
    #regions=['0N126E','0N54E','10N120W','10N80W','10S40E','20N157E','20S112E','20S67E','30N153W','30S153E','40N30E','40S30E','50N144W','50S69W','60S135E','70N120W','80S90E','0N126W','0N54W','10N160E','10S0E','10S40W','20N157W','20S112W','20S67W','30N51E','30S153W','40N30W','40S30W','50N72E','50S72E','60S135W','70S0E','80S90W','0N162E','0N90E','10N160W','10S120E','10S80E','20N22E','20S157E','30N0E','30N51W','30S51E','40N90E','40S90E','50N72W','60N135E','60S39W','70S120E','0N162W','0N90W','10N40E','10S120W','10S80W','20N22W','20S157W','30N102E','30S0E','30S51W','40N90W','40S90W','50S0E','60N135W','60S45E','70S120W','0N18E','10N0E','10N40W','10S160E','20N112E','20N67E','20S22E','30N102W','30S102E','40N150E','40S150E','50N0E','50S144E','60N45E','70N0E','80N90E','0N18W','10N120E','10N80E','10S160W','20N118W','20N67W','20S22W','30N153E','30S102W','40N150W','40S150W','50N144E','50S144W','60N45W','70N120E','80N90W']

    # Code here is written assuming dealing with 3d data, variable being (nt,nx,ny).
    # List of stash codes to process (Make sure you add any new diagnostics that need processing to all 3 of these variables.
    list_stash_sec=[0,1,1,1,1,1,2,2,2,3,3,3,3,3,3,4,4,9,30,30,30,16]
    list_stash_code=[24,202,205,207,208,235,201,207,205,217,225,226,234,236,245,203,204,217,405,406,461,222]
    list_stream=['a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','b','b','b','b','b','b','a']
    
    # list_stash_sec=[2]
    # list_stash_code=[207]
    # list_stream=['a']
    
    list_analysis_time=['T0000Z','T1200Z']
    name_str='ra1m'
    
    ndims = 2
    process_files(regions, start_date, end_date, list_analysis_time, name_str, list_stream,  list_stash_sec, list_stash_code, roseid, ndims)

def multi_level(start_day: int, region: str):
    # roseid='u-bj775'
    roseid='u-br800'
    #roseid='u-bj967'

    # Probably worth ignoring the first 24 hours of the simulation
    # so set start_date to 1 day after start of the actual runs.
    # start_day = int(sys.argv[1])
    start_month = 1
    # end_day = start_day + 1 #int(sys.argv[2])
    # end_month = start_month
    start_date = date(2017, start_month, start_day)
    end_date   = start_date+timedelta(days=1) #date(2017, end_month, end_day)

    # The name of all the domains needs to be added here. May be easiest to just type these out by hand. Or could try linking to a file stored somewhere with the full list.
    #regions=['80N90W','0N126E','0N54E']
    # region=sys.argv[2]
    #regions=['50S72E']
    regions=[region]
    #By doing an ls within /home/d04/frme/cylc-run/u-bj775/share/data/ancils you get the list of all the domain names.
    #regions=['0N126E','0N54E','10N120W','10N80W','10S40E','20N157E','20S112E','20S67E','30N153W','30S153E','40N30E','40S30E','50N144W','50S69W','60S135E','70N120W','80S90E','0N126W','0N54W','10N160E','10S0E','10S40W','20N157W','20S112W','20S67W','30N51E','30S153W','40N30W','40S30W','50N72E','50S72E','60S135W','70S0E','80S90W','0N162E','0N90E','10N160W','10S120E','10S80E','20N22E','20S157E','30N0E','30N51W','30S51E','40N90E','40S90E','50N72W','60N135E','60S39W','70S120E','0N162W','0N90W','10N40E','10S120W','10S80W','20N22W','20S157W','30N102E','30S0E','30S51W','40N90W','40S90W','50S0E','60N135W','60S45E','70S120W','0N18E','10N0E','10N40W','10S160E','20N112E','20N67E','20S22E','30N102W','30S102E','40N150E','40S150E','50N0E','50S144E','60N45E','70N0E','80N90E','0N18W','10N120E','10N80E','10S160W','20N118W','20N67W','20S22W','30N153E','30S102W','40N150W','40S150W','50N144E','50S144W','60N45W','70N120E','80N90W']

    # Code here is written assuming dealing with 4d data, variable being (nt,nz,nx,ny).
    # List of stash codes to process (Make sure you add any new diagnostics that need processing to all 3 of these variables.
    # list_stash_sec=[16,12,0,12]
    # list_stash_code=[4,181,10,182]
    # list_stream=['c','d','c','d']
    # Land and Ocean
    # list_stash_sec=[0,16,12,0,12,0,12,0,12,0,12,0,12]
    # list_stash_code=[4,4,181,10,182,254,183,12,184,272,189,273,190]
    # list_stream=['c','b','d','c','d','c','d','c','d','c','d','c','d']
    # Ocean only runs
    list_stash_sec = [0, 16, 0, 0, 0, 0, 0]
    list_stash_code = [4, 4, 10, 254, 12, 272, 273]
    list_stream = ['c', 'd', 'c', 'c', 'c', 'c', 'c']
    ###############################################################
    # Below here are things you probably will not need to change.
    ###############################################################
    # There is a new output file every hour.
    # The timestamps on the files is T000 to T011.
    # The simulation has a new analysis coming in every 12 hours
    # So for every day there is a T0000Z and a T1200Z and for each there is T000 to T011.
    # For each of those times there are 4 files containing:
    #   a) lots of 2d fields (surface and radiative fluxes, screen temperature, MSLP etc)
    #   b) 2d fields of precip, cloud, and total column moisture
    #   c) 3d fields of theta, qv, qcl, qcf, cfl, cff, bcf, qrain, qgraupel
    #   d) 3d field of increments to c).

    list_analysis_time=['T0000Z','T1200Z']
    name_str='ra1m'
    ndims = 3 
    process_files(regions, start_date, end_date, list_analysis_time, name_str, list_stream,  list_stash_sec, list_stash_code, roseid, ndims)

def advect_process(start_day, region):

    # roseid='u-bj775'
    roseid='u-br800'
    #roseid='u-bj967'

    # Probably worth ignoring the first 24 hours of the simulation
    # so set start_date to 1 day after start of the actual runs.
    # start_day = int(sys.argv[1])
    start_month = 1
    # end_day = start_day + 1 #int(sys.argv[2])
    # end_month = start_month
    start_date = date(2017, start_month, start_day)
    end_date   = start_date+timedelta(days=1) #date(2017, end_month, end_day)

    list_analysis_time=['T0000Z','T1200Z']
    name_str='ra1m'
    list_stream=['c','e']
    regions=[region]
    
    for single_date in daterange(start_date, end_date):
        for ana_time in np.arange(2):
        # for ana_time in np.arange(0,1,1):
            # Some of the runs start at 00Z, some at 12Z
            analysis_time=list_analysis_time[ana_time]
            for vt in np.arange(12):
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process LAMS')
    parser.add_argument('--single', action='store_true', default=False,
                    help='single level files processing')
    parser.add_argument('--multi', action='store_true', default=False,
                    help='multi level files processing')
    parser.add_argument('--advect', action='store_true', default=False,
                    help='advection files processing')                                
    parser.add_argument('--start-day', type=int, default=1, metavar='N',
                    help='date start day')
    parser.add_argument('--region', type=str, help='region to process e.g. 50S69W')
    args = parser.parse_args()
    
    single = args.single
    multi = args.multi
    advect = args.advect
    start_day = args.start_day
    region = args.region

    if single:
        single_level(start_day, region)
    elif multi:
        multi_level(start_day, region)
    elif advect:
        advect_process(start_day, region)