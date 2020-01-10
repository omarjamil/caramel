
import subprocess
import numpy as np
from datetime import date
import sys
from pathlib import Path
from cjm_functions import daterange

# Define some functions
def check_files_exist(regions: list, start_date: date, end_date: date, list_analysis_time: list, name_str: str, list_stream: list):
    """
    Check all the files that will be used in combine_files_per_subdomain
    actually exist.
    """
    missing_dates=[]
    # Create a list of files to read and then subsequently combine
    for single_date in daterange(start_date, end_date):
#    for ana_time in np.arange(0,len(list_analysis_time),1):
        for ana_time in np.arange(0,2,1):
        # Some of the runs start at 00Z, some at 12Z
            analysis_time=list_analysis_time[ana_time]
            for vt in np.arange(0,11+1,1):
            # vt=validitty time (i.e. T+?) range from T+0 to T+11 for sims that have analyses at 00Z and 12Z.
                for region_number in np.arange(0,len(regions),1):
                    # Loop over the various LAMs (this could be all 98 of them).
                    region=regions[region_number]
                    location='/net/spice/project/radiation/ML/CRM/data/u-bj775/{0}/'.format(region)
                    for stream_number in np.arange(0,len(list_stream),1):
                        stream=list_stream[stream_number]
                        infile=location+single_date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_pver'+stream+str(vt).zfill(3)+'.pp'
                        fpath = Path(infile)
                        if not fpath.is_file():
                            print("Region {0}, date {1}, script: {0}_{1}_retrieve.sh".format(region,single_date.strftime("%Y%m%d")))
                            # print("Does not exits: {0}".format(infile))

def run_missing_file_scripts(region: str, start_date: date):
    """
    Run the retrieval script created retreive_ml_lams_submission.py
    """
    start_date_str=start_date.strftime("%Y%m%d")
    fname='/scratch/ojamil/'+region+'_'+start_date_str+"_retrieve.sh"
    print(fname)  
    subprocess.call(fname,shell=False)
    

def main_check_files():
    roseid='u-bj775'
    #roseid='u-bj967'

    # Probably worth ignoring the first 24 hours of the simulation
    # so set start_date to 1 day after start of the actual runs.
    start_day = int(sys.argv[1])
    start_month = 1
    end_day = int(sys.argv[2])
    end_month = start_month
    start_date = date(2017, start_month, start_day)
    end_date   = date(2017, end_month, end_day)
    region=sys.argv[3]
    # The name of all the domains needs to be added here. May be easiest to just type these out by hand. Or could try linking to a file stored somewhere with the full list.
    #regions=['80N90W','0N126E','0N54E']

    #regions=['50S72E']
    regions=[region]
    #By doing an ls within /home/d04/frme/cylc-run/u-bj775/share/data/ancils you get the list of all the domain names.
    #regions=['0N126E','0N54E','10N120W','10N80W','10S40E','20N157E','20S112E','20S67E','30N153W','30S153E','40N30E','40S30E','50N144W','50S69W','60S135E','70N120W','80S90E','0N126W','0N54W','10N160E','10S0E','10S40W','20N157W','20S112W','20S67W','30N51E','30S153W','40N30W','40S30W','50N72E','50S72E','60S135W','70S0E','80S90W','0N162E','0N90E','10N160W','10S120E','10S80E','20N22E','20S157E','30N0E','30N51W','30S51E','40N90E','40S90E','50N72W','60N135E','60S39W','70S120E','0N162W','0N90W','10N40E','10S120W','10S80W','20N22W','20S157W','30N102E','30S0E','30S51W','40N90W','40S90W','50S0E','60N135W','60S45E','70S120W','0N18E','10N0E','10N40W','10S160E','20N112E','20N67E','20S22E','30N102W','30S102E','40N150E','40S150E','50N0E','50S144E','60N45E','70N0E','80N90E','0N18W','10N120E','10N80E','10S160W','20N118W','20N67W','20S22W','30N153E','30S102W','40N150W','40S150W','50N144E','50S144W','60N45W','70N120E','80N90W']

    # Code here is written assuming dealing with 4d data, variable being (nt,nz,nx,ny).
    # List of stash codes to process (Make sure you add any new diagnostics that need processing to all 3 of these variables.
    list_stash_sec=[16,12,0,12]
    list_stash_code=[4,181,10,182]
    list_stream=['c','d','c','d']

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
   
    check_files_exist(regions, start_date, end_date, list_analysis_time, name_str, list_stream)

def main_missing_file_scripts():
    
    start_day = int(sys.argv[2])
    start_month = 1
    run_region=sys.argv[1]
    start_date = date(2017, start_month, start_day)

    run_missing_file_scripts(run_region, start_date)
    
if __name__ == "__main__": 
    main_check_files()
    #main_missing_file_scripts()
