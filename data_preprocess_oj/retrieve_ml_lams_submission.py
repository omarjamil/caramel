import datetime
import subprocess
import sys
import os
import pandas as pd

all_regions=['10S160W','50S69W','50N144E','10S40E','50N144W','10S120W','10N120W','20S112W','0N90W','30N153W','80S90E','40S90W','10N80E','0N90E','10S80W','50S0E','70S0E','0N162W','30S102W','40N90E','70S120E','60N135W','70S120W','50N72E','40S30E','10N40W','20S22E','10N40E','30N105E','20N157E','40S150W','30N102E','40S30W','30S153W','0N54E','50S144W','20S67E','60N45E','80S90W','10S40W','60S45E','20S112E','20N67W','0N126W','0N126E','10N0E','20S157W','50N0E','20N118W','50S144E','30N51W','20N22W','40S90E','30S153E','30N0E','20N157W','40N150W','20N67E','0N162E','70N120E','30N51E','20S67W','20N22E','70N0E','40S150E','20S22W','80N90E','40N150E','70N120W','10N160W','50N72W','60S135W','60S39W','80N90W','60S135E','30S102E','10S0E','10N160E','30N102W','20N112E','10S160E','10S80E','60N135E','30S51E','30N153E','10N80W','10N120E','0N18W','30S51W','10S120E','40N30E','40N90W','0N54W','30S0E','60N45W','20S157E','50S72E','0N18E','40N30W']

regions_1=all_regions[:30]
regions_2=all_regions[30:60]
regions_3=all_regions[60:90]
regions_4=all_regions[90:]

def command(region, start_date, end_date):
    #script_call="{0} -r {1} -s {2} -e {3}".format("/home/h06/ojamil/analysis_tools/ML/CRM/CRM_code/data_preprocess/retrieve_ml_lams.py",region,start_date,end_date)
    #command_str = ['/bin/bash','-l','sbatch echo \"{0} --mem=1000 --ntasks=2 --time=1 --export=NONE --qos=normal\"'.format(script_call)]
    script_options = ['-r {0} -s {1} -e {2}'.format(region,start_date,end_date)]
    command_str=['sbatch --mem=1000 --ntasks=2 --time=3000 --export=NONE --qos=long','/home/h06/ojamil/analysis_tools/ML/CRM/CRM_code/data_preprocess/retrieve_ml_lams.py']
   
    return command_str+script_options

def main():
    begin = datetime.datetime(2017,1,1)
    end = datetime.datetime(2017,1,31)
    dates = (pd.date_range(begin,end)).to_pydatetime()
    # print(dates.to_pydatetime())
    for region in regions_1:
        for d in dates:
            start_date_str = datetime.datetime.strftime(d,"%Y%m%d")
            end_date_str = datetime.datetime.strftime(d+datetime.timedelta(days=1),"%Y%m%d")
            call_str = ' '.join(command(region,start_date_str,end_date_str))
            fname='/scratch/ojamil/'+region+'_'+start_date_str+"_retrieve.sh"
            print(fname)
            with open(fname,'w') as outfile:
                outfile.write("#!/bin/bash -l\n")
                outfile.write("echo $0 \n")
                outfile.write(call_str)
                outfile.write("\n")
                os.chmod(fname,0o775)
            subprocess.call(fname, shell=False)

if __name__ == "__main__":
    main()
