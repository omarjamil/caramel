#!/usr/bin/env

import datetime
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import gc
import iris
import tendencies

def combine_q_tednencies(region):
    """
    combine qcl, qv, qcf, qg, qrain tendencies into a single one
    """
    # qv, qcl, qcf, rain, graupel
    # q_stashes = [10,254,12,272,273]
    # q_adv_stashes = [12182,12183,12184,12189,12190]
    q_stashes = [254,12,272,273]
    q_adv_stashes = [12183,12184,12189,12190]
    q_cubelist = []
    phys_cubelist = []
    adv_cubelist = []
    # Read in the first q stash and then append to that data array in the second loop
    for subdomain in range(64):
        q = "/project/spice/radiation/ML/CRM/data/u-bj775/{2}/concat_stash_{1}/30_days_50S69W_km1p5_ra1m_30x30_subdomain_{0}_{1}.nc".format(str(subdomain).zfill(3), str(10).zfill(5), region)
        q_phys = "/project/spice/radiation/ML/CRM/data/u-bj775/{2}/tend_q_{1}/30_days_50S69W_km1p5_ra1m_30x30_subdomain_{0}_q_phys_{1}.nc".format(str(subdomain).zfill(3), str(10).zfill(5), region)
        q_adv = "/project/spice/radiation/ML/CRM/data/u-bj775/{2}/concat_stash_{1}/30_days_50S69W_km1p5_ra1m_30x30_subdomain_{0}_{1}.nc".format(str(subdomain).zfill(3), str(12182).zfill(5), region)
        phys_cubelist.append(iris.load_cube(q_phys))
        adv_cubelist.append(iris.load_cube(q_adv))
        q_cubelist.append(iris.load_cube(q))
    
    for q,qadv in zip(q_stashes,q_adv_stashes):
        print("Processing STASHES {0} {1}".format(q,qadv))
        for subdomain in range(64):
            q_file_name = "30_days_50S69W_km1p5_ra1m_30x30_subdomain_{0}_{1}.nc".format(str(subdomain).zfill(3), str(q).zfill(5))
            q_phys_file_name = "30_days_50S69W_km1p5_ra1m_30x30_subdomain_{0}_q_phys_{1}.nc".format(str(subdomain).zfill(3), str(q).zfill(5))
            q_adv_file_name = "30_days_50S69W_km1p5_ra1m_30x30_subdomain_{0}_{1}.nc".format(str(subdomain).zfill(3), str(qadv).zfill(5))

            q_file = "/project/spice/radiation/ML/CRM/data/u-bj775/{2}/concat_stash_{0}/{1}".format(str(q).zfill(5),q_file_name, region)
            q_phys_file = "/project/spice/radiation/ML/CRM/data/u-bj775/{2}/tend_q_{0}/{1}".format(str(q).zfill(5),q_phys_file_name, region)
            q_adv_file = "/project/spice/radiation/ML/CRM/data/u-bj775/{2}/concat_stash_{0}/{1}".format(str(qadv).zfill(5),q_adv_file_name, region)

            q_cube = iris.load_cube(q_file)
            qphys_cube = iris.load_cube(q_phys_file)
            qadv_cube = iris.load_cube(q_adv_file)
            (phys_cubelist[subdomain]).data =  phys_cubelist[subdomain].data + qphys_cube.data
            (adv_cubelist[subdomain]).data = adv_cubelist[subdomain].data + qadv_cube.data
            (q_cubelist[subdomain]).data = q_cubelist[subdomain].data + q_cube.data
            
    i = 0
    for q,p,a in zip(q_cubelist, phys_cubelist, adv_cubelist):
        q.var_name = "q_tot"
        q.long_name = "combined_q_quantities"
        p.var_name = "qphys_tot"
        p.long_name = "combined_q_quantities_phys_tendencies"
        a.long_name = "combine_q_quantities_advection"
        a.var_name = "qadv_tot"
        q.attributes['STASH'] = ''
        p.attributes['STASH'] = ''
        a.attributes['STASH'] = ''
        q.attributes['STASHES'] = '00010,00254,00012,00272,00273'
        p.attributes['STASHES'] = '00010,00254,00012,00272,00273'
        a.attributes['STASHES'] = '12182,12183,12184,12189,12190'
        print("Saving q total files for region {0} subdomain {1}".format(region, i))
        iris.fileformats.netcdf.save(p,"/project/spice/radiation/ML/CRM/data/u-bj775/{1}/tend_qphys_tot/30_days_50S69W_km1p5_ra1m_30x30_subdomain_{0}_qphys_tot.nc".format(str(i).zfill(3), region))
        iris.fileformats.netcdf.save(a,"/project/spice/radiation/ML/CRM/data/u-bj775/{1}/tend_qadv_tot/30_days_50S69W_km1p5_ra1m_30x30_subdomain_{0}_qadv_tot.nc".format(str(i).zfill(3), region))
        iris.fileformats.netcdf.save(q,"/project/spice/radiation/ML/CRM/data/u-bj775/{1}/q_tot/30_days_50S69W_km1p5_ra1m_30x30_subdomain_{0}_q_tot.nc".format(str(i).zfill(3), region))
        i += 1

def combine_q(region, in_prefix="30"):
    """
    combine qcl, qv, qcf, qg, qrain into a single one
    """
    # qv, qcl, qcf, rain, graupel
    # q_stashes = [10,254,12,272,273]
    q_stashes = [254,12,272,273]
    
    q_cubelist = []
    # Read in the first q stash 00010 and then append to that data array in the second loop
    for subdomain in range(64):
        q = "/project/spice/radiation/ML/CRM/data/u-bj775_/{2}/concat_stash_{1}/{3}_days_{2}_km1p5_ra1m_30x30_subdomain_{0}_{1}.nc".format(str(subdomain).zfill(3), str(10).zfill(5), region, in_prefix)
        q_cubelist.append(iris.load_cube(q))
    
    for q in q_stashes:
        print("Processing STASHES {0}".format(q))
        for subdomain in range(64):
            q_file_name = "{3}_days_{2}_km1p5_ra1m_30x30_subdomain_{0}_{1}.nc".format(str(subdomain).zfill(3), str(q).zfill(5),region,in_prefix)
            q_file = "/project/spice/radiation/ML/CRM/data/u-bj775_/{2}/concat_stash_{0}/{1}".format(str(q).zfill(5),q_file_name, region)
            q_cube = iris.load_cube(q_file)
            (q_cubelist[subdomain]).data = q_cubelist[subdomain].data + q_cube.data

    save_path="/project/spice/radiation/ML/CRM/data/u-bj775_/{0}/concat_stash_99821/".format(region)         
    try:
        os.makedirs(save_path)
    except OSError:
        pass
    i = 0
    for q in q_cubelist:
        q.var_name = "q_tot"
        q.long_name = "combined_q_quantities"
        q.attributes={}
        # q.attributes['STASH'] = ''
        # q.attributes['um_stash_source'] = ''
        q.attributes['STASHES'] = '00010,00254,00012,00272,00273'
        print("Saving q total files for region {0} subdomain {1}".format(region, i))
        iris.fileformats.netcdf.save(q,"{0}/{3}_days_{2}_km1p5_ra1m_30x30_subdomain_{1}_99821.nc".format(save_path, str(i).zfill(3),region,in_prefix))
        i += 1

def check_files_exist(region: str, date: datetime, subdomain: int, stash: int):
    """
    Check all the files that will be used in combine_files_per_subdomain
    actually exist.
    """
    # region='10N160E'
    analysis_time=['0000','1200']
    list_of_files = []
    datestr=date.strftime("%Y%m%d")
    location='/net/spice/project/radiation/ML/CRM/data/u-bj775_/{0}/stash_{1}/'.format(region, str(stash).zfill(5))
    # location="/project/spice/radiation/ML/CRM/data/u-bj775/{0}/adv/netcdf/advect_incr/".format(region)
    # Create a list of files to read and then subsequently combine
    for atime in analysis_time:
        for vtime in range(12):
            in_filename="{0}T{1}Z_{2}_km1p5_ra1m_30x30_subdomain_{3}_vt_{4}_{5}.nc".format(datestr,atime,region,str(subdomain).zfill(3),str(vtime).zfill(3),str(stash).zfill(5))
            # in_filename="{0}T{1}Z_{2}_km1p5_ra1m_30x30_subdomain_{3}_{4}_{5}.nc".format(datestr,atime,region,str(subdomain).zfill(3),str(vtime).zfill(3),str(stash).zfill(5))
            list_of_files.append(in_filename)
    for f in list_of_files:
        # print("Checking {0}".format(f))
        fpath = Path(location+f)
        if not fpath.is_file():
            print("Does not exits: {0}".format(location+f))
            
def combine_files_per_subdomain(region: str, date: datetime, subdomain: int, stash: int):
    """
    Combine files per subdomain and per day for a given
    stash code
    The files being combined are split per analysis time
    and validity time
    """
    # region='10N160E'
    analysis_time=['0000','1200']
    list_of_files = []
    datestr=date.strftime("%Y%m%d")
    location='/net/spice/project/radiation/ML/CRM/data/u-bj775_/{0}/stash_{1}/'.format(region, str(stash).zfill(5))
    out_location='/project/spice/radiation/ML/CRM/data/u-bj775_/{0}/concat_stash_{1}/'.format(region, str(stash).zfill(5))
    # if not os.path.exists(out_location): This can create race condition
    try:
        os.makedirs(out_location)
    except OSError:
        pass
    # Create a list of files to read and then subsequently combine
    for atime in analysis_time:
        for vtime in range(12):
            in_filename="{0}T{1}Z_{2}_km1p5_ra1m_30x30_subdomain_{3}_vt_{4}_{5}.nc".format(datestr,atime,region,str(subdomain).zfill(3),str(vtime).zfill(3),str(stash).zfill(5))
            # in_filename="{0}T{1}Z_{2}_km1p5_ra1m_30x30_subdomain_{3}_{4}_{5}.nc".format(datestr,atime,region,str(subdomain).zfill(3),str(vtime).zfill(3),str(stash).zfill(5))

            list_of_files.append(in_filename)
    
    out_filename=out_location+"{0}_{1}_km1p5_ra1m_30x30_subdomain_{2}_{3}.nc".format(datestr,region,str(subdomain).zfill(3),str(stash).zfill(5))

    # temporaray list to create a cubelist
    clist = []
    for f in list_of_files:
        filename = location+f
        print("Reading {0}".format(filename))
        cube = iris.load_cube(filename)            
        clist.append(cube)
    cubelist = iris.cube.CubeList(clist)
    print("Saving file {0}".format(out_filename))
    iris.fileformats.netcdf.save(cubelist.concatenate()[0],out_filename)

def combine_day_tseries(start_date: datetime, end_date: datetime, region: str, subdomain: int, stash: int):
    """
    Combine the per day files into a single file 
    """
    # First create a list of all the files that are going to be
    # combined into a single timeseries
    tdelta = datetime.timedelta(days=1)
    filelist = []
    running_date = start_date
    location='/project/spice/radiation/ML/CRM/data/u-bj775_/{0}/concat_stash_{1}/'.format(region, str(stash).zfill(5))
    while running_date <= end_date:
        d = running_date.strftime('%Y%m%d')
        running_date += tdelta
        fname = "{0}_{1}_km1p5_ra1m_30x30_subdomain_{2}_{3}.nc".format(d,region,str(subdomain).zfill(3),str(stash).zfill(5))
        filelist.append(location+fname)

    # Now check the files exist
    print("Checking files exist")
    for f in filelist:
        fpath = Path(f)
        if not fpath.is_file():
            raise ValueError("File {0} does not exist".format(f))
        
    # If nothing went wrong above, let's combine the files
    out_filename =  "{0}_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_{3}.nc".format(len(filelist),region,str(subdomain).zfill(3),str(stash).zfill(5))
    out_location='/project/spice/radiation/ML/CRM/data/u-bj775_/{0}/concat_stash_{1}/'.format(region, str(stash).zfill(5))
    clist = []
    for filename in filelist:
        print("Reading {0}".format(filename))
        cube = iris.load_cube(filename)            
        clist.append(cube)
    cubelist = iris.cube.CubeList(clist)
    print("Saving file {0}".format(out_filename))
    iris.fileformats.netcdf.save(cubelist.concatenate()[0],out_location+out_filename)

def combine_day_tseries_dayrange(region: str, subdomain: int, stash: int, days_range=range(1,32)):
    """
    Combine the per day files into a single file 
    """
    # First create a list of all the files that are going to be
    # combined into a single timeseries
    filelist = []
    location='/project/spice/radiation/ML/CRM/data/u-bj775_/{0}/concat_stash_{1}/'.format(region, str(stash).zfill(5))

    for day in days_range:
        date = datetime.date(2017, 1, day)
        d = date.strftime('%Y%m%d')
        fname = "{0}_{1}_km1p5_ra1m_30x30_subdomain_{2}_{3}.nc".format(d,region,str(subdomain).zfill(3),str(stash).zfill(5))
        filelist.append(location+fname)

    # Now check the files exist
    print("Checking files exist")
    for f in filelist:
        fpath = Path(f)
        if not fpath.is_file():
            raise ValueError("File {0} does not exist".format(f))
        
    # If nothing went wrong above, let's combine the files
    out_filename =  "{0}_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_{3}.nc".format("031525",region,str(subdomain).zfill(3),str(stash).zfill(5))
    out_location='/project/spice/radiation/ML/CRM/data/u-bj775_/{0}/concat_stash_{1}/'.format(region, str(stash).zfill(5))
    clist = []
    for filename in filelist:
        print("Reading {0}".format(filename))
        cube = iris.load_cube(filename)            
        clist.append(cube)
    cubelist = iris.cube.CubeList(clist)
    print("Saving file {0}".format(out_filename))
    iris.fileformats.netcdf.save(cubelist.concatenate()[0],out_location+out_filename)
    
def combine_files(region: str, day: int, stashes: list):
    date = datetime.date(2017, 1, day)
    #for stash in [10,12182,16004,12181]:
    for stash in stashes:    
        for subdomain in range(64):
            combine_files_per_subdomain(region, date, subdomain, stash)
            

def main_check_files_exist(region: str, stashes: list):
    for day in range(1,32):
        date = datetime.date(2017, 1, day)
        # for stash in [10,12182,16004,12181]:
        for stash in stashes:
            for subdomain in range(64):
                check_files_exist(region, date, subdomain, stash)
            
def main_combine_files(region: str, stashes: list, days_range=range(1,32)):
    # day = sys.argv[1]
    
    for day in days_range:
        combine_files(region, day, stashes)

    #day=9
    #print(day)
    #main_combine_files(day)
        
def main_combine_day_tseries(region: str, stashes: list):
    # region='10N160E'
    # Start from day 2 to ignore spin up day 1
    start_date = datetime.datetime(2017,1,2)
    end_date = datetime.datetime(2017,1,31)
    # for stash in [10,12182,16004,12181]:
    for stash in stashes:
        for subdomain in range(64):
            # combine_day_tseries(start_date, end_date, region, subdomain, stash)
            combine_day_tseries_dayrange(region, subdomain, stash, days_range=[3,15,25])


def calc_tendencies(region: str, in_prefix: str="30"):
    tendencies.main_Q_dot(region, in_prefix=in_prefix)
    tendencies.main_T_dot(region, in_prefix=in_prefix)            
    

        
if __name__ == "__main__":
    argument = sys.argv[1]
    region = sys.argv[2]
    stash = sys.argv[3]

    # stashes = [16004, 12181, 10, 12182, 254,12183,12,12184,272,12189,273,12190] #T, qv, qcl, qcf, qg
    # stashes = [4,24,1202,1205,1207,1208,1235,2201,2207,2205,3217,3225,3226,3234,3236,3245,4203,4204,9217,30405,30406,30461,16222,99181,99182]
    stashes = [int(stash)]

    if argument == '1':
        main_check_files_exist(region, stashes)
    elif argument == '2':
        main_combine_files(region, stashes, days_range=[3,15,25])
    elif argument == '3':
        main_combine_day_tseries(region, stashes)
    elif argument == '4':
        combine_q(region, in_prefix="031525")
    elif argument == '5':
        calc_tendencies(region, in_prefix="031525")
    