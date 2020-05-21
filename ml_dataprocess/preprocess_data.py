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
import tempfile
import traceback

# Ocean only from u-bs572 and u-bs573 set aside for validation '0N100W'
# leave out 0N0E as that does not get read into iris properly
regions=['0N100W','0N130W','0N15W','0N160E','0N160W','0N30W','0N50E','0N70E','0N88E','10N100W','10N120W','10N140W','10N145E','10N160E','10N170W','10N30W','10N50W','10N60E','10N88E','10S120W','10S140W','10S15W','10S170E','10S170W','10S30W','10S5E','10S60E','10S88E','10S90W','20N135E','20N145W','20N170E','20N170W','20N30W','20N55W','20N65E','20S0E','20S100W','20S105E','20S130W','20S160W','20S30W','20S55E','20S80E','21N115W','29N65W','30N130W','30N145E','30N150W','30N170E','30N170W','30N25W','30N45W','30S100W','30S10E','30S130W','30S15W','30S160W','30S40W','30S60E','30S88E','40N140W','40N150E','40N160W','40N170E','40N25W','40N45W','40N65W','40S0E','40S100E','40S100W','40S130W','40S160W','40S50E','40S50W','50N140W','50N149E','50N160W','50N170E','50N25W','50N45W','50S150E','50S150W','50S30E','50S30W','50S88E','50S90W','60N15W','60N35W','60S0E','60S140E','60S140W','60S70E','60S70W','70N0E','70S160W','70S40W','80N150W']

suite_id="u-bs572_20170116-30_conc"
# suite_id ="u-bs572_20170101-15_conc"

def nooverlap_smooth(arrayin, window=6):
    """
    Moving average with non-overlapping window
    """
    if arrayin.ndim > 1:
        x,y=arrayin.shape
        averaged = np.mean(arrayin.reshape(window,x//window,y,order='F'),axis=0)
    else:
        x = arrayin.shape[0]
        averaged = np.mean(arrayin.reshape(window,x//window, order='F'), axis=0)
    return averaged

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
        q = "/project/spice/radiation/ML/CRM/data/{3}/{2}/concat_stash_{1}/30_days_50S69W_km1p5_ra1m_30x30_subdomain_{0}_{1}.nc".format(str(subdomain).zfill(3), str(10).zfill(5), region,suite_id)
        q_phys = "/project/spice/radiation/ML/CRM/data/{3}/{2}/tend_q_{1}/30_days_50S69W_km1p5_ra1m_30x30_subdomain_{0}_q_phys_{1}.nc".format(str(subdomain).zfill(3), str(10).zfill(5), region, suite_id)
        q_adv = "/project/spice/radiation/ML/CRM/data/{3}/{2}/concat_stash_{1}/30_days_50S69W_km1p5_ra1m_30x30_subdomain_{0}_{1}.nc".format(str(subdomain).zfill(3), str(12182).zfill(5), region, suite_id)
        phys_cubelist.append(iris.load_cube(q_phys))
        adv_cubelist.append(iris.load_cube(q_adv))
        q_cubelist.append(iris.load_cube(q))
    
    for q,qadv in zip(q_stashes,q_adv_stashes):
        print("Processing STASHES {0} {1}".format(q,qadv))
        for subdomain in range(64):
            q_file_name = "30_days_50S69W_km1p5_ra1m_30x30_subdomain_{0}_{1}.nc".format(str(subdomain).zfill(3), str(q).zfill(5))
            q_phys_file_name = "30_days_50S69W_km1p5_ra1m_30x30_subdomain_{0}_q_phys_{1}.nc".format(str(subdomain).zfill(3), str(q).zfill(5))
            q_adv_file_name = "30_days_50S69W_km1p5_ra1m_30x30_subdomain_{0}_{1}.nc".format(str(subdomain).zfill(3), str(qadv).zfill(5))

            q_file = "/project/spice/radiation/ML/CRM/data/{3}/{2}/concat_stash_{0}/{1}".format(str(q).zfill(5),q_file_name, region, suite_id)
            q_phys_file = "/project/spice/radiation/ML/CRM/data/{3}/{2}/tend_q_{0}/{1}".format(str(q).zfill(5),q_phys_file_name, region, suite_id)
            q_adv_file = "/project/spice/radiation/ML/CRM/data/{3}/{2}/concat_stash_{0}/{1}".format(str(qadv).zfill(5),q_adv_file_name, region, suite_id)

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
        iris.fileformats.netcdf.save(p,"/project/spice/radiation/ML/CRM/data/{2}/{1}/tend_qphys_tot/30_days_50S69W_km1p5_ra1m_30x30_subdomain_{0}_qphys_tot.nc".format(str(i).zfill(3), region, suite_id))
        iris.fileformats.netcdf.save(a,"/project/spice/radiation/ML/CRM/data/{2}/{1}/tend_qadv_tot/30_days_50S69W_km1p5_ra1m_30x30_subdomain_{0}_qadv_tot.nc".format(str(i).zfill(3), region, suite_id))
        iris.fileformats.netcdf.save(q,"/project/spice/radiation/ML/CRM/data/{2}/{1}/q_tot/30_days_50S69W_km1p5_ra1m_30x30_subdomain_{0}_q_tot.nc".format(str(i).zfill(3), region, suite_id))
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
        q = "/project/spice/radiation/ML/CRM/data/{4}/{2}/concat_stash_{1}/{3}_days_{2}_km1p5_ra1m_30x30_subdomain_{0}_{1}.nc".format(str(subdomain).zfill(3), str(10).zfill(5), region, in_prefix, suite_id)
        q_cubelist.append(iris.load_cube(q))
    
    for q in q_stashes:
        print("Processing STASHES {0}".format(q))
        for subdomain in range(64):
            q_file_name = "{3}_days_{2}_km1p5_ra1m_30x30_subdomain_{0}_{1}.nc".format(str(subdomain).zfill(3), str(q).zfill(5),region,in_prefix)
            q_file = "/project/spice/radiation/ML/CRM/data/{3}/{2}/concat_stash_{0}/{1}".format(str(q).zfill(5),q_file_name, region, suite_id)
            q_cube = iris.load_cube(q_file)
            (q_cubelist[subdomain]).data = q_cubelist[subdomain].data + q_cube.data

    save_path="/project/spice/radiation/ML/CRM/data/{1}/{0}/concat_stash_99821/".format(region, suite_id)         
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
    location='/net/spice/project/radiation/ML/CRM/data/{2}/{0}/stash_{1}/'.format(region, str(stash).zfill(5), suite_id)
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
    location='/net/spice/project/radiation/ML/CRM/data/{2}/{0}/stash_{1}/'.format(region, str(stash).zfill(5), suite_id)
    out_location='/project/spice/radiation/ML/CRM/data/{2}/{0}/concat_stash_{1}/'.format(region, str(stash).zfill(5), suite_id)
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
        # clist.append(cube[:-1,...])
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
    location='/project/spice/radiation/ML/CRM/data/{2}/{0}/concat_stash_{1}/'.format(region, str(stash).zfill(5), suite_id)
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
    out_location='/project/spice/radiation/ML/CRM/data/{2}/{0}/concat_stash_{1}/'.format(region, str(stash).zfill(5), suite_id)
    clist = []
    for filename in filelist:
        print("Reading {0}".format(filename))
        cube = iris.load_cube(filename)            
        clist.append(cube)
    cubelist = iris.cube.CubeList(clist)
    print("Saving file {0}".format(out_filename))
    iris.fileformats.netcdf.save(cubelist.concatenate()[0],out_location+out_filename)

def combine_day_tseries_dayrange(region: str, subdomain: int, stash: int, days_range=range(1,32), month=7):
    """
    Combine the per day files into a single file 
    """
    # First create a list of all the files that are going to be
    # combined into a single timeseries
    filelist = []
    location='/project/spice/radiation/ML/CRM/data/{2}/{0}/concat_stash_{1}/'.format(region, str(stash).zfill(5), suite_id)

    for day in days_range:
        date = datetime.date(2017, month, day)
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
    prefix=''.join([str(i).zfill(2) for i in days_range])
    out_filename =  "{0}_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_{3}.nc".format(prefix,region,str(subdomain).zfill(3),str(stash).zfill(5))
    out_location='/project/spice/radiation/ML/CRM/data/{2}/{0}/concat_stash_{1}/'.format(region, str(stash).zfill(5), suite_id)
    clist = []
    for filename in filelist:
        print("Reading {0}".format(filename))
        cube = iris.load_cube(filename)            
        clist.append(cube)
    cubelist = iris.cube.CubeList(clist)
    print("Saving file {0}".format(out_filename))
    iris.fileformats.netcdf.save(cubelist.concatenate()[0],out_location+out_filename)
    
def combine_files(region: str, day: int, stashes: list, month=7):
    date = datetime.date(2017, month, day)
    #for stash in [10,12182,16004,12181]:
    for stash in stashes:    
        for subdomain in range(64):
            combine_files_per_subdomain(region, date, subdomain, stash)
            

def main_check_files_exist(region: str, stashes: list, month=7):
    for day in range(1,32):
        date = datetime.date(2017, month, day)
        # for stash in [10,12182,16004,12181]:
        for stash in stashes:
            for subdomain in range(64):
                check_files_exist(region, date, subdomain, stash)
            
def main_combine_files(region: str, stashes: list, days_range=range(1,31), month=7):
    # day = sys.argv[1]
    try:
        for day in days_range:
            combine_files(region, day, stashes, month=month)
        return 0
    except Exception as e:
        sys.stderr.write("error: " + str(e))
        sys.stderr.write(traceback.format_exc())
        return 1
    #day=9
    #print(day)
    #main_combine_files(day)
        
def main_combine_day_tseries(region: str, stashes: list, days_range=[3,4,5], month=7):
    # region='10N160E'
    # Start from day 2 to ignore spin up day 1
    try:
        # for stash in [10,12182,16004,12181]:
        for stash in stashes:
            for subdomain in range(64):
                # combine_day_tseries(start_date, end_date, region, subdomain, stash)
                combine_day_tseries_dayrange(region, subdomain, stash, days_range=days_range, month=month)
        return 0
    except Exception as e:
        sys.stderr.write("error: " + str(e))
        sys.stderr.write(traceback.format_exc())
        return 1

def average_data(region: str, stashes: list, in_prefix):
    # region='10N160E'
    # Start from day 2 to ignore spin up day 1
    try:
        # for stash in [10,12182,16004,12181]:
        for stash in stashes:
            for subdomain in range(64):
                # combine_day_tseries(start_date, end_date, region, subdomain, stash)
                in_filename =  "{0}_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_{3}.nc".format(in_prefix,region,str(subdomain).zfill(3),str(stash).zfill(5))
                out_filename = "3h_{0}_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_{3}.nc".format(in_prefix,region,str(subdomain).zfill(3),str(stash).zfill(5))
                in_location='/project/spice/radiation/ML/CRM/data/{2}/{0}/concat_stash_{1}/'.format(region, str(stash).zfill(5), suite_id)
                print("Reading {0}".format(in_location+in_filename))
                cube = iris.load_cube(in_location+in_filename)
                name = cube.standard_name
                ave_data = nooverlap_smooth(cube.data, window=18)
                if ave_data.data.ndim > 1:
                    time = iris.coords.DimCoord(np.arange(ave_data.shape[0]), long_name="time")
                    levels = iris.coords.DimCoord(np.arange(ave_data.shape[1]), long_name="model_levels")
                    ave_cube = iris.cube.Cube(ave_data, dim_coords_and_dims=[(time,0),(levels,1)], standard_name=name)
                else:
                    time = iris.coords.DimCoord(np.arange(ave_data.shape[0]), long_name="time")
                    ave_cube = iris.cube.Cube(ave_data, dim_coords_and_dims=[(time,0)], standard_name=name)
                iris.fileformats.netcdf.save(ave_cube,in_location+out_filename)            
        return 0
    except Exception as e:
        sys.stderr.write("error: " + str(e))
        sys.stderr.write(traceback.format_exc())
        return 1

def create_delta(region, in_stash, out_stash, in_prefix: str="30"):
    """
    create a new variable that is q(n+1) - q(n) and t(n+1) - t(n)
    """
    # qtot stash 99821 - new stash 99822
    # t stash 00004 - new stash 99905
    in_location='/project/spice/radiation/ML/CRM/data/{2}/{0}/concat_stash_{1}/'.format(region, str(in_stash).zfill(5), suite_id)
    out_location='/project/spice/radiation/ML/CRM/data/{2}/{0}/concat_stash_{1}/'.format(region, str(out_stash).zfill(5), suite_id)
    try:
        os.makedirs(out_location)
    except OSError:
        pass
    for subdomain in range(64):
        in_filename =  "{0}_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_{3}.nc".format(in_prefix,region,str(subdomain).zfill(3),str(in_stash).zfill(5))
        out_filename = "{0}_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_{3}.nc".format(in_prefix,region,str(subdomain).zfill(3),str(out_stash).zfill(5))
        cube = iris.load_cube(in_location+in_filename)
        diff = tendencies.array_diff(cube.data[:], 1)
        time_coord = iris.coords.DimCoord(cube.coord('time').points[:-1],standard_name="time",units=cube.coord('time').units)
        # model_lev_coord = iris.coords.DimCoord(q_dat.coord('model_level_number').points,long_name="model_level_number")
        model_lev_coord = iris.coords.DimCoord(cube.coord('model_levels').points,long_name="model_levels")
        # long_name = cube.long_name + "_diff"
        var_name = cube.var_name+"_diff"
        new_cube = iris.cube.Cube(diff,var_name=var_name, dim_coords_and_dims=[(time_coord,0),(model_lev_coord,1)])
        print("Saving diff data ... {0}".format(out_location+out_filename))
        iris.fileformats.netcdf.save(new_cube,out_location+out_filename)

def calc_tendencies(region: str, in_prefix: str="30"):
    tendencies.main_Q_dot(region, suite_id, in_prefix=in_prefix)
    tendencies.main_T_dot(region, suite_id, in_prefix=in_prefix)   


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
        fname="{0}-{1}-{2}-{3}-pp-".format(suite_id,region,argument,stash)
        tmpf = tempfile.NamedTemporaryFile(prefix=fname,suffix='.lck',dir='/scratch/ojamil/slurmlock',delete=False)
        ret = main_combine_files(region, stashes, days_range=range(16,31), month=1)
        if ret == 0:
            os.remove(tmpf.name)
    elif argument == '3':
        fname="{0}-{1}-{2}-{3}-pp-".format(suite_id,region,argument,stash, month=1)
        tmpf = tempfile.NamedTemporaryFile(prefix=fname,suffix='.lck',dir='/scratch/ojamil/slurmlock',delete=False)
        ret = main_combine_day_tseries(region, stashes, days_range=range(16,31), month=1)
        if ret == 0:
            os.remove(tmpf.name)
    elif argument == '3a':
        in_prefix = "161718192021222324252627282930"
        # in_prefix = "0203040506070809101112131415"
        average_data(region, stashes, in_prefix)
    elif argument == '4':
        combine_q(region, in_prefix="3h_161718192021222324252627282930")
    elif argument == '4a':
        # qtot stash 99821 - new stash 99822
        # t stash 00004 - new stash 99905
        # in_stash = 4
        # out_stash = 99905
        in_stash = 99821
        out_stash = 99822
        in_prefix = "3h_161718192021222324252627282930"
        # in_prefix = "3h_0203040506070809101112131415"
        create_delta(region, in_stash, out_stash, in_prefix=in_prefix)
    elif argument == '5':
        calc_tendencies(region, in_prefix="3h_0203040506070809101112131415")
    