#!/usr/bin/env python	
#
# Cyril Morcrette (2018), Met Office, UK
#
# Import some modules

from datetime import timedelta
import numpy as np
import iris
import iris.analysis
import os

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def make_stash_string(stashsec,stashcode):
    #
    stashsecstr=str(stashsec)
    if stashsec<10:
        stashsecstr='0'+stashsecstr
    # endif
    #
    stashcodestr=str(stashcode)
    if stashcode<100:
        stashcodestr='0'+stashcodestr
    # endif
    if stashcode<10:
        stashcodestr='0'+stashcodestr
    # endif
    stashstr_iris='m01s'+stashsecstr+'i'+stashcodestr
    stashstr_fout=stashsecstr+stashcodestr
    return {'stashstr_iris':stashstr_iris, 'stashstr_fout':stashstr_fout};

def step(start,inc,end):
    # A nice function to get around the stupid way that np.linspace works
    num=round((end-start)/inc)+1.0
    number_array=np.linspace(start,end,num)
    return number_array;

def generate_ml_filename_in(date,vt,ext,stream,name_str,analysis_time,region):
    # vt=validity time
    if vt < 10:
        vtstr='00'+str(vt) 
    else:
        vtstr='0'+str(vt)
    # endif
    filename=date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_pver'+stream+vtstr+ext
    print(filename)
    #
    return filename;

def generate_ml_filename_out(date,vt,ext,name_str,analysis_time,region,size,subregion,stashnumber):
    # vt=validity time
    if vt < 10:
        vtstr='00'+str(vt) 
    else:
        vtstr='0'+str(vt)
    # endif
    filename=date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+str(size)+'x'+str(size)+'_subdomain_'+str(subregion).zfill(3)+'_vt_'+vtstr+'_'+stashnumber+ext
    print(filename)
    #
    return filename;

def process_ml_lam_file(date,vt,roseid,name_str,analysis_time,stream,region,stash_sec,stash_code,ndims):
    # Currently hard-coded for user "frme"
    # tmppath='/project/spice/radiation/ML/CRM/data/'+roseid+'/'+region+'/'
    tmppath='/project/spice/radiation/ML/CRM/data/'+roseid+'_/'+region+'/'

    filename=generate_ml_filename_in(date,vt,'.pp',stream,name_str,analysis_time,region)
    filein=tmppath+filename
    # Read in data
    result = make_stash_string(stash_sec,stash_code)
    # tmppath_out='/project/spice/radiation/ML/CRM/data/'+roseid+'/'+region+'/stash_'+result['stashstr_fout']+'/'
    tmppath_out='/project/spice/radiation/ML/CRM/data/'+roseid+'_/'+region+'/stash_'+result['stashstr_fout']+'/'
    print(filein, result['stashstr_iris'])
    # if os.path.exists(tmppath_out) method can create a race condition when several scripts launched
    try:
        os.makedirs(tmppath_out)
    except OSError:
        pass
    fieldin = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    # Input array is 360 x 360
    # Take the central 240 x 240
    # N.B. python counts from 0
    # Check on whether the fields being processes is 2d or 3d (actually 3d or 4d with time dimension)
    if ndims==2:
        data=fieldin[:,60:300,60:300]
    if ndims==3:
        data=fieldin[:,:,60:300,60:300]
    size_of_subdomains=[30]
    for s in np.arange(0,len(size_of_subdomains),1):
        dx=size_of_subdomains[s]
        subregion=0
        nx=np.int(240/size_of_subdomains[s])
        print(nx)
        for j in np.arange(0,nx,1):
            for i in np.arange(0,nx,1):
                startx=i*dx
                endx=(i+1)*dx
                starty=j*dx
                endy=(j+1)*dx
                print('Subregion=',subregion,' extracting from:',startx,endx,starty,endy)
                print('X:{0} {1} Y:{2} {3}'.format(startx,endx,starty,endy))
                if ndims==2:
                    subdata=data[:,startx:endx,starty:endy]
                if ndims==3:
                    subdata=data[:,:,startx:endx,starty:endy]
                horiz_meaned_data = subdata.collapsed(['grid_longitude','grid_latitude'], iris.analysis.MEAN)
                filenameout=generate_ml_filename_out(date,vt,'.nc',name_str,analysis_time,region,size_of_subdomains[s],subregion,result['stashstr_fout'])
                fileout=tmppath_out+filenameout
                print(fileout)
                iris.save(horiz_meaned_data, fileout)
                subregion=subregion+1
    outcome=1
    return outcome;
