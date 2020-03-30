#!/usr/bin/env python	
#
# Cyril Morcrette (2018), Met Office, UK
#
# Import some modules

from datetime import timedelta
import numpy as np
import iris
import iris.analysis
import subprocess
import os

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)+1):
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
    tmppath='/project/spice/radiation/ML/CRM/data/'+roseid+'/'+region+'/'
    filename=generate_ml_filename_in(date,vt,'.pp',stream,name_str,analysis_time,region)
    filein=tmppath+filename
    result = make_stash_string(stash_sec,stash_code)
    print(result)
    tmppath_out='/project/spice/radiation/ML/CRM/data/'+roseid+'/'+region+'/stash_'+result['stashstr_fout']+'/'
    try:
        os.makedirs(tmppath_out)
    except OSError:
        pass
    # Read in data
    
    fieldin = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    # Input array is 360 x 360
    # Take the central 240 x 240
    # N.B. python counts from 0
    # Check on whether the fields being processes is 2d or 3d (actually 3d or 4d with time dimension)
    if ndims==2:
        data=fieldin[:,60:300,60:300]
    if ndims==3:
        data=fieldin[:,:,60:300,60:300]
    # Now call the function that extracts and averages the LAM data onto GCM grid-boxes
    
    outcome=mean_subdomains(data,date,vt,name_str,analysis_time,region,result['stashstr_fout'],ndims,tmppath_out)
    return outcome;

def mean_subdomains(data,date,vt,name_str,analysis_time,region,output_label,ndims,tmppath):
    # Function that extracts the LAM data over an area corresponding to a GCM grid-boxes
    # and then averages it and writes it out.
    size_of_subdomains=[30]
    for s in np.arange(0,len(size_of_subdomains),1):
        dx=size_of_subdomains[s]
        subregion=0
        nx=np.int(240/size_of_subdomains[s])
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
                filenameout=generate_ml_filename_out(date,vt,'.nc',name_str,analysis_time,region,size_of_subdomains[s],subregion,output_label)
                fileout=tmppath+filenameout
                iris.save(horiz_meaned_data, fileout)
                subregion=subregion+1
    outcome=1
    return outcome;

def extract_fields_for_advective_tendencies(date,vt,roseid,name_str,analysis_time,region):
    tmppath = '/project/spice/radiation/ML/CRM/data/'+roseid+'/'+region+'/'
    tmppath_nc = tmppath+'netcdf/'
    tmppath_99181 = tmppath+'stash_99181/'
    tmppath_99182 = tmppath+'stash_99182/'

    try:
        os.makedirs(tmppath_nc)
    except OSError:
        pass
    try:
        os.makedirs(tmppath_99181)
    except OSError:
        pass
    try:
        os.makedirs(tmppath_99182)
    except OSError:
        pass     
    
    filename=generate_ml_filename_in(date,vt,'.pp','c',name_str,analysis_time,region)
    filein=tmppath+filename
    #
    # Read in moisture data
    result = make_stash_string(0,10)
    qv = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    result = make_stash_string(0,254)
    qcl = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    result = make_stash_string(0,12)
    qcf = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    result = make_stash_string(0,272)
    qrain = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    result = make_stash_string(0,273)
    qgraup = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    #
    # Add all the water variables together
    qtotal = qv+qcl+qcf+qrain+qgraup
    #
    # Read in dry potential temperature data
    result = make_stash_string(0,4)
    theta_dry = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    
    # pressure on theta levels not output by nested models so use global mean profile 
    # ocean_mean_profile_201701151200Z_exner_mean = np.array[(1.0016,1.0004,0.99879,0.99672,0.9942,0.99122,0.98778,0.98389,0.97955,0.97477,0.96955,0.96389,0.9578,0.95128,0.94434,0.93699,0.92923,0.92106,0.91249,0.90352,0.89416,0.8844,0.87427,0.86375,0.85286,0.8416,0.82998,0.81799,0.80565,0.79296,0.77992,0.76654,0.75283,0.73881,0.72447,0.70985,0.69496,0.67983,0.66448,0.64889,0.63305,0.61696,0.6006,0.58394,0.56694,0.54955,0.53171,0.51336,0.49449,0.4751,0.45521,0.43483,0.41391,0.39236,0.37009,0.34711,0.32345,0.29919,0.27445,0.24943,0.22439,0.19967,0.17564,0.15263,0.13059,0.10925,0.088742,0.069395,0.051218,0.034236])
    
    # Read in pressure on theta levels
    filename=generate_ml_filename_in(date,vt,'.pp','f',name_str,analysis_time,region)
    filein=tmppath+filename
    result = make_stash_string(0,408)
    p_theta_levels = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    # Tl=T-(L/cp)*qcl
    # T=theta * exner
    # exner=(p/pref) ** kay
    kay=0.286
    exner=np.power((p_theta_levels.data/1.0e5),kay)
    # exner = ocean_mean_profile_201701151200Z_exner_mean
    lv=2.501e6
    lf=2.834e6
    cp=1005.0
    lvovercp=lv/cp
    lfovercp=lf/cp
    liq=qcl.data+qrain.data
    ice=qcf.data+qgraup.data
    # Calculate a liq/ice static temperature 
    theta=theta_dry.data-(lvovercp*liq/exner)-(lfovercp*ice/exner)
    #
    # Read in wind data
    # NB u wind is staggered half a grid-box to the west  and half a layer down.
    # NB v wind is staggered half a grid-box to the south and half a layer down.
    # NB w wind is on same grid and theta and q
    filename=generate_ml_filename_in(date,vt,'.pp','e',name_str,analysis_time,region)
    filein=tmppath+filename
    result = make_stash_string(0,2)
    u_wind = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    result = make_stash_string(0,3)
    v_wind = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    result = make_stash_string(0,150)
    w_wind = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    #
    flux_qtotal=u_dot_grad_field(u_wind,v_wind,w_wind,qtotal,'qtotal')
    flux_theta=u_dot_grad_field(u_wind,v_wind,w_wind,theta_dry.copy(theta),'theta')
    #
    write_out_intermediate_data=0
    if write_out_intermediate_data==1:
        # Write everything out so can potentially check things offline.
        #
        # vt=validity time
        if vt < 10:
            vtstr='00'+str(vt) 
        else:
            vtstr='0'+str(vt)
        # endif
        fileout=tmppath+'/netcdf/'+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_qtotal.nc'
        iris.save(qtotal, fileout)
        fileout=tmppath+'/netcdf/'+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_theta.nc'
        iris.save(theta, fileout)
        fileout=tmppath+'/netcdf/'+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_uwind.nc'
        iris.save(u_wind, fileout)
        fileout=tmppath+'/netcdf/'+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_vwind.nc'
        iris.save(v_wind, fileout)
        fileout=tmppath+'/netcdf/'+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_wwind.nc'
        iris.save(w_wind, fileout)
    # endif
    #
    write_out_total_flux_data=0
    if write_out_total_flux_data==1:
        fileout=tmppath+'/netcdf/'+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_python_flux_qtotal.nc'
        iris.save(flux_qtotal, fileout)
        fileout=tmppath+'/netcdf/'+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_python_flux_theta.nc'
        iris.save(flux_theta, fileout)
    # endif
    #
    # Make up some stash numbers so file has 5 digit ref but these are not actual UM stash codes
    # 99181 theta increment from advection
    # 99182 total q increment from advection
    outcome=mean_subdomains(flux_theta[:,:,60:300,60:300],date,vt,name_str,analysis_time,region,'99181',3,tmppath_99181)
    outcome=mean_subdomains(flux_qtotal[:,:,60:300,60:300],date,vt,name_str,analysis_time,region,'99182',3,tmppath_99182)
    return outcome;

def _extract_fields_for_advective_tendencies(date,vt,roseid,name_str,analysis_time,region):
    tmppath = '/project/spice/radiation/ML/CRM/data/'+roseid+'/'+region+'/'
    tmppath_nc = tmppath+'netcdf/'
    tmppath_99181 = tmppath+'stash_99181/'
    tmppath_99182 = tmppath+'stash_99182/'

    try:
        os.makedirs(tmppath_nc)
    except OSError:
        pass
    try:
        os.makedirs(tmppath_99181)
    except OSError:
        pass
    try:
        os.makedirs(tmppath_99182)
    except OSError:
        pass
    
        
    filename=generate_ml_filename_in(date,vt,'.pp','c',name_str,analysis_time,region)
    filein=tmppath+filename
    #
    # Read in moisture data
    result = make_stash_string(0,10)
    qv = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    result = make_stash_string(0,254)
    qcl = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    result = make_stash_string(0,12)
    qcf = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    result = make_stash_string(0,272)
    qrain = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    result = make_stash_string(0,273)
    qgraup = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    #
    # Add all the water variables together
    qtotal = qv+qcl+qcf+qrain+qgraup
    #TEST
    # qtotal = qv
    #TEST
    #
    # Read in dry potential temperature data
    result = make_stash_string(0,4)
    theta = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    #
    # Read in wind data
    # NB u wind is staggered half a grid-box to the west  and half a layer down.
    # NB v wind is staggered half a grid-box to the south and half a layer down.
    # NB w wind is on same grid and theta and q
    filename=generate_ml_filename_in(date,vt,'.pp','e',name_str,analysis_time,region)
    filein=tmppath+filename
    result = make_stash_string(0,2)
    u_wind = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    result = make_stash_string(0,3)
    v_wind = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    result = make_stash_string(0,150)
    w_wind = iris.load_cube(filein,iris.AttributeConstraint(STASH=result['stashstr_iris']))
    #
    flux_qtotal=u_dot_grad_field(u_wind,v_wind,w_wind,qtotal,'qtotal')
    flux_theta=u_dot_grad_field(u_wind,v_wind,w_wind,theta,'theta')
    #
    # vt=validity time
    if vt < 10:
        vtstr='00'+str(vt) 
    else:
        vtstr='0'+str(vt)
    write_out_intermediate_data=0
    if write_out_intermediate_data==1:
        # Write everything out so can potentially check things offline.
        fileout=tmppath_nc+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_qtotal.nc'
        iris.save(qtotal, fileout)
        fileout=tmppath_nc+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_theta.nc'
        iris.save(theta, fileout)
        fileout=tmppath_nc+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_uwind.nc'
        iris.save(u_wind, fileout)
        fileout=tmppath_nc+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_vwind.nc'
        iris.save(v_wind, fileout)
        fileout=tmppath_nc+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_wwind.nc'
        iris.save(w_wind, fileout)
    # endif
    #
    write_out_total_flux_data=0
    if write_out_total_flux_data==1:
        fileout=tmppath_nc+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_flux_qtotal.nc'
        iris.save(flux_qtotal, fileout)
        fileout=tmppath_nc+date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_'+vtstr+'_flux_theta.nc'
        iris.save(flux_theta, fileout)
    # endif
    #
    # Make up some stash numbers so file has 5 digit ref but these are not actual UM stash codes
    # 99181 theta increment from advection
    # 99182 total q increment from advection
    outcome=mean_subdomains(flux_theta[:,:,60:300,60:300],date,vt,name_str,analysis_time,region,'99181',3,tmppath_99181)
    outcome=mean_subdomains(flux_qtotal[:,:,60:300,60:300],date,vt,name_str,analysis_time,region,'99182',3,tmppath_99182)
    return outcome

def generate_filename_in(date,vt,ext,stream,name_str,analysis_time,region):
    if vt < 10:
        vtstr='00'+str(vt) 
    else:
        vtstr='0'+str(vt)
    # endif
    filename=date.strftime("%Y%m%d")+analysis_time+'_'+region+'_km1p5_'+name_str+'_pver'+stream+vtstr+ext
    print(filename)
    #
    return filename

def retrieve_a_file(date,vt,roseid,name_str,analysis_time,stream,region,flag):
    # If flag == 1, it deletes the file and then retrieves a clean copy
    # If flag == 0, it just deletes the file.
    #
    tmppath='/project/spice/radiation/ML/CRM/data/'+roseid+'/'+region+'/adv/'
    try:
        os.makedirs(tmppath)
    except OSError:
        pass
    
    moopath='moose:/devfc/'+roseid+'/field.pp/'
    filename=generate_filename_in(date,vt,'.pp',stream,name_str,analysis_time,region)
    # moo retrieval will not overwrite an existing file.
    # So remove any existing occurrence of the file we are trying to retrieve
    stalefile=tmppath+filename
    subprocess.call(["rm", stalefile])
    if flag==1:
        fullname=moopath+filename
        subprocess.call(["moo","get", fullname, tmppath])
    # endif
    outcome=1
    return outcome;

def u_dot_grad_field(u_wind,v_wind,w_wind,field,field_str):
    # Use L70 with an 80 km top (needed for calculating ddz).
    z_top_of_model =  80000.0
    eta_theta=np.array([ .0002500,  .0006667,  .0012500,  .0020000,  .0029167,  .0040000,  
                         .0052500,  .0066667,  .0082500,  .0100000,  .0119167,  .0140000,  
                         .0162500,  .0186667,  .0212500,  .0240000,  .0269167,  .0300000,  
                         .0332500,  .0366667,  .0402500,  .0440000,  .0479167,  .0520000,  
                         .0562500,  .0606667,  .0652500,  .0700000,  .0749167,  .0800000,  
                         .0852500,  .0906668,  .0962505,  .1020017,  .1079213,  .1140113,  
                         .1202745,  .1267154,  .1333406,  .1401592,  .1471838,  .1544313,  
                         .1619238,  .1696895,  .1777643,  .1861929,  .1950307,  .2043451,  
                         .2142178,  .2247466,  .2360480,  .2482597,  .2615432,  .2760868,  
                         .2921094,  .3098631,  .3296378,  .3517651,  .3766222,  .4046373,  
                         .4362943,  .4721379,  .5127798,  .5589045,  .6112759,  .6707432,  
                         .7382500,  .8148403,  .9016668, 1.0000000])
    height_theta_levels=eta_theta*z_top_of_model
    # Everything written assuming UM's staggered grid.
    # ------------------------------------------------
    # NB (x,y) space is:
    #                     theta(i,j+1)
    #                      
    #                     v(i,j+1)
    #                      
    # theta(i-1,j) u(i,j) theta(i,j)   u(i+1,j) theta(i+1,j)
    #                      
    #                     v(i,j)
    #                       
    #                     theta(i,j-1)
    #
    # ------------------------------------------------
    # NB (x,z) space is:
    #
    # 80000 m
    #                     76066m
    # ...                 ...           
    # 53m theta,w (i,k+1)
    #                     36m u,v (i,k+1)
    # 20m theta,w (i,k)
    #                     10m u,v (i,k)
    # ----------Ground level--------------------------
    #
    # NB dimensions are (time, height, latitude, longitude)
    # i.e.              (t,    z,      y,        x        )
    #
    # i component
    dqdx_lhs=(field.data[:,:,60:300,60:300]-field.data[:,:,60:300,59:299])/1500.0
    dqdx_rhs=(field.data[:,:,60:300,61:301]-field.data[:,:,60:300,60:300])/1500.0
    #
    nt=field.shape[0]
    nz=field.shape[1]
    nx=field.shape[2]
    ny=field.shape[3]
    #
    u_lhs_half_lev_below=u_wind.data[:,:,60:300,60:300]
    u_rhs_half_lev_below=u_wind.data[:,:,60:300,61:301]
    #
    # Do this to get array of correct size
    u_lhs_half_lev_above=u_lhs_half_lev_below*0.0
    u_rhs_half_lev_above=u_rhs_half_lev_below*0.0
    #
    for k in np.arange(0,nz-1,1):
        # Copy information from one layer higher up
        u_lhs_half_lev_above[:,k,:,:]=u_lhs_half_lev_below[:,k+1,:,:]
        u_rhs_half_lev_above[:,k,:,:]=u_rhs_half_lev_below[:,k+1,:,:]
    # For top-most level (i.e. 70th level (69th in python-speak) set it to same as layer below it
    u_lhs_half_lev_above[:,nz-1,:,:]=u_lhs_half_lev_above[:,nz-2,:,:]
    u_rhs_half_lev_above[:,nz-1,:,:]=u_rhs_half_lev_above[:,nz-2,:,:]
    #
    # Linear-average (no pressure or density weighting) of values on half level above 
    # and below to get horizontal wind on this theta level
    u_theta_lev_lhs=(u_lhs_half_lev_above+u_lhs_half_lev_below)*0.5
    u_theta_lev_rhs=(u_rhs_half_lev_above+u_rhs_half_lev_below)*0.5
    # Calculate flux coming in and going out
    u_dqdx_flux_in =u_theta_lev_lhs*dqdx_lhs
    u_dqdx_flux_out=u_theta_lev_rhs*dqdx_rhs
    #
    # j component
    dqdy_south=(field.data[:,:,60:300,60:300]-field.data[:,:,59:299,60:300])/1500.0
    dqdy_north=(field.data[:,:,61:301,60:300]-field.data[:,:,60:300,60:300])/1500.0
    #
    v_south_half_lev_below=v_wind.data[:,:,60:300,60:300]
    v_north_half_lev_below=v_wind.data[:,:,61:301,60:300]
    #
    # Do this to get array of correct size
    v_south_half_lev_above=v_south_half_lev_below*0.0
    v_north_half_lev_above=v_north_half_lev_below*0.0
    #
    for k in np.arange(0,nz-1,1):
        # Copy information from one layer higher up
        v_south_half_lev_above[:,k,:,:]=v_south_half_lev_below[:,k+1,:,:]
        v_north_half_lev_above[:,k,:,:]=v_north_half_lev_below[:,k+1,:,:]
    # For top-most level (i.e. 70th level (69th in python-speak) set it to same as layer below it
    v_south_half_lev_above[:,nz-1,:,:]=v_south_half_lev_above[:,nz-2,:,:]
    v_north_half_lev_above[:,nz-1,:,:]=v_north_half_lev_above[:,nz-2,:,:]
    #
    # Linear-average (no pressure or density wieghting) of values on half level above 
    # and below to get horizontal wind on this theta level
    v_theta_lev_south=(v_south_half_lev_above+v_south_half_lev_below)*0.5
    v_theta_lev_north=(v_north_half_lev_above+v_north_half_lev_below)*0.5
    # Calculate flux coming in and going out
    v_dqdy_flux_in =v_theta_lev_south*dqdy_south
    v_dqdy_flux_out=v_theta_lev_north*dqdy_north
    #
    # k component
    # NB w is held on same levels as theta and q
    w_half_lev_below=w_wind.data[:,:,60:300,60:300]*0.0
    w_half_lev_above=w_wind.data[:,:,60:300,60:300]*0.0
    for k in np.arange(1,nz-1,1):
        w_half_lev_below[:,k,:,:]=(w_wind.data[:,k,60:300,60:300]+w_wind.data[:,k-1,60:300,60:300])*0.5
    w_half_lev_below[:,0,:,:]=0.0
    #
    for k in np.arange(0,nz-2,1):
        w_half_lev_above[:,k,:,:]=(w_wind.data[:,k,60:300,60:300]+w_wind.data[:,k+1,60:300,60:300])*0.5
    w_half_lev_above[:,nz-1,:,:]=0.0
    #
    dqdz_half_lev_below=field.data[:,:,60:300,60:300]*0.0
    for k in np.arange(1,nz-1,1):
        dqdz_half_lev_below[:,k,:,:]=(field.data[:,k,60:300,60:300]-field.data[:,k-1,60:300,60:300])/(height_theta_levels[k]-height_theta_levels[k-1])
    dqdz_half_lev_below[:,0,:,:]=0.0
    #
    dqdz_half_lev_above=field.data[:,:,60:300,60:300]*0.0
    for k in np.arange(0,nz-2,1):
        dqdz_half_lev_above[:,k,:,:]=(field.data[:,k+1,60:300,60:300]-field.data[:,k,60:300,60:300])/(height_theta_levels[k+1]-height_theta_levels[k])
    dqdz_half_lev_above[:,nz-1,:,:]=0.0
    #
    # Calculate flux coming in and going out
    w_dqdz_flux_in =w_half_lev_below*dqdz_half_lev_below
    w_dqdz_flux_out=w_half_lev_above*dqdz_half_lev_above
    #
    # Combine i, j, k components
    #
    # Seem to get memory issues is try to do this in one line.
    # net_flux=(u_dqdx_flux_in-u_dqdx_flux_out)+(v_dqdy_flux_in-v_dqdy_flux_out)+(w_dqdz_flux_in-w_dqdz_flux_out)
    x_bit=(u_dqdx_flux_in-u_dqdx_flux_out)
    y_bit=(v_dqdy_flux_in-v_dqdy_flux_out)
    z_bit=(w_dqdz_flux_in-w_dqdz_flux_out)
    net_flux=x_bit+y_bit+z_bit
    #
    #
    net_flux_cube=field*0.0
    net_flux_cube.data[:,:,60:300,60:300]=net_flux
    return net_flux_cube;
