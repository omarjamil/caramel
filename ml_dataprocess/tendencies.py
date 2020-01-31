from netCDF4 import Dataset
import iris
import numpy as np
import os

def array_diff(inarray: np.array, step: int, axis: int=0):
    """
    Calculate element wise difference for 
    the array with a given step. Therefore
    the elements for difference do not have to consecutive
    """
    # Create a slice same shape as the in array
    slc1 = [slice(None)]*len(inarray.shape)
    slc2 = [slice(None)]*len(inarray.shape)
    end=-1*step
    axis_length = np.size(inarray,axis=axis)
    # The required axis will an actual slice over it
    # the other dimensions will be None which is the same as slice(:)
    slc1[axis] = slice(0,end)
    slc2[axis] = slice(step,axis_length)
    # Create two arrays and then take the difference
    a1 = inarray[tuple(slc1)]
    a2 = inarray[tuple(slc2)]
    diff = a2 - a1
    return diff

def nooverlap_smooth(arrayin, window=6):
    """
    Moving average with non-overlapping window
    """
    x,y=arrayin.shape
    averaged = np.mean(arrayin.reshape(window,x//window,y),axis=0)
    return averaged

def t_tendency(region: str, subdomain: int):
    """
    Physics tendency for q quantities
    specific humidity only to start
    Should combine QCL+QCF+Q_v+Q_graupel=Q_p
    dq/dt = Q_p(t_n) - Q_p(t_n-1) 
          = d(q_adv(t_n))/dt + d(q_phi(t_n))/dt
    We want d(q_phi(t_n))/dt which is the physics
    driven change to the quantity
    """
    t_phys_location='/project/spice/radiation/ML/CRM/data/u-bj775_/{0}/concat_stash_{1}/'.format(region, str(99904).zfill(5))
    
    try:
        os.makedirs(t_phys_location)
    except OSError:
        pass
    
    t_location='/project/spice/radiation/ML/CRM/data/u-bj775_/{0}/concat_stash_{1}/'.format(region, str(4).zfill(5))
    tadv_location='/project/spice/radiation/ML/CRM/data/u-bj775_/{0}/concat_stash_{1}/'.format(region, str(99181).zfill(5))
    d='30_days'
    t_file="{0}_{1}_km1p5_ra1m_30x30_subdomain_{2}_{3}.nc".format(d,region,str(subdomain).zfill(3),str(4).zfill(5))
    tadv_file="{0}_{1}_km1p5_ra1m_30x30_subdomain_{2}_{3}.nc".format(d,region,str(subdomain).zfill(3),str(99181).zfill(5))
    
    # t_dat = Dataset(t_file)
    # tadv_dat = Dataset(tadv_file)
    # t_array = t_dat['specific_humidity'][:]
    # tadv_array = tadv_dat['change_over_time_in_specific_humidity_due_to_advection'][:]
    t_dat = iris.load_cube(t_location+t_file)
    tadv_dat = iris.load_cube(tadv_location+tadv_file)
    t_array = t_dat.data[:]
    tadv_array = tadv_dat.data[:]
    t_diff = array_diff(t_array, 1)
    t_phys = t_diff - tadv_array[1:]
    
    # Save the data after normalising
    # Normalise on the fly? Allows different normalisations for training
    #t_phys_norm = normalise_data(t_phys)
    
    time_coord = iris.coords.DimCoord(t_dat.coord('time').points[:-1],standard_name="time",units=t_dat.coord('time').units)
    model_lev_coord = iris.coords.DimCoord(t_dat.coord('model_level_number').points,standard_name="model_level_number")
    new_cube = iris.cube.Cube(t_phys,long_name="t_phys",dim_coords_and_dims=[(time_coord,0),(model_lev_coord,1)])
    outfile="{0}_{1}_km1p5_ra1m_30x30_subdomain_{2}_{3}.nc".format(d,region,str(subdomain).zfill(3),str(99904).zfill(5))
    print("Saving T tendency ... {0}".format(t_phys_location+outfile))
    iris.fileformats.netcdf.save(new_cube,t_phys_location+outfile)
    



def q_tendency_qdot(region: str, subdomain: int):
    """
    Physics tendency for q quantities using qadv_dot and qtot

    specific humidity only to start
    Should combine QCL+QCF+Q_v+Q_graupel=Q_p
    dq/dt = Q_p(t_n) - Q_p(t_n-1) 
          = d(q_adv(t_n))/dt + d(q_phi(t_n))/dt
    We want d(q_phi(t_n))/dt which is the physics
    driven change to the quantity
    """
    q_phys_location='/project/spice/radiation/ML/CRM/data/u-bj775_/{0}/concat_stash_{1}/'.format(region, str(99983).zfill(5))
    try:
        os.makedirs(q_phys_location)
    except OSError:
        pass
    
    q_location='/project/spice/radiation/ML/CRM/data/u-bj775_/{0}/concat_stash_{1}/'.format(region, str(99821).zfill(5))
    qadv_location='/project/spice/radiation/ML/CRM/data/u-bj775_/{0}/concat_stash_{1}/'.format(region, str(99182).zfill(5))
    d='30_days'
    q_file="{0}_{1}_km1p5_ra1m_30x30_subdomain_{2}_{3}.nc".format(d,region,str(subdomain).zfill(3), str(99821).zfill(5))
    qadv_file="{0}_{1}_km1p5_ra1m_30x30_subdomain_{2}_{3}.nc".format(d,region,str(subdomain).zfill(3),str(99182).zfill(5))
    
    # q_dat = Dataset(q_file)
    # qadv_dat = Dataset(qadv_file)
    # q_array = q_dat['specific_humidity'][:]
    # qadv_array = qadv_dat['change_over_time_in_specific_humidity_due_to_advection'][:]
    q_dat = iris.load_cube(q_location+q_file)
    qadv_dat = iris.load_cube(qadv_location+qadv_file)
    q_array = q_dat.data[:]
    qadv_array = qadv_dat.data[:]*600.
    q_diff = array_diff(q_array, 1)
    q_phys = q_diff - qadv_array[:-1]
    
    # Save the data after normalising
    
    time_coord = iris.coords.DimCoord(q_dat.coord('time').points[:-1],standard_name="time",units=q_dat.coord('time').units)
    model_lev_coord = iris.coords.DimCoord(q_dat.coord('model_level_number').points,standard_name="model_level_number")
    new_cube = iris.cube.Cube(q_phys,long_name="q_phys",dim_coords_and_dims=[(time_coord,0),(model_lev_coord,1)])
    outfile="{0}_{1}_km1p5_ra1m_30x30_subdomain_{2}_{3}.nc".format(d,region,str(subdomain).zfill(3), str(99983).zfill(5))
    print("Saving q tendency ... {0}".format(q_phys_location+outfile))
    iris.fileformats.netcdf.save(new_cube,q_phys_location+outfile)

    # Hourly meaned data

    # q_array_smooth = nooverlap_smooth(q_dat.data[:,:], window=6)
    # qadv_array_smooth = nooverlap_smooth(qadv_dat.data[:,:]*600., window=6)
    # q_diff_smooth = array_diff(q_array_smooth, 1)
    # q_phys_smooth = q_diff_smooth - qadv_array_smooth[:-1]

    # time_coord_s = iris.coords.DimCoord(range(len(q_phys_smooth)),standard_name="time")
    # model_lev_coord_s = iris.coords.DimCoord(q_dat.coord('model_level_number').points,standard_name="model_level_number")
    # new_cube_s = iris.cube.Cube(q_phys_smooth,long_name="q_phys",dim_coords_and_dims=[(time_coord_s,0),(model_lev_coord_s,1)])
    # outfile_s="{0}_{1}_km1p5_ra1m_30x30_subdomain_{2}_q_phys_smooth.nc".format(d,region,str(subdomain).zfill(3))
    # print("Saving smoothed q tendency ... {0}".format(q_phys_location+outfile_s))
    # iris.fileformats.netcdf.save(new_cube_s,q_phys_location+outfile_s)

def main_Q_dot(region):
    for subdomain in range(64):
        q_tendency_qdot(region,subdomain)

def main_T_dot(region):
    for subdomain in range(64):
        t_tendency(region,subdomain)

if __name__ == "__main__":
    region="80S90W"
    # main_Q()
    main_Q_dot(region)
    main_T_dot(region)