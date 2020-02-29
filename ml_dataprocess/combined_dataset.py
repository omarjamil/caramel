"""
Take the concated datasets per stash and per subdomain and combine for all the regions into a single dataset
"""

import numpy as np
from netCDF4 import Dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import h5py
import os
import iris
from itertools import chain

nn_data_stashes = {
    4:"air_potential_temperature",
    99181:"t_adv",
    99182:"q_adv",
    99821:"q_tot",
    1207:"toa_incoming_shortwave_flux",
    3217:"surface_upward_sensible_heat_flux",
    3234:"surface_upward_latent_heat_flux",
    99904:"t_phys",
    99983:"q_phys"
}

multi_stashes = {
    4:"air_potential_temperature",
    10:"specific_humidity",
    12:"mass_fraction_of_cloud_ice_in_air",
    254:"mass_fraction_of_cloud_liquid_water_in_air",
    272:"mass_fraction_of_rain_in_air",
    273:"mass_fraction_of_graupel_in_air",
    12181:"change_over_time_in_air_temperature_due_to_advection",
    12182:"change_over_time_in_specific_humidity_due_to_advection",
    12183:"change_over_time_in_mass_fraction_of_cloud_liquid_water_in_air_due_to_advection",
    12184:"change_over_time_in_mass_fraction_of_cloud_ice_in_air_due_to_advection",
    12189:"change_over_time_in_mass_fraction_of_rain_in_air_due_to_advection",
    12190:"change_over_time_in_mass_fraction_of_graupel_in_air_due_to_advection",
    16004:"air_temperature",
    99181:"unknown",
    99182:"unknown",
    99821:"q_tot",
    99904:"t_phys",
    99983:"q_phys"
}
surface_stashes = {
    24:"surface_temperature",
    1202:"m01s01i202",
    1205:"toa_outgoing_shortwave_flux",
    1207:"toa_incoming_shortwave_flux",
    1208:"toa_outgoing_shortwave_flux",
    2201:"surface_net_downward_longwave_flux",
    2205:"toa_outgoing_longwave_flux",
    2207:"surface_downwelling_longwave_flux_in_air",
    3217:"surface_upward_sensible_heat_flux",
    3234:"surface_upward_latent_heat_flux",
    3225:"x_wind",
    3226:"y_wind",
    3236:"air_temperature",
    3245:"relative_humidity",
    4203:"stratiform_rainfall_flux",
    4204:"stratiform_snowfall_flux",
    9217:"cloud_area_fraction_assuming_maximum_random_overlap",
    16222:"air_pressure_at_sea_level",
    30405:"atmosphere_cloud_liquid_water_content",
    30406:"atmosphere_cloud_ice_content",
    30461:"m01s30i461",
}


regions = ['50N144W','10S120W','10N120W','20S112W','0N90W','30N153W','80S90E','40S90W','10N80E','0N90E','10S80W','50S0E','70S0E','0N162W','30S102W','40N90E','70S120E','60N135W','70S120W','50N72E','40S30E','10N40W','20S22E','10N40E','30N105E','20N157E','40S150W','40S30W','30S153W','0N54E','50S144W','20S67E','60N45E','10S40W','60S45E','20S112E','20N67W','0N126W','0N126E','10N0E','20S157W','50N0E','20N118W','50S144E','30N51W','20N22W','40S90E','30S153E','30N0E','20N157W','40N150W','20N67E','0N162E','70N120E','30N51E','20S67W','20N22E','70N0E','40S150E','20S22W','80N90E','40N150E','70N120W','10N160W','50N72W','60S135W','60S39W','80N90W','60S135E','30S102E','10S0E','10N160E','30N102W','20N112E','10S160E','10S80E','60N135E','30S51E','30N153E','10N120E','0N18W','30S51W','10S120E','40N30E','40N90W','0N54W','30S0E','60N45W','20S157E','50S72E','0N18E','40N30W']

crm_data = "/project/spice/radiation/ML/CRM/data"

def standardise_data_transform(dataset: np.array([]), region: str, save_fname: str="std_fit.hdf5", levs: bool=False):
    """
    Manually standardise data based instead of using sklearn standarad scaler
    """
    save_location = "{0}/models/normaliser/{1}/".format(crm_data,region)
    # save_location = "{0}/models/normaliser/{1}_noshuffle/".format(crm_data,region)
    try:
        os.makedirs(save_location)
    except OSError:
        pass
    # per level normalisation
    if levs:
        mean = np.array([np.mean(dataset, axis=0)])
        scale = np.array([np.std(dataset, axis=0)])
    else:
        # Mean across the entire dataset and levels
        mean = np.array([np.mean(dataset)])
        scale = np.array([np.std(dataset)])
    params = {"mean_":mean, "scale_":scale}
    with h5py.File(save_location+save_fname, 'w') as hfile:
        for k, v in params.items():  
            hfile.create_dataset(k,data=v)
    results = (dataset - mean)/scale
    return results, dataset

def combine_multi_level_files(in_prefix="031525"):
    """
    Combine all the regions into a single file
    """
    new_region = "9999NEWS"
    out_basedir = "/project/spice/radiation/ML/CRM/data/u-bj775_/{0}/".format(new_region)
    try:
        os.makedirs(out_basedir)
    except OSError:
        pass

    for s in multi_stashes:
        for subdomain in range(64):
            out_dir = out_basedir+"concat_stash_{0}/".format(str(s).zfill(5))
            try:
                os.makedirs(out_dir)
            except OSError:
                pass
            outfile="{0}/{3}_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_{4}.nc".format(out_dir, new_region, str(subdomain).zfill(3),in_prefix, str(s).zfill(5))
            var_ = None
            var_cube = None
            irx = 0
            for r in regions:
                in_dir = "/project/spice/radiation/ML/CRM/data/u-bj775_/{0}/concat_stash_{1}/".format(r,str(s).zfill(5))
                in_file = "{0}/{3}_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_{4}.nc".format(in_dir, r, str(subdomain).zfill(3),in_prefix, str(s).zfill(5))
                # print("Processing file: {0}".format(in_file))
                data = Dataset(in_file)
                if s in [99904, 99983]:
                    var = data[multi_stashes[s]][:]
                else:
                    var = data[multi_stashes[s]][:-1,:]
                if irx == 0:
                    var_ = var
                else:
                    var_ = np.concatenate((var_,var),axis=0)
                irx += 1
            time = iris.coords.DimCoord(np.arange(var_.shape[0]), long_name="time")
            levels = iris.coords.DimCoord(np.arange(var_.shape[1]), long_name="model_levels")
            var_cube = iris.cube.Cube(var_, dim_coords_and_dims=[(time,0),(levels,1)])
            var_cube.var_name = multi_stashes[s]
            var_cube.attributes['STASHES'] = str(s).zfill(5)
            print("Saving file {0}".format(outfile))
            iris.fileformats.netcdf.save(var_cube, outfile)

def combine_surface_level_files(in_prefix="031525"):
    """
    Combine all the regions into a single file
    """
    new_region = "9999NEWS"
    out_basedir = "/project/spice/radiation/ML/CRM/data/u-bj775_/{0}/".format(new_region)
    try:
        os.makedirs(out_basedir)
    except OSError:
        pass

    for s in surface_stashes:
        for subdomain in range(64):
            out_dir = out_basedir+"concat_stash_{0}/".format(str(s).zfill(5))
            try:
                os.makedirs(out_dir)
            except OSError:
                pass
            outfile="{0}/{3}_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_{4}.nc".format(out_dir, new_region, str(subdomain).zfill(3),in_prefix, str(s).zfill(5))
            var_ = None
            var_cube = None
            irx = 0
            for r in regions:
                in_dir = "/project/spice/radiation/ML/CRM/data/u-bj775_/{0}/concat_stash_{1}/".format(r,str(s).zfill(5))
                in_file = "{0}/{3}_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_{4}.nc".format(in_dir, r, str(subdomain).zfill(3),in_prefix, str(s).zfill(5))
                # print("Processing file: {0}".format(in_file))
                data = Dataset(in_file)
                if s in [99904, 99983]:
                    var = data[surface_stashes[s]][:]
                else:
                    var = data[surface_stashes[s]][:-1]
                if irx == 0:
                    var_ = var
                else:
                    var_ = np.concatenate((var_,var),axis=0)
                irx += 1
            time = iris.coords.DimCoord(np.arange(var_.shape[0]), long_name="time")
            var_cube = iris.cube.Cube(var_, dim_coords_and_dims=[(time,0)])
            var_cube.var_name = surface_stashes[s]
            var_cube.attributes['STASHES'] = str(s).zfill(5)
            print("Saving file {0}".format(outfile))
            iris.fileformats.netcdf.save(var_cube, outfile)

def combine_subdomains(region: str, in_prefix="031525"):
    """
    After combining data from all the regions
    now combine data from all the subregions into a single dataset per stash 
    """
    stashes = {**surface_stashes, **multi_stashes}
    for s in stashes:
        inoutdir = "/project/spice/radiation/ML/CRM/data/u-bj775_/{0}/concat_stash_{1}".format(region, str(s).zfill(5))
        outfile = "{0}/{1}_days_{2}_km1p5_ra1m_30x30_{3}.nc".format(inoutdir, in_prefix, region, str(s).zfill(5))
        var_ = None
        var_cube = None
        
        for subdomain in range(64):
            infile = "{0}/{1}_days_{2}_km1p5_ra1m_30x30_subdomain_{3}_{4}.nc".format(inoutdir, in_prefix, region, str(subdomain).zfill(3),str(s).zfill(5))
            print("{0}".format(infile))
            data = Dataset(infile)
            var = data[stashes[s]][:]
            if subdomain == 0:
                var_ = var
            else:
                var_ = np.concatenate((var_,var),axis=0)
        if var_.ndim > 1:
            time = iris.coords.DimCoord(np.arange(var_.shape[0]), long_name="time")
            levels = iris.coords.DimCoord(np.arange(var_.shape[1]), long_name="model_levels")
            var_cube = iris.cube.Cube(var_, dim_coords_and_dims=[(time,0),(levels,1)])
        else:
            time = iris.coords.DimCoord(np.arange(var_.shape[0]), long_name="time")
            var_cube = iris.cube.Cube(var_, dim_coords_and_dims=[(time,0)])
        if s == 99181:
            var_cube.var_name = "t_adv"
        elif s == 99182:
            var_cube.var_name = "q_adv"
        else:
            var_cube.var_name = stashes[s]
        var_cube.attributes['STASHES'] = str(s).zfill(5)
        print("Saving file {0}".format(outfile))
        iris.fileformats.netcdf.save(var_cube, outfile)

def nn_dataset(region:str, in_prefix="031525"):
    """
    Create dataset for the neural network training and testing
    """   
    # NN input data
    data_labels = []
    data = []
    raw_data = []
    for s in nn_data_stashes:
        indir = "/project/spice/radiation/ML/CRM/data/u-bj775_/{0}/concat_stash_{1}".format(region, str(s).zfill(5))
        infile="{0}/{1}_days_{2}_km1p5_ra1m_30x30_{3}.nc".format(indir, in_prefix, region, str(s).zfill(5))
        print("Processing {0}".format(infile))
        dataf = Dataset(infile)
        var = dataf[nn_data_stashes[s]][:]
        std_fname=nn_data_stashes[s]+".hdf5"
        normed_var, raw_var = standardise_data_transform(var, region, save_fname=std_fname)
        data_labels.append(nn_data_stashes[s])
        data.append(normed_var)
        raw_data.append(raw_var)
    
    data_std_split = train_test_split(*data, shuffle=True, random_state=18)
    raw_data_split = train_test_split(*raw_data, shuffle=True, random_state=18)
    # data_std_split = train_test_split(*data, shuffle=False, random_state=18)
    data_labels = list(chain(*zip(data_labels,data_labels)))

    train_test_datadir = "{0}/models/datain/".format(crm_data)
    
    # fname = 'train_test_data_{0}_std.hdf5'.format(region)
    # # fname = 'train_test_data_{0}_noshuffle_std.hdf5'.format(region)
    # with h5py.File(train_test_datadir+fname, 'w') as hfile:
    #     i = 0
    #     while (i < len(data_std_split)):
    #         train_name = data_labels[i]+"_train"
    #         print("Saving normalised data {0}".format(train_name))
    #         train_data = data_std_split[i]
    #         if train_data.ndim == 1:
    #             train_data = train_data.reshape(-1,1)
    #         hfile.create_dataset(train_name,data=train_data)
    #         i+=1
    #         test_name = data_labels[i]+"_test"
    #         print("Saving normalised data {0}".format(test_name))
    #         test_data = data_std_split[i]
    #         if test_data.ndim == 1:
    #             test_data = test_data.reshape(-1,1)
    #         hfile.create_dataset(test_name,data=test_data)
    #         i+=1

    fname = 'train_test_data_{0}_raw.hdf5'.format(region)
    # fname = 'train_test_data_{0}_noshuffle_std.hdf5'.format(region)
    with h5py.File(train_test_datadir+fname, 'w') as hfile:
        i = 0
        while (i < len(data_std_split)):
            train_name = data_labels[i]+"_train"
            print("Saving normalised data {0}".format(train_name))
            train_data = raw_data_split[i]
            if train_data.ndim == 1:
                train_data = train_data.reshape(-1,1)
            hfile.create_dataset(train_name,data=train_data)
            i+=1
            test_name = data_labels[i]+"_test"
            print("Saving normalised data {0}".format(test_name))
            test_data = raw_data_split[i]
            if test_data.ndim == 1:
                test_data = test_data.reshape(-1,1)
            hfile.create_dataset(test_name,data=test_data)
            i+=1

if __name__ == "__main__":
    # combine_multi_level_files()
    # combine_surface_level_files()
    # combine_subdomains("9999NEWS")
    nn_dataset("9999NEWS")