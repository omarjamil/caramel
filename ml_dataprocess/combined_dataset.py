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
from scipy import stats

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
    # 12181:"change_over_time_in_air_temperature_due_to_advection",
    # 12182:"change_over_time_in_specific_humidity_due_to_advection",
    # 12183:"change_over_time_in_mass_fraction_of_cloud_liquid_water_in_air_due_to_advection",
    # 12184:"change_over_time_in_mass_fraction_of_cloud_ice_in_air_due_to_advection",
    # 12189:"change_over_time_in_mass_fraction_of_rain_in_air_due_to_advection",
    # 12190:"change_over_time_in_mass_fraction_of_graupel_in_air_due_to_advection",
    16004:"air_temperature",
    99181:"unknown",
    99182:"unknown",
    # 99181:"t_adv",
    # 99182:"q_adv",
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


# regions = ['50N144W','10S120W','10N120W','20S112W','0N90W','30N153W','80S90E','40S90W','10N80E','0N90E','10S80W','50S0E','70S0E','0N162W','30S102W','40N90E','70S120E','60N135W','70S120W','50N72E','40S30E','10N40W','20S22E','10N40E','30N105E','20N157E','40S150W','40S30W','30S153W','0N54E','50S144W','20S67E','60N45E','10S40W','60S45E','20S112E','20N67W','0N126W','0N126E','10N0E','20S157W','50N0E','20N118W','50S144E','30N51W','20N22W','40S90E','30S153E','30N0E','20N157W','40N150W','20N67E','0N162E','70N120E','30N51E','20S67W','20N22E','70N0E','40S150E','20S22W','80N90E','40N150E','70N120W','10N160W','50N72W','60S135W','60S39W','80N90W','60S135E','30S102E','10S0E','10N160E','30N102W','20N112E','10S160E','10S80E','60N135E','30S51E','30N153E','10N120E','0N18W','30S51W','10S120E','40N30E','40N90W','0N54W','30S0E','60N45W','20S157E','50S72E','0N18E','40N30W']

# Ocean only from u-bs572 and u-bs573 set aside for validation '0N100W'
# leave out 0N0E as that does not get read into iris properly
regions=['0N130W','0N15W','0N160E','0N160W','0N30W','0N50E','0N70E','0N88E','10N100W','10N120W','10N140W','10N145E','10N160E','10N170W','10N30W','10N50W','10N60E','10N88E','10S120W','10S140W','10S15W','10S170E','10S170W','10S30W','10S5E','10S60E','10S88E','10S90W','20N135E','20N145W','20N170E','20N170W','20N30W','20N55W','20N65E','20S0E','20S100W','20S105E','20S130W','20S160W','20S30W','20S55E','20S80E','21N115W','29N65W','30N130W','30N145E','30N150W','30N170E','30N170W','30N25W','30N45W','30S100W','30S10E','30S130W','30S15W','30S160W','30S40W','30S60E','30S88E','40N140W','40N150E','40N160W','40N170E','40N25W','40N45W','40N65W','40S0E','40S100E','40S100W','40S130W','40S160W','40S50E','40S50W','50N140W','50N149E','50N160W','50N170E','50N25W','50N45W','50S150E','50S150W','50S30E','50S30W','50S88E','50S90W','60N15W','60N35W','60S0E','60S140E','60S140W','60S70E','60S70W','70N0E','70S160W','70S40W','80N150W']

crm_data = "/project/spice/radiation/ML/CRM/data"
suite_id = "u-bs572_conc"



def standardise_data_transform(dataset: np.array([]), region: str, save_fname: str="std_fit.hdf5", levs: bool=True, robust: bool=False, return_raw=False):
    """
    Manually standardise data based instead of using sklearn standarad scaler
    robust: Use median and quantiles for scaling
    """
    save_location = "{0}/models/normaliser/{1}_scalar/".format(crm_data,region)
    # save_location = "{0}/models/normaliser/{1}_noshuffle/".format(crm_data,region)
    try:
        os.makedirs(save_location)
    except OSError:
        pass
    # per level normalisation
    if levs:
        if robust:
            mean = np.median(dataset, axis=0)
            q2 = np.quantile(dataset, 0.90, axis=0)
            q1 = np.quantile(dataset, 0.10, axis=0)
            scale = q2 - q1
        else:
            mean = np.array([np.mean(dataset, axis=0)])
            scale = np.array([np.std(dataset, axis=0)])
    else:
        # Mean across the entire dataset and levels
        if robust:
            mean = np.array([np.median(dataset)])
            q2 = np.array([np.quantile(dataset, 0.90)])
            q1 = np.array([np.quantile(dataset, 0.10)])
            scale = q2 - q1
        else:
            mean = np.array([np.mean(dataset)])
            scale = np.array([np.std(dataset)])
    params = {"mean_":mean, "scale_":scale}
    with h5py.File(save_location+save_fname, 'w') as hfile:
        for k, v in params.items():  
            hfile.create_dataset(k,data=v)
    results = (dataset - mean)/scale
    if return_raw:
        return results, dataset
    else:
        return results

def truncate_data(dataset: np.array([]), n_sigma: int=3):
    """
    Truncate data to n*sigma values
    """
    idx = (np.abs(stats.zscore(dataset,axis=0)) < n_sigma).all(axis=1)
    d = dataset[idx]
    return d

def truncation_idx(region: str, suite_id: str, in_prefix: str, n_sigma: int=3):
    """
    In order to truncate all the data with the same indices
    use qphys to work out 3 sigma truncation and use these indices
    """
    s = 99983
    indir = "/project/spice/radiation/ML/CRM/data/{2}/{0}/concat_stash_{1}".format(region, str(s).zfill(5), suite_id)
    infile="{0}/{1}_days_{2}_km1p5_ra1m_30x30_{3}.nc".format(indir, in_prefix, region, str(s).zfill(5))
    print("Calculating tuncation indices from {0}".format(infile))
    dataf = Dataset(infile)
    var = dataf[nn_data_stashes[s]][:]
    idx = (np.abs(stats.zscore(var,axis=0)) < n_sigma).all(axis=1)
    return idx

def truncation_idx_combined(suite_id: str, in_prefix: str, n_sigma: int=3):
    """
    In order to truncate all the data with the same indices
    use qphys to work out 3 sigma truncation and use these indices
    """
    s = 99983
    indir = "/project/spice/radiation/ML/CRM/data/{0}".format(suite_id)
    infile="{0}/{1}_km1p5_ra1m_30x30_{2}.nc".format(indir, in_prefix, str(s).zfill(5))
    print("Calculating tuncation indices from {0}".format(infile))
    dataf = Dataset(infile)
    var = dataf[nn_data_stashes[s]][:]
    idx = (np.abs(stats.zscore(var,axis=0)) < n_sigma).all(axis=1)
    return idx

def combine_multi_level_files(in_prefix="031525", suite_id="u-br800", new_region="9999LEAU"):
    """
    Combine all the regions into a single file
    """
    # new_region = "9999NEWS"
    out_basedir = "/project/spice/radiation/ML/CRM/data/{1}/{0}/".format(new_region, suite_id)
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
                in_dir = "/project/spice/radiation/ML/CRM/data/{2}/{0}/concat_stash_{1}/".format(r,str(s).zfill(5), suite_id)
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

def combine_surface_level_files(in_prefix="031525", suite_id="u-bj775_", new_region="9999LEAU"):
    """
    Combine all the regions into a single file
    """
    # new_region = "9999NEWS"
    out_basedir = "/project/spice/radiation/ML/CRM/data/{1}/{0}/".format(new_region,suite_id)
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
                in_dir = "/project/spice/radiation/ML/CRM/data/{2}/{0}/concat_stash_{1}/".format(r,str(s).zfill(5), suite_id)
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

def combine_subdomains(region: str, in_prefix="031525", suite_id="u-br800"):
    """
    After combining data from all the regions
    now combine data from all the subregions into a single dataset per stash 
    """
    stashes = {**surface_stashes, **multi_stashes}
    for s in stashes:
        inoutdir = "/project/spice/radiation/ML/CRM/data/{2}/{0}/concat_stash_{1}".format(region, str(s).zfill(5), suite_id)
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


def nn_dataset_raw(region:str, in_prefix="031525", suite_id="u-br800", truncate: bool=False):
    """
    Create dataset. This save raw data as well as normalised
    """   
    # NN input data
    data_labels = []
    raw_data = []
    if truncate:
        trunc_idx = truncation_idx(region, suite_id, in_prefix)

    for s in nn_data_stashes:
        indir = "/project/spice/radiation/ML/CRM/data/{2}/{0}/concat_stash_{1}".format(region, str(s).zfill(5), suite_id)
        infile="{0}/{1}_days_{2}_km1p5_ra1m_30x30_{3}.nc".format(indir, in_prefix, region, str(s).zfill(5))
        print("Processing {0}".format(infile))
        dataf = Dataset(infile)
        if truncate:
            var = dataf[nn_data_stashes[s]][trunc_idx]
        else:
            var = dataf[nn_data_stashes[s]][:]
        if s in [99181, 99182]:
            print("Multiplying advected quantity {0} with 600.".format(nn_data_stashes[s]))
            var *= 600.
        raw_var = var
        data_labels.append(nn_data_stashes[s])
        raw_data.append(raw_var)
    
    data_labels = list(chain(*zip(data_labels,data_labels)))
    train_test_datadir = "{0}/models/datain/".format(crm_data)

    raw_data_split = train_test_split(*raw_data, shuffle=True, random_state=18, test_size=0.1)
    fname = 'train_test_data_{0}_raw.hdf5'.format(region)
    # fname = 'train_test_data_{0}_noshuffle_std.hdf5'.format(region)
    with h5py.File(train_test_datadir+fname, 'w') as hfile:
        i = 0
        while (i < len(raw_data_split)):
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

def nn_dataset_std(region:str, in_prefix="031525", suite_id="u-br800", truncate: bool=False):
    """
    Create dataset for the neural network training and testing
    """   
    # NN input data
    data_labels = []
    data = []
    if truncate:
        trunc_idx = truncation_idx(region, suite_id, in_prefix)

    for s in nn_data_stashes:
        indir = "/project/spice/radiation/ML/CRM/data/{2}/{0}/concat_stash_{1}".format(region, str(s).zfill(5), suite_id)
        infile="{0}/{1}_days_{2}_km1p5_ra1m_30x30_{3}.nc".format(indir, in_prefix, region, str(s).zfill(5))
        print("Processing {0}".format(infile))
        dataf = Dataset(infile)
        if truncate:
            var = dataf[nn_data_stashes[s]][trunc_idx]
        else:
            var = dataf[nn_data_stashes[s]][:]
        std_fname=nn_data_stashes[s]+".hdf5"
        normed_var = standardise_data_transform(var, region, save_fname=std_fname, return_raw=False, levs=False)
        data_labels.append(nn_data_stashes[s])
        data.append(normed_var)
    
    data_std_split = train_test_split(*data, shuffle=True, test_size=0.1, random_state=18)
    # data_std_split = train_test_split(*data, shuffle=False, random_state=18)
    data_labels = list(chain(*zip(data_labels,data_labels)))

    train_test_datadir = "{0}/models/datain/".format(crm_data)
    
    fname = 'train_test_data_{0}_scalar_std.hdf5'.format(region)
    # fname = 'train_test_data_{0}_noshuffle_std.hdf5'.format(region)
    with h5py.File(train_test_datadir+fname, 'w') as hfile:
        i = 0
        while (i < len(data_std_split)):
            train_name = data_labels[i]+"_train"
            print("Saving normalised data {0}".format(train_name))
            train_data = data_std_split[i]
            if train_data.ndim == 1:
                train_data = train_data.reshape(-1,1)
            hfile.create_dataset(train_name,data=train_data)
            i+=1
            test_name = data_labels[i]+"_test"
            print("Saving normalised data {0}".format(test_name))
            test_data = data_std_split[i]
            if test_data.ndim == 1:
                test_data = test_data.reshape(-1,1)
            hfile.create_dataset(test_name,data=test_data)
            i+=1

def save_standardise_data_vars(dataset: np.array([]), region: str, save_fname: str="std_fit.hdf5", levs: bool=True, robust: bool=False):
    """
    Manually standardise data based instead of using sklearn standarad scaler
    robust: Use median and quantiles for scaling
    """
    save_location = "{0}/models/normaliser/{1}_standardise_mx/".format(crm_data,region)
    # save_location = "{0}/models/normaliser/{1}_noshuffle/".format(crm_data,region)
    try:
        os.makedirs(save_location)
    except OSError:
        pass
    # per level normalisation
    if levs:
        if robust:
            mean = np.median(dataset, axis=0)
            q2 = np.quantile(dataset, 0.90, axis=0)
            q1 = np.quantile(dataset, 0.10, axis=0)
            scale = q2 - q1
        else:
            mean = np.array([np.mean(dataset, axis=0)])
            minmaxrange = np.max(dataset, axis=0) - np.min(dataset, axis=0)
            std = np.std(dataset, axis=0)
            scale = np.array([np.maximum(minmaxrange,std)])
    else:
        # Mean across the entire dataset and levels
        if robust:
            mean = np.array([np.median(dataset)])
            q2 = np.array([np.quantile(dataset, 0.90)])
            q1 = np.array([np.quantile(dataset, 0.10)])
            scale = q2 - q1
        else:
            mean = np.array([np.mean(dataset)])
            scale = np.array([np.std(dataset)])
    params = {"mean_":mean, "scale_":scale}
    with h5py.File(save_location+save_fname, 'w') as hfile:
        for k, v in params.items():  
            hfile.create_dataset(k,data=v)

def save_normalise_data_vars(dataset: np.array([]), region: str, save_fname: str="std_fit.hdf5", levs: bool=True):
    """
    Manually standardise data based instead of using sklearn standarad scaler
    robust: Use median and quantiles for scaling
    """
    save_location = "{0}/models/normaliser/{1}_normalise/".format(crm_data,region)
    # save_location = "{0}/models/normaliser/{1}_noshuffle/".format(crm_data,region)
    try:
        os.makedirs(save_location)
    except OSError:
        pass
    # per level normalisation
    if levs:
        mean = np.array([np.min(dataset, axis=0)])
        scale = np.array([np.max(dataset, axis=0) - np.min(dataset, axis=0)])
    else:
        mean = np.array([np.min(dataset)])
        scale = np.array([np.max(dataset) - np.min(dataset)])
    params = {"mean_":mean, "scale_":scale}
    with h5py.File(save_location+save_fname, 'w') as hfile:
        for k, v in params.items():  
            hfile.create_dataset(k,data=v)

def nn_normalisation_vars(region:str, in_prefix="031525", suite_id="u-br800"):
    """
    Save data normalisation variables
    """   
    # NN input data

    for s in nn_data_stashes:
        indir = "/project/spice/radiation/ML/CRM/data/{2}/{0}/concat_stash_{1}".format(region, str(s).zfill(5), suite_id)
        infile="{0}/{1}_days_{2}_km1p5_ra1m_30x30_{3}.nc".format(indir, in_prefix, region, str(s).zfill(5))
        print("Processing {0}".format(infile))
        dataf = Dataset(infile)
        var = dataf[nn_data_stashes[s]][:]
        std_fname=nn_data_stashes[s]+".hdf5"
        if s in [99181,99182]:
            print("Multiplying advected quantities with 600.")
            var *= 600.
        save_standardise_data_vars(var, region, save_fname=std_fname, levs=True)
        # save_normalise_data_vars(var, region, save_fname=std_fname, levs=True)

if __name__ == "__main__":
    # Run the following in order 
    # u-bs572 has January runs so 021501AQ
    # u-bs573 has July runs so 0201507AQ

    # combine_multi_level_files(in_prefix="0203040506070809101112131415", suite_id="u-bs573_conc", new_region="021507AQ")
    #combine_surface_level_files(in_prefix="0203040506070809101112131415", suite_id="u-bs573_conc", new_region="021507AQ")
    # combine_subdomains("021501AQ", in_prefix="0203040506070809101112131415", suite_id="u-bs572_20170101-15_conc")
    # nn_dataset_raw("021501AQ", in_prefix="0203040506070809101112131415", suite_id="u-bs572_20170101-15_conc", truncate=False)
    # nn_dataset_std("021501AQ", in_prefix="0203040506070809101112131415", suite_id="u-bs572_conc", truncate=False)
    nn_normalisation_vars("021501AQ", in_prefix="0203040506070809101112131415", suite_id="u-bs572_20170101-15_conc")