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

crm_data = "/project/spice/radiation/ML/CRM/data"

def normalise_with_global_profile(stash: int, dataset: np.array([]), region: str):
    """
    Use global mean and standard deviation to normalise any regional data
    """
    normaliser_loc = "{0}/models/normaliser/{1}/".format(crm_data,region)
    qphys_normaliser_std = h5py.File('{0}/q_phys.hdf5'.format(normaliser_loc),'r')
    q_normaliser_std = h5py.File('{0}/q_tot.hdf5'.format(normaliser_loc),'r')
    qadv_normaliser_std = h5py.File('{0}/q_adv.hdf5'.format(normaliser_loc),'r')
    t_normaliser_std = h5py.File('{0}/air_potential_temperature.hdf5'.format(normaliser_loc),'r')
    tadv_normaliser_std = h5py.File('{0}/t_adv.hdf5'.format(normaliser_loc),'r')
    tphys_normaliser_std = h5py.File('{0}/t_phys.hdf5'.format(normaliser_loc),'r')
    shf_std = h5py.File('{0}/surface_upward_sensible_heat_flux.hdf5'.format(normaliser_loc),'r')
    lhf_std = h5py.File('{0}/surface_upward_latent_heat_flux.hdf5'.format(normaliser_loc),'r')
    swtoa_std = h5py.File('{0}/toa_incoming_shortwave_flux.hdf5'.format(normaliser_loc),'r')
    
    normaliser_stashes = {
    4:t_normaliser_std,
    99181:tadv_normaliser_std,
    99182:qadv_normaliser_std,
    99821:q_normaliser_std,
    1207:swtoa_std,
    3217:shf_std,
    3234:lhf_std,
    99904:tphys_normaliser_std,
    99983:qphys_normaliser_std}


    results = (dataset - normaliser_stashes[stash]['mean_'][:])/normaliser_stashes[stash]['scale_'][:]
    return results, dataset
    

def standardise_data_transform(dataset: np.array([]), region: str, save_fname: str="std_fit.hdf5", levs: bool=True, robust: bool=False):
    """
    Manually standardise data based instead of using sklearn standarad scaler
    robust: Use median and quantiles for scaling
    """
    save_location = "{0}/models/normaliser/{1}/".format(crm_data,region)
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
    return results, dataset

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
            if s in [99904, 99983]:
                var = data[stashes[s]][:]
            else:
                var = data[stashes[s]][:-1]
                
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

def nn_dataset_per_subdomain(region:str, in_prefix="031525", suite_id="u-br800", truncate: bool=True, global_profile: bool=False):
    """
    Create dataset for the neural network training and testing
    """   
    # NN input data
    

    if truncate:
        trunc_idx = truncation_idx(region, suite_id, in_prefix)

    for subdomain in range(64):
        data_labels = []
        data = []
        raw_data = []
        for s in nn_data_stashes:
            indir = "/project/spice/radiation/ML/CRM/data/{2}/{0}/concat_stash_{1}".format(region, str(s).zfill(5), suite_id)
            infile="{0}/{1}_days_{2}_km1p5_ra1m_30x30_subdomain_{4}_{3}.nc".format(indir, in_prefix, region, str(s).zfill(5), str(subdomain).zfill(3))
            print("Processing {0}".format(infile))
            dataf = Dataset(infile)
            if s == 99181:
                readname = "unknown"
                varname = "t_adv"
            elif s == 99182:
                readname = "unknown"
                varname = "q_adv"
            else:
                readname = nn_data_stashes[s]
                varname = nn_data_stashes[s]
            if truncate:
                var = dataf[readname][trunc_idx]
            else:
                if s in [99904, 99983]:
                    var = dataf[readname][:]
                else:
                    var = dataf[readname][:-1]
            
            std_fname=varname+".hdf5"
            if global_profile:
                global_norm="9999LEAU_std"
                print("Applying global mean and std for normalising from {0}".format(global_norm))
                normed_var, raw_var = normalise_with_global_profile(s, var, global_norm)
            else:
                normed_var, raw_var = standardise_data_transform(var, region, save_fname=std_fname)
            data_labels.append(varname)
            data.append(normed_var)
            raw_data.append(raw_var)
        
        data_std_split = train_test_split(*data, test_size=0.99, shuffle=False, random_state=18)
        # data_std_split = train_test_split(*data, shuffle=False, random_state=18)
        data_labels = list(chain(*zip(data_labels,data_labels)))

        train_test_datadir = "{0}/models/datain/".format(crm_data)
        
        fname = 'validation_data_{0}_{1}.hdf5'.format(region, str(subdomain).zfill(3))
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

def nn_dataset(region:str, in_prefix="031525", suite_id="u-br800", truncate: bool=True, global_profile: bool=False):
    """
    Create dataset for the neural network training and testing
    """   
    # NN input data
    data_labels = []
    data = []
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
        std_fname=nn_data_stashes[s]+".hdf5"
        if global_profile:
            global_norm="9999LEAU_std"
            print("Applying global mean and std for normalising from {0}".format(global_norm))
            normed_var, raw_var = normalise_with_global_profile(s, var, global_norm)
        else:
            normed_var, raw_var = standardise_data_transform(var, region, save_fname=std_fname)
        data_labels.append(nn_data_stashes[s])
        data.append(normed_var)
        raw_data.append(raw_var)
    
    data_std_split = train_test_split(*data, shuffle=False, random_state=18)
    # data_std_split = train_test_split(*data, shuffle=False, random_state=18)
    data_labels = list(chain(*zip(data_labels,data_labels)))

    train_test_datadir = "{0}/models/datain/".format(crm_data)
    
    fname = 'train_test_data_levs_{0}_std.hdf5'.format(region)
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

if __name__ == "__main__":
    region="60S140W"
    # combine_subdomains(region, in_prefix="030405", suite_id="u-br800")
    # nn_dataset(region, in_prefix="030405", suite_id="u-br800", truncate=False, global_profile=True)
    nn_dataset_per_subdomain(region, in_prefix="060708", suite_id="u-br800", truncate=False, global_profile=True)