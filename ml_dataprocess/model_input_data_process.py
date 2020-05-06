import numpy as np
from netCDF4 import Dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# import joblib
import h5py
import os
"""
Datafeed for ML model
"""
crm_data = "/project/spice/radiation/ML/CRM/data"
data_path = "{0}/u-bj775".format(crm_data)

def nooverlap_smooth(arrayin, window=6):
    """
    Moving average with non-overlapping window
    """
    x,y=arrayin.shape
    averaged = np.mean(arrayin.reshape(window,x//window,y,order='F'),axis=0)
    return averaged

def read_combined_tseries(filepath: str, region: str, var_name: str, var_stash: int):
    """
    Read the inputfile and concat all 64 subdomains into one long timeseries
    """
    var_data = None
    for subdomain in range(64):
        stashdir = "concat_stash_{0}".format(str(var_stash).zfill(5))
        datadir = "{0}/{1}".format(filepath,stashdir)
        inputfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_{3}.nc".format(datadir, region, str(subdomain).zfill(3), str(var_stash).zfill(5))
        print("Reading variable {2} subdomain {0} for {1}".format(str(subdomain),region, var_name))
        datafile = Dataset(inputfile)
        data = datafile.variables[var_name][:-1]
        if subdomain == 0:
            var_data = data
        else:
            var_data = np.concatenate((var_data,data),axis=0)

    return var_data


def read_udotgrad_tend(region: str):
    """
    Tendencies calculated via u.gradq and u.gradtheta
    """
    q_dir = "{0}/{1}/concat_stash_99182".format(data_path, region)
    t_dir = "{0}/{1}/concat_stash_99181".format(data_path, region)
    qphys_dir = "{0}/{1}/concat_stash_99983".format(data_path, region)
    tphys_dir = "{0}/{1}/concat_stash_99904".format(data_path, region)
    q_ = None
    t_ = None
    qphys_ = None

    for subdomain in range(64):
        print("Reading u.grad tend subdomain {0} for {1}".format(str(subdomain),region))
        qfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_99182.nc".format(q_dir, region, str(subdomain).zfill(3))
        tfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_99181.nc".format(t_dir, region, str(subdomain).zfill(3))
        qphysfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_99983.nc".format(qphys_dir, region, str(subdomain).zfill(3))
        tphysfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_99904.nc".format(tphys_dir, region, str(subdomain).zfill(3))
        qdata = Dataset(qfile)
        tdata = Dataset(tfile)
        qphysdata = Dataset(qphysfile)
        tphysdata = Dataset(tphysfile)
        
        q = qdata.variables['unknown'][:-1]
        t = tdata.variables['unknown'][:-1]
        qphys = qphysdata.variables['q_phys'][:]
        tphys = tphysdata.variables['t_phys'][:]
        if subdomain == 0:
            q_ = q
            t_ = t
            qphys_ = qphys
            tphys_ = tphys
        else:
            q_ = np.concatenate((q_,q),axis=0)
            t_ = np.concatenate((t_,t),axis=0)
            qphys_ = np.concatenate((qphys_,qphys),axis=0)
            tphys_ = np.concatenate((tphys_,tphys),axis=0)
    
    return q_,t_,qphys_,tphys_

def smoothed_vars(region:str):
    """
    Read and return variables that have been averaged
    """
    q_dir = "{0}/{1}/concat_stash_99821".format(data_path, region)
    t_dir = "{0}/{1}/concat_stash_00004".format(data_path, region)
    q_ = None
    t_ = None
    qnext_ = None
    qadv_dir = "{0}/{1}/concat_stash_99182".format(data_path, region)
    tadv_dir = "{0}/{1}/concat_stash_99181".format(data_path, region)
    qphys_dir = "{0}/{1}/concat_stash_99983".format(data_path, region)
    qadv_ = None
    tadv_ = None
    qphys_ = None

    for subdomain in range(64):
        print("Smoothed variables tend subdomain {0} for {1}".format(str(subdomain),region))
        qadvfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_99182.nc".format(qadv_dir, region, str(subdomain).zfill(3))
        tadvfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_99181.nc".format(tadv_dir, region, str(subdomain).zfill(3))
        qphysfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_99983.nc".format(qphys_dir, region, str(subdomain).zfill(3))
        qfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_99821.nc".format(q_dir, region, str(subdomain).zfill(3))
        tfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_00004.nc".format(t_dir, region, str(subdomain).zfill(3))
        qdata = Dataset(qfile)
        tdata = Dataset(tfile)
        qadvdata = Dataset(qadvfile)
        tadvdata = Dataset(tadvfile)
        qphysdata = Dataset(qphysfile)
        
        qadv = nooverlap_smooth(qadvdata.variables['unknown'][:], window=6)[:-1]
        tadv = nooverlap_smooth(tadvdata.variables['unknown'][:], window=6)[:-1]
        q = nooverlap_smooth(qdata.variables['q_tot'][:], window=6)[:-1]
        t = nooverlap_smooth(tdata.variables['air_potential_temperature'][:], window=6)[:-1]
        qnext = nooverlap_smooth(qdata.variables['q_tot'][:], window=6)[1:]
        qphys = qphysdata.variables['q_phys'][:]

        if subdomain == 0:
            qadv_ = qadv
            tadv_ = tadv
            qphys_ = qphys
            q_ = q
            t_ = t
            qnext_ = qnext
        else:
            qadv_ = np.concatenate((qadv_,qadv),axis=0)
            tadv_ = np.concatenate((tadv_,tadv),axis=0)
            qphys_ = np.concatenate((qphys_,qphys),axis=0)
            q_ = np.concatenate((q_,q),axis=0)
            t_ = np.concatenate((t_,t),axis=0)
            qnext_ = np.concatenate((qnext_,qnext),axis=0)
    
    return q_,qnext_,t_,qadv_,tadv_,qphys_

def read_combined_qT(region: str):
    """
    """
    q_dir = "{0}/{1}/concat_stash_99821".format(data_path, region)
    t_dir = "{0}/{1}/concat_stash_00004".format(data_path, region)
    q_ = None
    t_ = None
    qnext_ = None
    
    for subdomain in range(64):
        print("Reading qT subdomain {0} for {1}".format(str(subdomain),region))
        qfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_99821.nc".format(q_dir, region, str(subdomain).zfill(3))
        tfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_00004.nc".format(t_dir, region, str(subdomain).zfill(3))
        qdata = Dataset(qfile)
        tdata = Dataset(tfile)
        
        q = qdata.variables['q_tot'][:-1]
        t = tdata.variables['air_potential_temperature'][:-1]
        qnext = qdata.variables['q_tot'][1:]

        if subdomain == 0:
            q_ = q
            t_ = t
            qnext_ = qnext
        else:
            q_ = np.concatenate((q_,q),axis=0)
            t_ = np.concatenate((t_,t),axis=0)
            qnext_ = np.concatenate((qnext_,qnext),axis=0)
    return q_,t_,qnext_


def model_trainining_ios(region: str):
    """
    Reading ML training inputs and outputs

    Desired ML outputs:
     TOA (outgoing SW) (W/M2) => 01205 (1)
     TOA (outgoing LW) (W/M2) => 02205 (1)
     Total Surface SW (W/M2) => 01235 (1)
     Total Surface LW (W/M2) => 02207 (1)
    Precip:
     Large scale rainfall rate (KG/M2/S) => 04203 (1)
     Large scale snowfall rate (KG/M2/S) => 04204 (1)
    Tendencies:
     Qtot(phys) = Qv + QCL + QCF + Qrain + Qg (KG/KG) => (70)
     T(phys) (K) => 12181 (70)
    
    Vector size: 1+1+1+1+1+1+70+70 --> 146

    ML Inputs:
     TOA (incoming SW) (W/M2) => 01207 (1)
     Surface Latent Heat Flux (W/M2) => 03234 (1)
     Surface Sensible Heat Flux (W/M2) => 03217 (1)
     Pressure at Mean Sea Level (Pa) => 16222 (1)
     Qtot(adv) => Qv +  QCL + QCF + Qrain + Qgraupel (KG/KG) => 12182 (70) + 12183 (70) + 12184 (70) + 12189 (70) + 12190 (70) Or better to calculate from Qtotal and u,v,w winds => (70)
     Q_total => Qv +  QCL + QCF + Qrain + Qgraupel (KG/KG) => 00010 (70) + 00254 (70) + 00012 (70) + 00272 (70) + 00273 (70) => (70)
     T(adv) => 12181 (70)
     T => 16004 (70)
  
    Vector size: 1+1+1+1+70+70+70+70 --> 284
    """
    output_stashes = [1205, 2205, 1235, 2207, 4203, 4204]
    output_var_names = {1205:"toa_outgoing_shortwave_flux",2205:"toa_outgoing_longwave_flux",1235:"surface_downwelling_shortwave_flux_in_air",2207:"surface_downwelling_longwave_flux_in_air",4203:"stratiform_rainfall_flux",4204:"stratiform_snowfall_flux"}
    output_dict={}
    input_stashes = [1207, 3234, 3217, 16222]
    input_var_names = {1207:"toa_incoming_shortwave_flux", 3234:"surface_upward_latent_heat_flux", 3217:"surface_upward_sensible_heat_flux", 16222:"air_pressure_at_sea_level"}
    input_dict = {}
    datadir = "{0}/{1}".format(data_path, region)
    q_s,qnext_s, T_s,qadv_s,tadv_s, qphys_s = smoothed_vars(region)
    q,T,qnext = read_combined_qT(region)
    # qphys,qadv,tphys,tadv = read_combined_tendencies(region)
    qadv_dot, tadv_dot, qphys_dot, tphys_dot = read_udotgrad_tend(region)

    for inps in input_stashes:
        input_dict[input_var_names[inps]] = read_combined_tseries(datadir, region, input_var_names[inps], inps)

    for outs in output_stashes:
        output_dict[output_var_names[outs]] =  read_combined_tseries(datadir, region, output_var_names[outs], outs)
        
    return q,qnext,qphys_dot,T,tphys_dot,qadv_dot,tadv_dot, q_s,qnext_s, T_s,qadv_s,tadv_s, qphys_s, input_dict["toa_incoming_shortwave_flux"],input_dict["surface_upward_latent_heat_flux"],input_dict["surface_upward_sensible_heat_flux"],input_dict["air_pressure_at_sea_level"],output_dict["toa_outgoing_shortwave_flux"], output_dict["toa_outgoing_longwave_flux"], output_dict["surface_downwelling_shortwave_flux_in_air"], output_dict["surface_downwelling_longwave_flux_in_air"], output_dict["stratiform_rainfall_flux"], output_dict["stratiform_snowfall_flux"]


def standardise_data(dataset: np.array([]), save_fname: str="std_fit.joblib"):
    """
    """
    save_location="{0}/models/normaliser/".format(crm_data)
    scaler = preprocessing.StandardScaler()
    scaler_fit = scaler.fit(dataset)
    joblib.dump(scaler_fit,save_location+save_fname)
    return scaler_fit

def standardise_data_transform(dataset: np.array([]), region: str, save_fname: str="std_fit.joblib", levs: bool=True, robust: bool=False):
    """
    Manually standardise data based instead of using sklearn standarad scaler
    """
    save_location = "{0}/models/normaliser/{1}_levs/".format(crm_data,region)
    try:
        os.makedirs(save_location)
    except OSError:
        pass
    if levs:
        if robust:
            mean = np.median(dataset, axis=0)
            q2 = np.quantile(dataset, 0.66, axis=0)
            q1 = np.quantile(dataset, 0.33, axis=0)
            scale = q2 - q1
        else:
            mean = np.mean(dataset, axis=0)
            scale = np.std(dataset, axis=0)
    else:
        # Mean across the entire dataset and levels
        if robust:
            mean = np.array([np.median(dataset)])
            q2 = np.array([np.quantile(dataset, 0.66)])
            q1 = np.array([np.quantile(dataset, 0.33)])
            scale = q2 - q1
        else:
            mean = np.array([np.mean(dataset)])
            scale = np.array([np.std(dataset)])
    params = {"mean_":mean, "scale_":scale}
    with h5py.File(save_location+save_fname.replace('joblib','hdf5'), 'w') as hfile:
        for k, v in params.items():  
            hfile.create_dataset(k,data=v)
    results = (dataset - mean)/scale
    return results

def minmax_data(dataset: np.array([]), save_fname: str="minmax_fit.joblib", frange: tuple=(-1,1)):
    """
    """
    save_location="{0}/models/normaliser/".format(crm_data)
    scaler = preprocessing.MinMaxScaler(feature_range=frange)
    scaler_fit = scaler.fit(dataset)
    with h5py.File(save_location+save_fname.replace('joblib','hdf5'), 'w') as hfile:
        for k, v in scaler_fit.__dict__.items():  
            hfile.create_dataset(k,data=v)
    joblib.dump(scaler_fit,save_location+save_fname)
    return scaler_fit

def driving_data_all_std(region: str):
    """
    This is for getting normalised driving data including surface variables
    data array shape = (n_samples, n_features) => (n_tsteps, n_levels)
    """
    qtot,qtot_next,qphys_dot,T,tphys,qadv_dot, tadv_dot, qtot_s, qtot_next_s, T_s, qadv_dot_s, tadv_s, qphys_dot_s, sw_toa_down,latent_up,sensible_up, mslp, sw_toa_up, lw_toa_up, sw_down, lw_down, rain, snow = model_trainining_ios(region)
    qadd_dot = qtot + qadv_dot*600.
    tadd_dot = T + tadv_dot*600.
    sw_toa_down = sw_toa_down.reshape(-1,1)
    latent_up = latent_up.reshape(-1,1)
    sensible_up = sensible_up.reshape(-1,1)
    mslp = mslp.reshape(-1,1)
    sw_toa_up = sw_toa_up.reshape(-1,1)
    lw_toa_up = lw_toa_up.reshape(-1,1)
    sw_down = sw_down.reshape(-1,1)
    lw_down = lw_down.reshape(-1,1)
    rain = rain.reshape(-1,1)
    snow = snow.reshape(-1,1)
    # Smoothed vars
    # qtot_s, T_s, qadv_dot_s, tadv_s, qphys_dot_s
    qadd_dot_s = qtot_s + qadv_dot_s

    # train_test_datadir="data/models/datain/"
    # np.savez(train_test_datadir+"data_tot_raw_"+region, qtot=qtot,qphys_tot=qphys_tot,qadv=qadv,T=T,tphys=tphys,tadv=tadv,sw_toa_down=sw_toa_down,latent_up=latent_up,sensible_up=sensible_up, mslp=mslp, sw_toa_up=sw_toa_up, lw_toa_up=lw_toa_up, sw_down=sw_down, lw_down=lw_down, rain=rain, snow=snow)
    scale_qtot = standardise_data_transform(qtot, region, save_fname="std_qtot.joblib")
    scale_qtot_next = standardise_data_transform(qtot_next, region, save_fname="std_qtot_next.joblib")
    scale_qtot_s = standardise_data_transform(qtot_s, region, save_fname="std_qtot_s.joblib")
    scale_qtot_next_s = standardise_data_transform(qtot_next_s, region, save_fname="std_qtot_next_s.joblib")
    scale_qadv_dot = standardise_data_transform(qadv_dot, region, save_fname="std_qadv_dot.joblib")
    scale_qadv_dot_s = standardise_data_transform(qadv_dot_s, region, save_fname="std_qadv_dot_s.joblib")
    scale_qphys_dot = standardise_data_transform(qphys_dot, region, save_fname="std_qphysdot.joblib")
    scale_qphys_dot_s = standardise_data_transform(qphys_dot_s, region, save_fname="std_qphysdot_s.joblib")
    scale_qadd_dot = standardise_data_transform(qadd_dot, region, save_fname="std_qadd_dot.joblib")
    scale_qadd_dot_s = standardise_data_transform(qadd_dot_s, region, save_fname="std_qadd_dot_s.joblib")
    scale_tadd_dot = standardise_data_transform(qadd_dot, region, save_fname="std_tadd_dot.joblib")
    scale_T = standardise_data_transform(T, region, save_fname="std_T.joblib")
    scale_tphys = standardise_data_transform(tphys, region, save_fname="std_tphys.joblib")
    scale_tadv_dot = standardise_data_transform(tadv_dot, region, save_fname="std_tadv_dot.joblib")
    scale_sw_toa_down = standardise_data_transform(sw_toa_down, region, save_fname="std_sw_toa_down.joblib")
    scale_latent_up = standardise_data_transform(latent_up, region, save_fname="std_latent_up.joblib")
    scale_sensible_up = standardise_data_transform(sensible_up, region, save_fname="std_sensible_up.joblib")
    scale_mslp = standardise_data_transform(mslp, region, save_fname="std_mslp.joblib")
    scale_sw_toa_up = standardise_data_transform(sw_toa_up, region, save_fname="std_sw_toa_up.joblib")
    scale_lw_toa_up = standardise_data_transform(lw_toa_up, region, save_fname="std_lw_toa_up.joblib")
    scale_sw_down = standardise_data_transform(sw_down, region, save_fname="std_sw_down.joblib")
    scale_lw_down = standardise_data_transform(lw_down, region, save_fname="std_lw_down.joblib")
    scale_rain = standardise_data_transform(rain, region, save_fname="std_rain.joblib")
    scale_snow = standardise_data_transform(snow, region, save_fname="std_snow.joblib")

    std_qtot, std_qtot_test = train_test_split(scale_qtot, shuffle=False)
    std_qtot_next, std_qtot_next_test = train_test_split(scale_qtot_next, shuffle=False)
    std_qtot_s, std_qtot_test_s = train_test_split(scale_qtot_s, shuffle=False)
    std_qtot_next_s, std_qtot_next_test_s = train_test_split(scale_qtot_next_s, shuffle=False)
    std_qadv_dot, std_qadv_dot_test = train_test_split(scale_qadv_dot, shuffle=False)
    std_qadv_dot_s, std_qadv_dot_test_s = train_test_split(scale_qadv_dot_s, shuffle=False)
    std_qphys_dot, std_qphys_dot_test = train_test_split(scale_qphys_dot, shuffle=False)
    std_qphys_dot_s, std_qphys_dot_test_s = train_test_split(scale_qphys_dot_s, shuffle=False)
    std_qadd_dot, std_qadd_dot_test = train_test_split(scale_qadd_dot, shuffle=False)
    std_qadd_dot_s, std_qadd_dot_test_s = train_test_split(scale_qadd_dot_s, shuffle=False)
    std_tadd_dot, std_tadd_dot_test = train_test_split(scale_tadd_dot, shuffle=False)
    std_T, std_T_test = train_test_split(scale_T, shuffle=False)
    std_tphys, std_tphys_test = train_test_split(scale_tphys, shuffle=False)
    std_tadv_dot, std_tadv_dot_test = train_test_split(scale_tadv_dot, shuffle=False)
    std_sw_toa_down, std_sw_toa_down_test = train_test_split(scale_sw_toa_down, shuffle=False)
    std_latent_up, std_latent_up_test = train_test_split(scale_latent_up, shuffle=False)
    std_sensible_up, std_sensible_up_test = train_test_split(scale_sensible_up, shuffle=False)
    std_mslp, std_mslp_test = train_test_split(scale_mslp, shuffle=False)
    std_sw_toa_up, std_sw_toa_up_test = train_test_split(scale_sw_toa_up, shuffle=False)
    std_lw_toa_up, std_lw_toa_up_test = train_test_split(scale_lw_toa_up, shuffle=False)
    std_sw_down, std_sw_down_test = train_test_split(scale_sw_down, shuffle=False)
    std_lw_down, std_lw_down_test = train_test_split(scale_lw_down, shuffle=False)
    std_rain, std_rain_test = train_test_split(scale_rain, shuffle=False)
    std_snow, std_snow_test = train_test_split(scale_snow, shuffle=False)

    qtot_raw, qtot_test_raw = train_test_split(qtot, shuffle=False)
    qtot_raw_s, qtot_test_raw_s = train_test_split(qtot_s, shuffle=False)
    qadv_dot_raw, qadv_dot_test_raw = train_test_split(qadv_dot, shuffle=False)
    qadv_dot_raw_s, qadv_dot_test_raw_s = train_test_split(qadv_dot_s, shuffle=False)
    qphys_dot_raw, qphys_dot_test_raw = train_test_split(qphys_dot, shuffle=False)
    qphys_dot_raw_s, qphys_dot_test_raw_s = train_test_split(qphys_dot_s, shuffle=False)
    qadd_dot_raw, qadd_dot_test_raw = train_test_split(qadd_dot, shuffle=False)
    qadd_dot_raw_s, qadd_dot_test_raw_s = train_test_split(qadd_dot_s, shuffle=False)
    tadd_dot_raw, tadd_dot_test_raw = train_test_split(tadd_dot, shuffle=False)
    T_raw, T_test_raw = train_test_split(T, shuffle=False)
    tphys_raw, tphys_test_raw = train_test_split(tphys, shuffle=False)
    tadv_dot_raw, tadv_dot_test_raw = train_test_split(tadv_dot, shuffle=False)
    sw_toa_down_raw, sw_toa_down_test_raw = train_test_split(sw_toa_down, shuffle=False)
    latent_up_raw, latent_up_test_raw = train_test_split(latent_up, shuffle=False)
    sensible_up_raw, sensible_up_test_raw = train_test_split(sensible_up, shuffle=False)
    mslp_raw, mslp_test_raw = train_test_split(mslp, shuffle=False)
    sw_toa_up_raw, sw_toa_up_test_raw = train_test_split(sw_toa_up, shuffle=False)
    lw_toa_up_raw, lw_toa_up_test_raw = train_test_split(lw_toa_up, shuffle=False)
    sw_down_raw, sw_down_test_raw = train_test_split(sw_down, shuffle=False)
    lw_down_raw, lw_down_test_raw = train_test_split(lw_down, shuffle=False)
    rain_raw, rain_test_raw = train_test_split(rain, shuffle=False)
    snow_raw, snow_test_raw = train_test_split(snow, shuffle=False)


    #return qtot_raw, qphys_tot_raw, qadv_raw, qadv_dot_raw, qadd_raw, qadd_dot_raw, tadd_raw, tadd_dot_raw, T_raw, tphys_raw, tadv_raw, tadv_dot_raw, sw_toa_down_raw, latent_up_raw, sensible_up_raw, mslp_raw, sw_toa_up_raw, lw_toa_up_raw, sw_down_raw, lw_down_raw, rain_raw, snow_raw, qtot_test_raw, qphys_tot_test_raw, qadv_test_raw, qadv_dot_test_raw, qadd_test_raw, qadd_dot_test_raw, tadd_test_raw, tadd_dot_test_raw, T_test_raw, tphys_test_raw, tadv_test_raw, tadv_dot_test_raw, sw_toa_down_test_raw, latent_up_test_raw, sensible_up_test_raw, mslp_test_raw, sw_toa_up_test_raw, lw_toa_up_test_raw, sw_down_test_raw, lw_down_test_raw, rain_test_raw, snow_test_raw, std_qtot, std_qphys_tot, std_qadv, std_qadv_dot, std_qadd, std_qadd_dot, std_tadd, std_tadd_dot, std_T, std_tphys, std_tadv, std_tadv_dot, std_sw_toa_down,std_latent_up,std_sensible_up, std_mslp, std_sw_toa_up, std_lw_toa_up, std_sw_down, std_lw_down, std_rain, std_snow, std_qtot_test, std_qphys_tot_test, std_qadv_test, std_qadv_dot_test, std_qadd_test, std_qadd_dot_test, std_tadd_test, std_tadd_dot_test, std_T_test,std_tphys_test,std_tadv_test, std_tadv_dot_test, std_sw_toa_down_test,std_latent_up_test,std_sensible_up_test, std_mslp_test, std_sw_toa_up_test, std_lw_toa_up_test, std_sw_down_test, std_lw_down_test, std_rain_test, std_snow_test
    # variables = {"qtot":std_qtot,"qtot_next":std_qtot_next, "qtot_s":std_qtot_s,"qtot_next_s":std_qtot_next_s, "qphys_dot":std_qphys_dot, "qphys_dot_s":std_qphys_dot_s, "qadv_dot":std_qadv_dot, "qadv_dot_s":std_qadv_dot_s,  "qadd_dot":std_qadd_dot, "qadd_dot_s":std_qadd_dot_s, "tadd_dot":std_tadd_dot, "T":std_T, "tphys":std_tphys,  "tadv_dot":std_tadv_dot, "sw_toa_down":std_sw_toa_down, "latent_up":std_latent_up, "sensible_up":std_sensible_up, "mslp":std_mslp, "sw_toa_up":std_sw_toa_up, "lw_toa_up":std_lw_toa_up, "sw_down":std_sw_down, "lw_down":std_lw_down, "rain":std_rain, "snow":std_snow, "qtot_test":std_qtot_test, "qtot_next_test":std_qtot_next_test, "qtot_test_s":std_qtot_test_s, "qtot_next_test_s":std_qtot_next_test_s,  "qphys_dot_test":std_qphys_dot_test, "qphys_dot_test_s":std_qphys_dot_test_s,  "qadv_dot_test":std_qadv_dot_test, "qadv_dot_test_s":std_qadv_dot_test_s,  "qadd_dot_test":std_qadd_dot_test, "qadd_dot_test_s":std_qadd_dot_test_s, "tadd_dot_test":std_tadd_dot_test, "T_test":std_T_test, "tphys_test":std_tphys_test,  "tadv_dot_test":std_tadv_dot_test, "sw_toa_down_test":std_sw_toa_down_test, "latent_up_test":std_latent_up_test, "sensible_up_test":std_sensible_up_test, "mslp_test":std_mslp_test, "sw_toa_up_test":std_sw_toa_up_test, "lw_toa_up_test":std_lw_toa_up_test, "sw_down_test":std_sw_down_test, "lw_down_test":std_lw_down_test, "rain_test":std_rain_test, "snow_test":std_snow_test, "qtot_raw":qtot_raw, "qtot_raw_s":qtot_raw_s,  "qphys_dot_raw":qphys_dot_raw, "qphys_dot_raw_s":qphys_dot_raw_s,  "qadv_dot_raw":qadv_dot_raw, "qadv_dot_raw_s":qadv_dot_raw_s,  "qadd_dot_raw":qadd_dot_raw, "qadd_dot_raw_s":qadd_dot_raw_s,  "tadd_dot_raw":tadd_dot_raw, "T_raw":T_raw, "tphys_raw":tphys_raw,  "tadv_dot_raw":tadv_dot_raw, "sw_toa_down_raw":sw_toa_down_raw, "latent_up_raw":latent_up_raw, "sensible_up_raw":sensible_up_raw, "mslp_raw":mslp_raw, "sw_toa_up_raw":sw_toa_up_raw, "lw_toa_up_raw":lw_toa_up_raw, "sw_down_raw":sw_down_raw, "lw_down_raw":lw_down_raw, "rain_raw":rain_raw, "snow_raw":snow_raw, "qtot_test_raw":qtot_test_raw, "qtot_test_raw_s":qtot_test_raw_s, "qphys_dot_test_raw":qphys_dot_test_raw, "qphys_dot_test_raw_s":qphys_dot_test_raw_s,  "qadv_dot_test_raw":qadv_dot_test_raw, "qadv_dot_test_raw_s":qadv_dot_test_raw_s,  "qadd_dot_test_raw":qadd_dot_test_raw, "qadd_dot_test_raw_s":qadd_dot_test_raw_s,  "tadd_dot_test_raw":tadd_dot_test_raw, "T_test_raw":T_test_raw, "tphys_test_raw":tphys_test_raw,  "tadv_dot_test_raw":tadv_dot_test_raw, "sw_toa_down_test_raw":sw_toa_down_test_raw, "latent_up_test_raw":latent_up_test_raw, "sensible_up_test_raw":sensible_up_test_raw, "mslp_test_raw":mslp_test_raw, "sw_toa_up_test_raw":sw_toa_up_test_raw, "lw_toa_up_test_raw":lw_toa_up_test_raw, "sw_down_test_raw":sw_down_test_raw, "lw_down_test_raw":lw_down_test_raw, "rain_test_raw":rain_test_raw, "snow_test_raw":snow_test_raw}

    variables = {"q_tot_train":std_qtot,"q_adv_train":std_qadv_dot, "air_potential_temperature_train":std_T, "t_adv_train":std_tadv_dot, "toa_incoming_shortwave_flux_train":std_sw_toa_down, "surface_upward_sensible_heat_flux_train":std_sensible_up, "surface_upward_latent_heat_flux_train":std_latent_up, "t_phys_train":std_tphys, "q_phys_train":std_qphys_dot, "q_tot_test":std_qtot_test,"q_adv_test":std_qadv_dot_test, "air_potential_temperature_test":std_T_test, "t_adv_test":std_tadv_dot_test, "toa_incoming_shortwave_flux_test":std_sw_down_test, "surface_upward_sensible_heat_flux_test":std_sensible_up_test, "surface_upward_latent_heat_flux_test":std_latent_up_test, "t_phys_test":std_tphys_test, "q_phys_test":std_qphys_dot_test}

    return variables

def driving_data_stats(region: str):
    """
    Inspect driving data mean, max, min etc.
    """
    train_test_datadir="{0}/models/datain/".format(crm_data)
    dataset = np.load(train_test_datadir+"data_tot_raw_"+region+".npz")
    variables = ["qtot","qphys_tot","qadv","T","tphys","tadv","sw_toa_down","latent_up","sensible_up","mslp","sw_toa_up", "lw_toa_up","sw_down","lw_down","rain","snow"]
    for v in variables:
        data = dataset[v]
        data_min = np.min(data)
        data_max = np.max(data)
        data_sig = np.std(data,axis=0)
        print("{0}\nmin: {1} \nmax: {2}\nmax-min: {4}\nstd: {3}".format(v,data_min,data_max,data_sig,data_max-data_min))
        print("")
    
def train_test_data_save_all(region: str):

    # variables = driving_data_all_minmax(region)
    variables = driving_data_all_std(region)

    train_test_datadir = "{0}/models/datain/".format(crm_data)
    # Save in hdf5 format
    filename='train_test_data_all_lev_{0}_std.hdf5'.format(region)
    with h5py.File(train_test_datadir+filename, 'w') as hfile:
        for k, v in variables.items():  
            hfile.create_dataset(k,data=v)

    
   
if __name__ == "__main__":
    region = "50S69W"
    train_test_data_save_all(region)
    
    # qphys,qadv,tphys,tadv = read_tendencies(region)
    # std_qadv = standardise_data(qadv,save_fname="std_qadv.joblib")
    # std_qphys = standardise_data(qphys,save_fname="std_qphys.joblib")
    # std_tadv = standardise_data(tadv,save_fname="std_tadv.joblib")
    # std_tphys = standardise_data(tphys,save_fname="std_tphys.joblib")
    # qadv_norm = std_qadv.transform(qadv)
    # qphys_norm = std_qphys.transform(qphys)
    # tadv_norm = std_tadv.transform(tadv)
    # tphys_norm = std_tphys.transform(tphys)
    # print(qadv_norm.shape, qphys_norm.shape)
    # qadv_train, qadv_test, qphys_train, qphys_test,  tadv_train, tadv_test, tphys_train, tphys_test = train_test_split(qadv_norm, qphys_norm, tadv_norm, tphys_norm, shuffle=False)
    # print(qadv_train.shape, qadv_test.shape, qphys_train.shape, qphys_test.shape)
    # print(tadv_train.shape, tadv_test.shape, tphys_train.shape, tphys_test.shape)
    
    
