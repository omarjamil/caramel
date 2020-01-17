import numpy as np
from netCDF4 import Dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import joblib
import h5py

"""
Datafeed for ML model
"""
crm_data = "/project/spice/radiation/ML/CRM/data"
data_path = "{0}/u-bj775".format(crm_data)

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

def read_combined_tendencies(region: str):
    """
    Read physics tendencies for q and T
    """
    qphys_dir = "{0}/{1}/tend_qphys_tot".format(data_path, region)
    qadv_dir = "{0}/{1}/tend_qadv_tot".format(data_path, region)
    tphys_dir = "{0}/{1}/tend_t_16004".format(data_path, region)
    tadv_dir = "{0}/{1}/concat_stash_12181".format(data_path, region)

    qphys_ = None
    qadv_ = None
    tphys_ = None
    tadv_ = None
        
    for subdomain in range(64):
        print("Reading tend_comb for subdomain {0} for {1}".format(str(subdomain),region))
        qphysfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_qphys_tot.nc".format(qphys_dir, region, str(subdomain).zfill(3))
        qadvfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_qadv_tot.nc".format(qadv_dir, region, str(subdomain).zfill(3))
        tphysfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_t_phys_16004.nc".format(tphys_dir, region, str(subdomain).zfill(3))
        tadvfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_12181.nc".format(tadv_dir, region, str(subdomain).zfill(3))
        
        qphysdata = Dataset(qphysfile)
        qadvdata = Dataset(qadvfile)
        tphysdata = Dataset(tphysfile)
        tadvdata = Dataset(tadvfile)
                
        qphys = qphysdata.variables['qphys_tot'][:]
        qadv = qadvdata.variables['qadv_tot'][:-1]
        tphys = tphysdata.variables['t_phys'][:]
        tadv = tadvdata.variables['change_over_time_in_air_temperature_due_to_advection'][:-1]
        
        if subdomain == 0:
            qphys_ = qphys
            qadv_ = qadv
            tphys_ = tphys
            tadv_ = tadv
        else:
            qphys_ = np.concatenate((qphys_,qphys),axis=0)
            qadv_ = np.concatenate((qadv_,qadv),axis=0)
            tphys_ = np.concatenate((tphys_,tphys),axis=0)
            tadv_ = np.concatenate((tadv_,tadv),axis=0)
            
    return qphys_,qadv_,tphys_,tadv_

def read_udotgrad_tend(region: str):
    """
    Tendencies calculated via u.gradq and u.gradtheta
    """
    q_dir = "{0}/{1}/concat_stash_99182".format(data_path, region)
    t_dir = "{0}/{1}/concat_stash_99181".format(data_path, region)
    q_ = None
    t_ = None
    
    for subdomain in range(64):
        print("Reading u.grad tend subdomain {0} for {1}".format(str(subdomain),region))
        qfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_99182.nc".format(q_dir, region, str(subdomain).zfill(3))
        tfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_99181.nc".format(t_dir, region, str(subdomain).zfill(3))
        qdata = Dataset(qfile)
        tdata = Dataset(tfile)
        
        q = qdata.variables['unknown'][:-1]
        t = tdata.variables['unknown'][:-1]
        
        if subdomain == 0:
            q_ = q
            t_ = t
        else:
            q_ = np.concatenate((q_,q),axis=0)
            t_ = np.concatenate((t_,t),axis=0)

    return q_,t_

def read_combined_qT(region: str):
    """
    """
    q_dir = "{0}/{1}/q_tot".format(data_path, region)
    t_dir = "{0}/{1}/concat_stash_16004".format(data_path, region)
    q_ = None
    t_ = None
    qnext_ = None
    
    for subdomain in range(64):
        print("Reading qT subdomain {0} for {1}".format(str(subdomain),region))
        qfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_q_tot.nc".format(q_dir, region, str(subdomain).zfill(3))
        tfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_16004.nc".format(t_dir, region, str(subdomain).zfill(3))
        qdata = Dataset(qfile)
        tdata = Dataset(tfile)
        
        q = qdata.variables['q_tot'][:-1]
        t = tdata.variables['air_temperature'][:-1]
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

    q,T,qnext = read_combined_qT(region)
    qphys,qadv,tphys,tadv = read_combined_tendencies(region)
    qadv_dot, tadv_dot = read_udotgrad_tend(region)

    for inps in input_stashes:
        input_dict[input_var_names[inps]] = read_combined_tseries(datadir, region, input_var_names[inps], inps)

    for outs in output_stashes:
        output_dict[output_var_names[outs]] =  read_combined_tseries(datadir, region, output_var_names[outs], outs)
        
    return q,qnext,qphys,qadv,T,tphys,tadv,qadv_dot,tadv_dot, input_dict["toa_incoming_shortwave_flux"],input_dict["surface_upward_latent_heat_flux"],input_dict["surface_upward_sensible_heat_flux"],input_dict["air_pressure_at_sea_level"],output_dict["toa_outgoing_shortwave_flux"], output_dict["toa_outgoing_longwave_flux"], output_dict["surface_downwelling_shortwave_flux_in_air"], output_dict["surface_downwelling_longwave_flux_in_air"], output_dict["stratiform_rainfall_flux"], output_dict["stratiform_snowfall_flux"]

def read_tendencies_combined(region: str):
    """
    Read the combined q tendencies
    """
    qphys_dir = "{0}/{1}/tend_qphys_tot".format(data_path, region)
    qadv_dir = "{0}/{1}/tend_qadv_tot".format(data_path, region)
    q_dir = "{0}/{1}/q_tot".format(data_path, region)
    tphys_dir = "{0}/{1}/tend_t_16004".format(data_path, region)
    tadv_dir = "{0}/{1}/concat_stash_12181".format(data_path, region)
    t_dir = "{0}/{1}/concat_stash_16004".format(data_path, region)

    qphys_ = None
    qadv_ = None
    q_ = None
    tphys_ = None
    tadv_ = None
    t_ = None
    
    for subdomain in range(64):
        print("Reading subdomain {0} for {1}".format(str(subdomain),region))
        qphysfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_qphys_tot.nc".format(qphys_dir, region, str(subdomain).zfill(3))
        qadvfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_qadv_tot.nc".format(qadv_dir, region, str(subdomain).zfill(3))
        qfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_q_tot.nc".format(q_dir, region, str(subdomain).zfill(3))
        tphysfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_t_phys_16004.nc".format(tphys_dir, region, str(subdomain).zfill(3))
        tadvfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_12181.nc".format(tadv_dir, region, str(subdomain).zfill(3))
        tfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_16004.nc".format(t_dir, region, str(subdomain).zfill(3))

        qphysdata = Dataset(qphysfile)
        qadvdata = Dataset(qadvfile)
        qdata = Dataset(qfile)
        tphysdata = Dataset(tphysfile)
        tadvdata = Dataset(tadvfile)
        tdata = Dataset(tfile)
        
        qphys = qphysdata.variables['qphys_tot'][:]
        qadv = qadvdata.variables['qadv_tot'][1:]
        q = qdata.variables['q_tot'][1:]
        tphys = tphysdata.variables['t_phys'][:]
        tadv = tadvdata.variables['change_over_time_in_air_temperature_due_to_advection'][1:]
        t = tdata.variables['air_temperature'][1:]
        
        if subdomain == 0:
            qphys_ = qphys
            qadv_ = qadv
            q_ = q
            tphys_ = tphys
            tadv_ = tadv
            t_ = t
        else:
            qphys_ = np.concatenate((qphys_,qphys),axis=0)
            qadv_ = np.concatenate((qadv_,qadv),axis=0)
            q_ = np.concatenate((q_,q),axis=0)
            tphys_ = np.concatenate((tphys_,tphys),axis=0)
            tadv_ = np.concatenate((tadv_,tadv),axis=0)
            t_ = np.concatenate((t_,t),axis=0)

    return qphys_,qadv_,q_,tphys_,tadv_,t_

def read_tendencies(region: str):
    """
    The tendencies are per subdomain
    """
    qphys_dir = "{0}/{1}/tend_q_00010/".format(data_path, region)
    qadv_dir = "{0}/{1}/concat_stash_12182/".format(data_path, region)
    q_dir = "{0}/{1}/concat_stash_00010/".format(data_path, region)
    tphys_dir = "{0}/{1}/tend_t_16004/".format(data_path, region)
    tadv_dir = "{0}/{1}/concat_stash_12181/".format(data_path, region)
    t_dir = "{0}/{1}/concat_stash_16004/".format(data_path, region)
    
    qphys_ = None
    qadv_ = None
    q_ = None
    tphys_ = None
    tadv_ = None
    t_ = None
    
    for subdomain in range(64):
        print("Reading subdomain {0} for {1}".format(str(subdomain),region))
        qphysfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_q_phys_00010.nc".format(qphys_dir, region, str(subdomain).zfill(3))
        qadvfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_12182.nc".format(qadv_dir, region, str(subdomain).zfill(3))
        qfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_00010.nc".format(q_dir, region, str(subdomain).zfill(3))
        tphysfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_t_phys_16004.nc".format(tphys_dir, region, str(subdomain).zfill(3))
        tadvfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_12181.nc".format(tadv_dir, region, str(subdomain).zfill(3))
        tfile = "{0}/30_days_{1}_km1p5_ra1m_30x30_subdomain_{2}_16004.nc".format(t_dir, region, str(subdomain).zfill(3))

        qphysdata = Dataset(qphysfile)
        qadvdata = Dataset(qadvfile)
        qdata = Dataset(qfile)
        tphysdata = Dataset(tphysfile)
        tadvdata = Dataset(tadvfile)
        tdata = Dataset(tfile)
        
        qphys = qphysdata.variables['q_phys'][:]
        qadv = qadvdata.variables['change_over_time_in_specific_humidity_due_to_advection'][1:]
        q = qdata.variables['specific_humidity'][1:]
        tphys = tphysdata.variables['t_phys'][:]
        tadv = tadvdata.variables['change_over_time_in_air_temperature_due_to_advection'][1:]
        t = tdata.variables['air_temperature'][1:]
        
        if subdomain == 0:
            qphys_ = qphys
            qadv_ = qadv
            q_ = q
            tphys_ = tphys
            tadv_ = tadv
            t_ = t
        else:
            qphys_ = np.concatenate((qphys_,qphys),axis=0)
            qadv_ = np.concatenate((qadv_,qadv),axis=0)
            q_ = np.concatenate((q_,q),axis=0)
            tphys_ = np.concatenate((tphys_,tphys),axis=0)
            tadv_ = np.concatenate((tadv_,tadv),axis=0)
            t_ = np.concatenate((t_,t),axis=0)

    return qphys_,qadv_,q_,tphys_,tadv_,

def standardise_data(dataset: np.array([]), save_fname: str="std_fit.joblib"):
    """
    """
    save_location="{0}/models/normaliser/".format(crm_data)
    scaler = preprocessing.StandardScaler()
    scaler_fit = scaler.fit(dataset)
    joblib.dump(scaler_fit,save_location+save_fname)
    return scaler_fit

def standardise_data_transform(dataset: np.array([]), save_fname: str="std_fit.joblib"):
    """
    Manually standardise data based instead of using sklearn standarad scaler
    """
    save_location = "{0}/models/normaliser/".format(crm_data)
    mean = np.array([np.mean(dataset)])
    scale = np.array([np.std(dataset)])
    params = {"mean_":mean, "scale_":scale}
    with h5py.File(save_location+save_fname.replace('joblib','hdf5'), 'w') as hfile:
        for k, v in params.items():  
            hfile.create_dataset(k,data=v)
    
    return (dataset - mean)/scale

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

def driving_data(region: str):
    """
    Data for driving the ML model
    data array shape = (n_samples, n_features) => (n_tsteps, n_levels)
    """
    # region = "50S69W"
    # qphys,qadv,q,tphys,tadv,t = read_tendencies_combined(region)
    qphys,qadv,q,tphys,tadv,t = read_tendencies(region)

    std_qadv = standardise_data(qadv,save_fname="std_qadv_comb.joblib")
    std_q = standardise_data(q,save_fname="std_q.joblib")
    #std_qphys = standardise_data(qphys,save_fname="std_qphys.joblib")
    std_qphys = minmax_data(qphys,save_fname="minmax_qphys_comb.joblib")
    #std_tadv = standardise_data(tadv,save_fname="std_tadv.joblib")
    std_tadv = minmax_data(tadv,save_fname="minmax_tadv.joblib")
    #std_tphys = standardise_data(tphys,save_fname="std_tphys.joblib")
    std_tphys = minmax_data(tphys, save_fname="minmax_tphys.joblib")
    std_t = standardise_data(t,save_fname="std_t.joblib")
    q_norm = std_q.transform(q)
    qadv_norm = std_qadv.transform(qadv)
    qphys_norm = std_qphys.transform(qphys)
    t_norm = std_t.transform(t)
    tadv_norm = std_tadv.transform(tadv)
    tphys_norm = std_tphys.transform(tphys)

    # q_train, q_test, qadv_train, qadv_test, qphys_train, qphys_test,  t_train, t_test, tadv_train, tadv_test, tphys_train, tphys_test
    a = train_test_split(q, q_norm, qadv, qadv_norm, qphys, qphys_norm, t, t_norm, tadv, tadv_norm, tphys, tphys_norm, shuffle=False)
    return a

def driving_data_raw(region: str):
    qphys,qadv,q,tphys,tadv,t = read_tendencies(region)
    return qphys,qadv,q,tphys,tadv,t

def driving_data_all_minmax(region: str):
    """
    This is for getting normalised driving data including surface variables
    data array shape = (n_samples, n_features) => (n_tsteps, n_levels)
    """
    qtot,qtot_next,qphys_tot,qadv,T,tphys,tadv,qadv_dot, tadv_dot, sw_toa_down,latent_up,sensible_up, mslp, sw_toa_up, lw_toa_up, sw_down, lw_down, rain, snow = model_trainining_ios(region)
    qadd = qtot + qadv
    qadd_dot = qtot + qadv_dot*600.
    tadd = T + tadv
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
    # train_test_datadir="data/models/datain/"
    # np.savez(train_test_datadir+"data_tot_raw_"+region, qtot=qtot,qphys_tot=qphys_tot,qadv=qadv,T=T,tphys=tphys,tadv=tadv,sw_toa_down=sw_toa_down,latent_up=latent_up,sensible_up=sensible_up, mslp=mslp, sw_toa_up=sw_toa_up, lw_toa_up=lw_toa_up, sw_down=sw_down, lw_down=lw_down, rain=rain, snow=snow)
    scale_qtot = minmax_data(qtot, save_fname="minmax_qtot.joblib",frange=(-1,1))
    scale_qadv = minmax_data(qadv, save_fname="minmax_qadvtot.joblib")
    scale_qadv_dot = minmax_data(qadv_dot, save_fname="minmax_qadv_dot.joblib")
    scale_qphys = minmax_data(qphys_tot, save_fname="minmax_qphystot.joblib")
    scale_qadd = minmax_data(qadd, save_fname="minmax_qadd.joblib")
    scale_qadd_dot = minmax_data(qadd_dot, save_fname="minmax_qadd_dot.joblib")
    scale_tadd = minmax_data(qadd, save_fname="minmax_tadd.joblib")
    scale_tadd_dot = minmax_data(qadd_dot, save_fname="minmax_tadd_dot.joblib")
    scale_T = minmax_data(T, save_fname="minmax_T.joblib",frange=(-1,1))
    scale_tphys = minmax_data(tphys, save_fname="minmax_tphys.joblib")
    scale_tadv = minmax_data(tadv, save_fname="minmax_tadv.joblib")
    scale_tadv_dot = minmax_data(tadv_dot, save_fname="minmax_tadv_dot.joblib")
    scale_sw_toa_down = minmax_data(sw_toa_down, save_fname="minmax_sw_toa_down.joblib",frange=(-1,1))
    scale_latent_up = minmax_data(latent_up, save_fname="minmax_latent_up.joblib",frange=(-1,1))
    scale_sensible_up = minmax_data(sensible_up, save_fname="minmax_sensible_up.joblib",frange=(-1,1))
    scale_mslp = minmax_data(mslp, save_fname="minmax_mslp.joblib",frange=(-1,1))
    scale_sw_toa_up = minmax_data(sw_toa_up, save_fname="minmax_sw_toa_up.joblib",frange=(-1,1))
    scale_lw_toa_up = minmax_data(lw_toa_up, save_fname="minmax_lw_toa_up.joblib",frange=(-1,1))
    scale_sw_down = minmax_data(sw_down, save_fname="minmax_sw_down.joblib",frange=(-1,1))
    scale_lw_down = minmax_data(lw_down, save_fname="minmax_lw_down.joblib",frange=(-1,1))
    scale_rain = minmax_data(rain, save_fname="minmax_rain.joblib",frange=(-1,1))
    scale_snow = minmax_data(snow, save_fname="minmax_snow.joblib",frange=(-1,1))

    std_qtot, std_qtot_test = train_test_split(scale_qtot.transform(qtot), shuffle=False)
    std_qtot_next, std_qtot_next_test = train_test_split(scale_qtot.transform(qtot_next), shuffle=False)
    std_qadv, std_qadv_test = train_test_split(scale_qadv.transform(qadv), shuffle=False)
    std_qadv_dot, std_qadv_dot_test = train_test_split(scale_qadv_dot.transform(qadv_dot), shuffle=False)
    std_qphys_tot, std_qphys_tot_test = train_test_split(scale_qphys.transform(qphys_tot), shuffle=False)
    std_qadd, std_qadd_test = train_test_split(scale_qadd.transform(qadd), shuffle=False)
    std_qadd_dot, std_qadd_dot_test = train_test_split(scale_qadd_dot.transform(qadd_dot), shuffle=False)
    std_tadd, std_tadd_test = train_test_split(scale_tadd.transform(tadd), shuffle=False)
    std_tadd_dot, std_tadd_dot_test = train_test_split(scale_tadd_dot.transform(tadd_dot), shuffle=False)
    std_T, std_T_test = train_test_split(scale_T.transform(T), shuffle=False)
    std_tphys, std_tphys_test = train_test_split(scale_tphys.transform(tphys), shuffle=False)
    std_tadv, std_tadv_test = train_test_split(scale_tadv.transform(tadv), shuffle=False)
    std_tadv_dot, std_tadv_dot_test = train_test_split(scale_tadv_dot.transform(tadv_dot), shuffle=False)
    std_sw_toa_down, std_sw_toa_down_test = train_test_split(scale_sw_toa_down.transform(sw_toa_down), shuffle=False)
    std_latent_up, std_latent_up_test = train_test_split(scale_latent_up.transform(latent_up), shuffle=False)
    std_sensible_up, std_sensible_up_test = train_test_split(scale_sensible_up.transform(sensible_up), shuffle=False)
    std_mslp, std_mslp_test = train_test_split(scale_mslp.transform(mslp), shuffle=False)
    std_sw_toa_up, std_sw_toa_up_test = train_test_split(scale_sw_toa_up.transform(sw_toa_up), shuffle=False)
    std_lw_toa_up, std_lw_toa_up_test = train_test_split(scale_lw_toa_up.transform(lw_toa_up), shuffle=False)
    std_sw_down, std_sw_down_test = train_test_split(scale_sw_down.transform(sw_down), shuffle=False)
    std_lw_down, std_lw_down_test = train_test_split(scale_lw_down.transform(lw_down), shuffle=False)
    std_rain, std_rain_test = train_test_split(scale_rain.transform(rain), shuffle=False)
    std_snow, std_snow_test = train_test_split(scale_snow.transform(snow), shuffle=False)

    qtot_raw, qtot_test_raw = train_test_split(qtot, shuffle=False)
    qadv_raw, qadv_test_raw = train_test_split(qadv, shuffle=False)
    qadv_dot_raw, qadv_dot_test_raw = train_test_split(qadv_dot, shuffle=False)
    qphys_tot_raw, qphys_tot_test_raw = train_test_split(qphys_tot, shuffle=False)
    qadd_raw, qadd_test_raw = train_test_split(qadd, shuffle=False)
    qadd_dot_raw, qadd_dot_test_raw = train_test_split(qadd_dot, shuffle=False)
    tadd_raw, tadd_test_raw = train_test_split(tadd, shuffle=False)    
    tadd_dot_raw, tadd_dot_test_raw = train_test_split(tadd_dot, shuffle=False)
    T_raw, T_test_raw = train_test_split(T, shuffle=False)
    tphys_raw, tphys_test_raw = train_test_split(tphys, shuffle=False)
    tadv_raw, tadv_test_raw = train_test_split(tadv, shuffle=False)
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
    variables = {"qtot":std_qtot,"qtot_next":std_qtot_next, "qphys_tot":std_qphys_tot, "qadv":std_qadv, "qadv_dot":std_qadv_dot, "qadd":std_qadd, "qadd_dot":std_qadd_dot, "tadd":std_tadd, "tadd_dot":std_tadd_dot, "T":std_T, "tphys":std_tphys, "tadv":std_tadv, "tadv_dot":std_tadv_dot, "sw_toa_down":std_sw_toa_down, "latent_up":std_latent_up, "sensible_up":std_sensible_up, "mslp":std_mslp, "sw_toa_up":std_sw_toa_up, "lw_toa_up":std_lw_toa_up, "sw_down":std_sw_down, "lw_down":std_lw_down, "rain":std_rain, "snow":std_snow, "qtot_test":std_qtot_test, "qtot_next_test":std_qtot_next_test,"qphys_test":std_qphys_tot_test, "qadv_test":std_qadv_test, "qadv_dot_test":std_qadv_dot_test, "qadd_test":std_qadd_test, "qadd_dot_test":std_qadd_dot_test, "tadd_test":std_tadd_test, "tadd_dot_test":std_tadd_dot_test, "T_test":std_T_test, "tphys_test":std_tphys_test, "tadv_test":std_tadv_test, "tadv_dot_test":std_tadv_dot_test, "sw_toa_down_test":std_sw_toa_down_test, "latent_up_test":std_latent_up_test, "sensible_up_test":std_sensible_up_test, "mslp_test":std_mslp_test, "sw_toa_up_test":std_sw_toa_up_test, "lw_toa_up_test":std_lw_toa_up_test, "sw_down_test":std_sw_down_test, "lw_down_test":std_lw_down_test, "rain_test":std_rain_test, "snow_test":std_snow_test, "qtot_raw":qtot, "qphys_raw":qphys_tot_raw, "qadv_raw":qadv_raw, "qadv_dot_raw":qadv_dot_raw, "qadd_raw":qadd_raw, "qadd_dot_raw":qadd_dot_raw, "tadd_raw":tadd_raw, "tadd_dot_raw":tadd_dot_raw, "T_raw":T_raw, "tphys_raw":tphys_raw, "tadv_raw":tadv_raw, "tadv_dot_raw":tadv_dot_raw, "sw_toa_down_raw":sw_toa_down_raw, "latent_up_raw":latent_up_raw, "sensible_up_raw":sensible_up_raw, "mslp_raw":mslp_raw, "sw_toa_up_raw":sw_toa_up_raw, "lw_toa_up_raw":lw_toa_up_raw, "sw_down_raw":sw_down_raw, "lw_down_raw":lw_down_raw, "rain_raw":rain_raw, "snow_raw":snow_raw, "qtot_test_raw":qtot_test_raw, "qphys_test_raw":qphys_tot_test_raw, "qadv_test_raw":qadv_test_raw, "qadv_dot_test_raw":qadv_dot_test_raw, "qadd_test_raw":qadd_test_raw, "qadd_dot_test_raw":qadd_dot_test_raw, "tadd_test_raw":tadd_test_raw, "tadd_dot_test_raw":tadd_dot_test_raw, "T_test_raw":T_test_raw, "tphys_test_raw":tphys_test_raw, "tadv_test_raw":tadv_test_raw, "tadv_dot_test_raw":tadv_dot_test_raw, "sw_toa_down_test_raw":sw_toa_down_test_raw, "latent_up_test_raw":latent_up_test_raw, "sensible_up_test_raw":sensible_up_test_raw, "mslp_test_raw":mslp_test_raw, "sw_toa_up_test_raw":sw_toa_up_test_raw, "lw_toa_up_test_raw":lw_toa_up_test_raw, "sw_down_test_raw":sw_down_test_raw, "lw_down_test_raw":lw_down_test_raw, "rain_test_raw":rain_test_raw, "snow_test_raw":snow_test_raw}

    return variables

def driving_data_all_std(region: str):
    """
    This is for getting normalised driving data including surface variables
    data array shape = (n_samples, n_features) => (n_tsteps, n_levels)
    """
    qtot,qtot_next,qphys_tot,qadv,T,tphys,tadv,qadv_dot, tadv_dot, sw_toa_down,latent_up,sensible_up, mslp, sw_toa_up, lw_toa_up, sw_down, lw_down, rain, snow = model_trainining_ios(region)
    qadd = qtot + qadv
    qadd_dot = qtot + qadv_dot*600.
    tadd = T + tadv
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
    # train_test_datadir="data/models/datain/"
    # np.savez(train_test_datadir+"data_tot_raw_"+region, qtot=qtot,qphys_tot=qphys_tot,qadv=qadv,T=T,tphys=tphys,tadv=tadv,sw_toa_down=sw_toa_down,latent_up=latent_up,sensible_up=sensible_up, mslp=mslp, sw_toa_up=sw_toa_up, lw_toa_up=lw_toa_up, sw_down=sw_down, lw_down=lw_down, rain=rain, snow=snow)
    scale_qtot = standardise_data_transform(qtot, save_fname="std_qtot.joblib")
    scale_qadv = standardise_data_transform(qadv, save_fname="std_qadvtot.joblib")
    scale_qadv_dot = standardise_data_transform(qadv_dot, save_fname="std_qadv_dot.joblib")
    scale_qphys = standardise_data_transform(qphys_tot, save_fname="std_qphystot.joblib")
    scale_qadd = standardise_data_transform(qadd, save_fname="std_qadd.joblib")
    scale_qadd_dot = standardise_data_transform(qadd_dot, save_fname="std_qadd_dot.joblib")
    scale_tadd = standardise_data_transform(qadd, save_fname="std_tadd.joblib")
    scale_tadd_dot = standardise_data_transform(qadd_dot, save_fname="std_tadd_dot.joblib")
    scale_T = standardise_data_transform(T, save_fname="std_T.joblib")
    scale_tphys = standardise_data_transform(tphys, save_fname="std_tphys.joblib")
    scale_tadv = standardise_data_transform(tadv, save_fname="std_tadv.joblib")
    scale_tadv_dot = standardise_data_transform(tadv_dot, save_fname="std_tadv_dot.joblib")
    scale_sw_toa_down = standardise_data_transform(sw_toa_down, save_fname="std_sw_toa_down.joblib")
    scale_latent_up = standardise_data_transform(latent_up, save_fname="std_latent_up.joblib")
    scale_sensible_up = standardise_data_transform(sensible_up, save_fname="std_sensible_up.joblib")
    scale_mslp = standardise_data_transform(mslp, save_fname="std_mslp.joblib")
    scale_sw_toa_up = standardise_data_transform(sw_toa_up, save_fname="std_sw_toa_up.joblib")
    scale_lw_toa_up = standardise_data_transform(lw_toa_up, save_fname="std_lw_toa_up.joblib")
    scale_sw_down = standardise_data_transform(sw_down, save_fname="std_sw_down.joblib")
    scale_lw_down = standardise_data_transform(lw_down, save_fname="std_lw_down.joblib")
    scale_rain = standardise_data_transform(rain, save_fname="std_rain.joblib")
    scale_snow = standardise_data_transform(snow, save_fname="std_snow.joblib")

    std_qtot, std_qtot_test = train_test_split(scale_qtot, shuffle=False)
    std_qtot_next, std_qtot_next_test = train_test_split(scale_qtot, shuffle=False)
    std_qadv, std_qadv_test = train_test_split(scale_qadv, shuffle=False)
    std_qadv_dot, std_qadv_dot_test = train_test_split(scale_qadv_dot, shuffle=False)
    std_qphys_tot, std_qphys_tot_test = train_test_split(scale_qphys, shuffle=False)
    std_qadd, std_qadd_test = train_test_split(scale_qadd, shuffle=False)
    std_qadd_dot, std_qadd_dot_test = train_test_split(scale_qadd_dot, shuffle=False)
    std_tadd, std_tadd_test = train_test_split(scale_tadd, shuffle=False)
    std_tadd_dot, std_tadd_dot_test = train_test_split(scale_tadd_dot, shuffle=False)
    std_T, std_T_test = train_test_split(scale_T, shuffle=False)
    std_tphys, std_tphys_test = train_test_split(scale_tphys, shuffle=False)
    std_tadv, std_tadv_test = train_test_split(scale_tadv, shuffle=False)
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
    qadv_raw, qadv_test_raw = train_test_split(qadv, shuffle=False)
    qadv_dot_raw, qadv_dot_test_raw = train_test_split(qadv_dot, shuffle=False)
    qphys_tot_raw, qphys_tot_test_raw = train_test_split(qphys_tot, shuffle=False)
    qadd_raw, qadd_test_raw = train_test_split(qadd, shuffle=False)
    qadd_dot_raw, qadd_dot_test_raw = train_test_split(qadd_dot, shuffle=False)
    tadd_raw, tadd_test_raw = train_test_split(tadd, shuffle=False)    
    tadd_dot_raw, tadd_dot_test_raw = train_test_split(tadd_dot, shuffle=False)
    T_raw, T_test_raw = train_test_split(T, shuffle=False)
    tphys_raw, tphys_test_raw = train_test_split(tphys, shuffle=False)
    tadv_raw, tadv_test_raw = train_test_split(tadv, shuffle=False)
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
    variables = {"qtot":std_qtot,"qtot_next":std_qtot_next, "qphys_tot":std_qphys_tot, "qadv":std_qadv, "qadv_dot":std_qadv_dot, "qadd":std_qadd, "qadd_dot":std_qadd_dot, "tadd":std_tadd, "tadd_dot":std_tadd_dot, "T":std_T, "tphys":std_tphys, "tadv":std_tadv, "tadv_dot":std_tadv_dot, "sw_toa_down":std_sw_toa_down, "latent_up":std_latent_up, "sensible_up":std_sensible_up, "mslp":std_mslp, "sw_toa_up":std_sw_toa_up, "lw_toa_up":std_lw_toa_up, "sw_down":std_sw_down, "lw_down":std_lw_down, "rain":std_rain, "snow":std_snow, "qtot_test":std_qtot_test, "qtot_next_test":std_qtot_next_test,"qphys_test":std_qphys_tot_test, "qadv_test":std_qadv_test, "qadv_dot_test":std_qadv_dot_test, "qadd_test":std_qadd_test, "qadd_dot_test":std_qadd_dot_test, "tadd_test":std_tadd_test, "tadd_dot_test":std_tadd_dot_test, "T_test":std_T_test, "tphys_test":std_tphys_test, "tadv_test":std_tadv_test, "tadv_dot_test":std_tadv_dot_test, "sw_toa_down_test":std_sw_toa_down_test, "latent_up_test":std_latent_up_test, "sensible_up_test":std_sensible_up_test, "mslp_test":std_mslp_test, "sw_toa_up_test":std_sw_toa_up_test, "lw_toa_up_test":std_lw_toa_up_test, "sw_down_test":std_sw_down_test, "lw_down_test":std_lw_down_test, "rain_test":std_rain_test, "snow_test":std_snow_test, "qtot_raw":qtot, "qphys_raw":qphys_tot_raw, "qadv_raw":qadv_raw, "qadv_dot_raw":qadv_dot_raw, "qadd_raw":qadd_raw, "qadd_dot_raw":qadd_dot_raw, "tadd_raw":tadd_raw, "tadd_dot_raw":tadd_dot_raw, "T_raw":T_raw, "tphys_raw":tphys_raw, "tadv_raw":tadv_raw, "tadv_dot_raw":tadv_dot_raw, "sw_toa_down_raw":sw_toa_down_raw, "latent_up_raw":latent_up_raw, "sensible_up_raw":sensible_up_raw, "mslp_raw":mslp_raw, "sw_toa_up_raw":sw_toa_up_raw, "lw_toa_up_raw":lw_toa_up_raw, "sw_down_raw":sw_down_raw, "lw_down_raw":lw_down_raw, "rain_raw":rain_raw, "snow_raw":snow_raw, "qtot_test_raw":qtot_test_raw, "qphys_test_raw":qphys_tot_test_raw, "qadv_test_raw":qadv_test_raw, "qadv_dot_test_raw":qadv_dot_test_raw, "qadd_test_raw":qadd_test_raw, "qadd_dot_test_raw":qadd_dot_test_raw, "tadd_test_raw":tadd_test_raw, "tadd_dot_test_raw":tadd_dot_test_raw, "T_test_raw":T_test_raw, "tphys_test_raw":tphys_test_raw, "tadv_test_raw":tadv_test_raw, "tadv_dot_test_raw":tadv_dot_test_raw, "sw_toa_down_test_raw":sw_toa_down_test_raw, "latent_up_test_raw":latent_up_test_raw, "sensible_up_test_raw":sensible_up_test_raw, "mslp_test_raw":mslp_test_raw, "sw_toa_up_test_raw":sw_toa_up_test_raw, "lw_toa_up_test_raw":lw_toa_up_test_raw, "sw_down_test_raw":sw_down_test_raw, "lw_down_test_raw":lw_down_test_raw, "rain_test_raw":rain_test_raw, "snow_test_raw":snow_test_raw}

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
    
def train_test_data_save(region: str):
    dataset = driving_data(region)
    q_train, q_test, q_norm_train, q_norm_test, qadv_train, qadv_test, qadv_norm_train, qadv_norm_test, qphys_train, qphys_test, qphys_norm_train, qphys_norm_test, t_train, t_test, t_norm_train, t_norm_test, tadv_train, tadv_test, tadv_norm_train, tadv_norm_test, tphys_train, tphys_test, tphys_norm_train, tphys_norm_test = dataset[0], dataset[1], dataset[2], dataset[3], dataset[4], dataset[5], dataset[6], dataset[7], dataset[8], dataset[9], dataset[10], dataset[11], dataset[12], dataset[13], dataset[14], dataset[15], dataset[16], dataset[17], dataset[18], dataset[19], dataset[20], dataset[21], dataset[22], dataset[23]
    
    train_test_datadir = "{0}/models/datain/".format(crm_data)
    np.savez(train_test_datadir+'train_test_data_'+region,q_train=q_train, q_test=q_test, qadv_train=qadv_train, qadv_test=qadv_test, qphys_train=qphys_train, qphys_test=qphys_test, t_train=t_train, t_test=t_test, tadv_train=tadv_train, tadv_test=tadv_test, tphys_train=tphys_train, tphys_test=tphys_test, q_norm_train=q_norm_train, q_norm_test=q_norm_test, qadv_norm_train=qadv_norm_train, qadv_norm_test=qadv_norm_test, qphys_norm_train=qphys_norm_train, qphys_norm_test=qphys_norm_test, t_norm_train=t_norm_train, t_norm_test=t_norm_test, tadv_norm_train=tadv_norm_train, tadv_norm_test=tadv_norm_test, tphys_norm_train=tphys_norm_train, tphys_norm_test=tphys_norm_test)

def train_test_data_save_all(region: str):

    # variables = driving_data_all_minmax(region)
    variables = driving_data_all_std(region)

    train_test_datadir = "{0}/models/datain/".format(crm_data)
    # Save in hdf5 format
    filename='train_test_data_all_{0}_std.hdf5'.format(region)
    with h5py.File(train_test_datadir+filename, 'w') as hfile:
        for k, v in variables.items():  
            hfile.create_dataset(k,data=v)

    
def raw_data_save(region: str):
    region = "50S69W"
    qphys,qadv,q,tphys,tadv,t = read_tendencies(region)
    datadir = "{0}/models/datain/".format(crm_data)
    np.savez(datadir+'raw_data_'+region, qphys=qphys, qadv=qadv, q=q, tphys=tphys,tadv=tadv,t=t)
    
if __name__ == "__main__":
    region = "50S69W"
    # train_test_data_save(region)
    # driving_data_all(region)
    # driving_data_stats(region)
    # raw_data_save(region)
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
    
    
