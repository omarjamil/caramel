"""
Read the data saved by model_input_data_process.py
and make available for model training and testing
"""

import numpy as np
from netCDF4 import Dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import joblib
import h5py

# locations={ "train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain"}

class Data_IO(object):
    def __init__(self, region, locations):
        self.region = region
        self.locations = locations
        self.train_data_in, self.train_data_out, self.test_data_in, self.test_data_out = self.scm_model_data()
        self.q_norm_train = self.train_data_in["qtot"]
        self.qnext_norm_train = self.train_data_in["qtot_next"]
        self.qadv_norm_train = self.train_data_in["qadv"]
        self.qadv_dot_norm_train = self.train_data_in["qadv_dot"]
        self.qphys_norm_train = self.train_data_out["qphys_tot"]
        self.qphys_dot_norm_train = self.train_data_out["qphys_dot"]

        self.q_norm_test = self.test_data_in["qtot_test"]
        self.qnext_norm_test = self.test_data_in["qtot_next_test"]
        self.qadv_norm_test = self.test_data_in["qadv_test"]
        self.qadv_dot_norm_test = self.test_data_in["qadv_dot_test"]
        self.qphys_norm_test = self.test_data_out["qphys_test"]
        self.qphys_dot_norm_test = self.test_data_out["qphys_dot_test"]
        self.qadd_train = self.train_data_in["qadd"]
        self.qadd_dot_train = self.train_data_in["qadd_dot"]
        self.qadd_test = self.test_data_in["qadd_test"]
        self.qadd_dot_test = self.test_data_in["qadd_dot_test"]
        self.q_test_raw = self.test_data_in["qtot_test_raw"]
        self.qadv_test_raw = self.test_data_in["qadv_test_raw"]
        self.qadv_dot_test_raw = self.test_data_in["qadv_dot_test_raw"]
        self.qphys_test_raw = self.test_data_out["qphys_test_raw"]
        self.qphys_dot_test_raw = self.test_data_out["qphys_dot_test_raw"]
        
    def scm_model_data(self):
        """
        Data for model with all SCM type inputs and outputs
        """
        # dataset_file = "train_test_data_all_m1p1{0}.npz".format(region)
        # dataset=np.load("{0}/{1}".format(locations["train_test_datadir"],dataset_file))
        dataset_file = "{0}/train_test_data_all_{1}_std.hdf5".format(self.locations["train_test_datadir"],self.region)
        dataset=h5py.File(dataset_file,'r')
        
        qtot = dataset["qtot"]
        qtot_next = dataset["qtot_next"]
        qphys_tot = dataset["qphys_tot"]
        qphys_dot = dataset["qphys_dot"]
        qadv = dataset["qadv"]
        qadv_dot = dataset["qadv_dot"]
        qadd = dataset["qadd"]
        qadd_dot = dataset["qadd_dot"] #qtot + qadv_dot*600.   
        tadd = dataset["tadd"]
        tadd_dot = dataset["tadd_dot"] 
        T = dataset["T"]
        tphys = dataset["tphys"]
        tadv = dataset["tadv"]
        tadv_dot = dataset["tadv_dot"]
        sw_toa_down = dataset["sw_toa_down"]
        latent_up = dataset["latent_up"]
        sensible_up = dataset["sensible_up"]
        mslp = dataset["mslp"]
        sw_toa_up = dataset["sw_toa_up"]
        lw_toa_up = dataset["lw_toa_up"]
        sw_down = dataset["sw_down"]
        lw_down = dataset["lw_down"]
        rain = dataset["rain"]
        snow = dataset["snow"]
        # train_model_data_in = np.concatenate((qtot, qadv, T, tadv, sw_toa_down, latent_up, sensible_up, mslp), axis=1)
        # train_model_data_out = np.concatenate((qphys_tot, tphys, sw_toa_up, lw_toa_up, sw_down, lw_down, rain, snow), axis=1)

        qtot_test = dataset["qtot_test"]
        qtot_next_test = dataset["qtot_next_test"]
        qphys_test = dataset["qphys_test"]   
        qphys_dot_test = dataset["qphys_dot_test"]   
        qadv_test = dataset["qadv_test"]
        qadv_dot_test = dataset["qadv_dot_test"]
        qadd_test = dataset["qadd_test"]
        qadd_dot_test = dataset["qadd_dot_test"]
        tadd_test = dataset["tadd_test"]
        tadd_dot_test = dataset["tadd_dot_test"]
        T_test = dataset["T_test"]
        tphys_test = dataset["tphys_test"]
        tadv_test = dataset["tadv_test"]
        tadv_dot_test = dataset["tadv_dot_test"]
        sw_toa_down_test = dataset["sw_toa_down_test"]
        latent_up_test = dataset["latent_up_test"]
        sensible_up_test = dataset["sensible_up_test"]
        mslp_test = dataset["mslp_test"]
        sw_toa_up_test = dataset["sw_toa_up_test"]
        lw_toa_up_test = dataset["lw_toa_up_test"]
        sw_down_test = dataset["sw_down_test"]
        lw_down_test = dataset["lw_down_test"]
        rain_test = dataset["rain_test"]
        snow_test = dataset["snow_test"]

        qtot_test_raw = dataset["qtot_test_raw"]
        qadv_test_raw = dataset["qadv_test_raw"]
        qadv_dot_test_raw = dataset["qadv_dot_test_raw"]
        qadd_test_raw = dataset["qadd_test_raw"]    
        qadd_dot_test_raw = dataset["qadd_dot_test_raw"]
        tadd_test_raw = dataset["tadd_test_raw"]    
        tadd_dot_test_raw = dataset["tadd_dot_test_raw"]
        qphys_test_raw = dataset["qphys_test_raw"]
        qphys_dot_test_raw = dataset["qphys_dot_test_raw"]
        qadd_test_raw = qtot_test_raw[:] + qadv_test_raw[:]  
        # test_model_data_in = np.concatenate((qtot_test, qadv_test, T_test, tadv_test, sw_toa_down_test, latent_up_test, sensible_up_test, mslp_test), axis=1)
        # test_model_data_out = np.concatenate((qphys_test, tphys_test, sw_toa_up_test, lw_toa_up_test, sw_down_test, lw_down_test, rain_test, snow_test), axis=1)

        train_data_in = {"qtot":qtot, "qtot_next":qtot_next, "qadv":qadv, "qadv_dot":qadv_dot, "qadd":qadd, "qadd_dot":qadd_dot, "T":T, "tadv":tadv, "tadv_dot":tadv_dot, "tadd":tadd, "tadd_dot":tadd_dot, "sw_toa_down":sw_toa_down, "latent_up":latent_up, "sensible_up":sensible_up, "mslp":mslp}
        train_data_out = {"qphys_tot":qphys_tot, "qphys_dot":qphys_dot, "tphys":tphys, "sw_toa_up":sw_toa_up, "low_toa_up":lw_toa_up, "sw_down":sw_down, "lw_down":lw_down, "rain":rain, "snow":snow}
        test_data_in = {"qtot_test":qtot_test, "qtot_next_test":qtot_next_test, "qadv_test":qadv_test, "qadv_dot_test":qadv_dot_test, "qadd_test":qadd_test, "qadd_dot_test":qadd_dot_test, "qadd_test_raw":qadd_test_raw, "qadd_dot_test_raw":qadd_dot_test_raw, "T_test":T_test, "tadv_test":tadv_test, "tadv_dot_test":tadv_dot_test, "tadd_test":tadd_test, "tadd_dot_test":tadd_dot_test, "tadd_test_raw":tadd_test_raw, "tadd_dot_test_raw":tadd_dot_test_raw, "sw_toa_down_test":sw_toa_down_test, "latent_up_test":latent_up_test, "sensible_up_test":sensible_up_test, "mslp":mslp_test, "qadv_test_raw":qadv_test_raw, "qadv_dot_test_raw":qadv_dot_test_raw, "qtot_test_raw":qtot_test_raw}
        test_data_out = {"qphys_test":qphys_test, "qphys_dot_test":qphys_dot_test, "tphys_test":tphys_test, "sw_toa_up_test":sw_toa_up_test, "low_toa_up_test":lw_toa_up_test, "sw_down_test":sw_down_test, "lw_down_test":lw_down_test, "rain_test":rain_test, "snow_test":snow_test, "qphys_test_raw":qphys_test_raw, "qphys_dot_test_raw":qphys_dot_test_raw}
        
        return train_data_in, train_data_out, test_data_in, test_data_out