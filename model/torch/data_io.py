"""
Read the data saved by model_input_data_process.py
and make available for model training and testing
"""

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import h5py
import torch

# locations={ "train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain"}

class Data_IO_validation(object):
    def __init__(self, region, locations):
        self.region = region
        self.locations = locations

    def get_data(self, subdomain):   
        dataset_file = "{0}/validation_{1}/validation_data_{1}_{2}.hdf5".format(self.locations["train_test_datadir"],self.region, str(subdomain).zfill(3))
        print("Reading dataset file: {0}".format(dataset_file))
        dataset=h5py.File(dataset_file,'r')

        self.q_tot_test = dataset["q_tot_test"]
        self.q_tot_adv_test = dataset["q_adv_test"]
        self.theta_test = dataset["air_potential_temperature_test"]
        self.theta_adv_test = dataset["t_adv_test"]
        self.sw_toa_test = dataset["toa_incoming_shortwave_flux_test"]
        self.shf_test = dataset["surface_upward_sensible_heat_flux_test"]
        self.lhf_test = dataset["surface_upward_latent_heat_flux_test"]
        self.theta_phys_test = dataset["t_phys_test"]
        self.qphys_test = dataset["q_phys_test"]

# class Data_IO(object):
#     def __init__(self, region, locations):
#         self.region = region
#         self.locations = locations
        
#         dataset_file = "{0}/train_test_data_{1}.hdf5".format(self.locations["train_test_datadir"],self.region)
#         print("Reading dataset file: {0}".format(dataset_file))
#         dataset=h5py.File(dataset_file,'r')
       
#         self.q_tot_train = dataset["q_tot_train"]
#         self.q_tot_adv_train = dataset["q_adv_train"]
#         self.theta_train = dataset["air_potential_temperature_train"]
#         self.theta_adv_train = dataset["t_adv_train"]
#         self.sw_toa_train = dataset["toa_incoming_shortwave_flux_train"]
#         self.shf_train = dataset["surface_upward_sensible_heat_flux_train"]
#         self.lhf_train = dataset["surface_upward_latent_heat_flux_train"]
#         self.theta_phys_train = dataset["t_phys_train"]
#         self.qphys_train = dataset["q_phys_train"]


#         self.q_tot_test = dataset["q_tot_test"]
#         self.q_tot_adv_test = dataset["q_adv_test"]
#         self.theta_test = dataset["air_potential_temperature_test"]
#         self.theta_adv_test = dataset["t_adv_test"]
#         self.sw_toa_test = dataset["toa_incoming_shortwave_flux_test"]
#         self.shf_test = dataset["surface_upward_sensible_heat_flux_test"]
#         self.lhf_test = dataset["surface_upward_latent_heat_flux_test"]
#         self.theta_phys_test = dataset["t_phys_test"]
#         self.qphys_test = dataset["q_phys_test"]



class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, dat_type, nlevs, dataset_file, data_frac=1.):
        super().__init__()
        self.dat_type = dat_type
        self.dataset_file = dataset_file
        self.nlevs = nlevs
        
        if dat_type == "train":
            print("Reading dataset file: {0}".format(dataset_file))
            dataset=h5py.File(dataset_file,'r')
            self.q_tot_train = dataset["q_tot_train"]
            self.q_tot_adv_train = dataset["q_adv_train"]
            self.theta_train = dataset["air_potential_temperature_train"]
            self.theta_adv_train = dataset["t_adv_train"]
            self.sw_toa_train = dataset["toa_incoming_shortwave_flux_train"]
            self.shf_train = dataset["surface_upward_sensible_heat_flux_train"]
            self.lhf_train = dataset["surface_upward_latent_heat_flux_train"]
            self.theta_phys_train = dataset["t_phys_train"]
            self.qphys_train = dataset["q_phys_train"]
            
            self.npoints = int(self.q_tot_train.shape[0] * data_frac)
            self.x3data = [self.q_tot_train[:self.npoints], self.q_tot_adv_train[:self.npoints], self.theta_train[:self.npoints], self.theta_adv_train[:self.npoints]]
            # self.x3data = [self.q_tot_train[:self.npoints] + self.q_tot_adv_train[:self.npoints], self.theta_train[:self.npoints] + self.theta_adv_train[:self.npoints]]
            self.x2data = [self.sw_toa_train[:self.npoints], self.lhf_train[:self.npoints], self.shf_train[:self.npoints]]
            self.ydata = [self.qphys_train[:self.npoints], self.theta_phys_train[:self.npoints]]
            # self.ydata = [self.theta_phys_train[:self.npoints]]
            # self.ydata = [self.qphys_train[:self.npoints]]
        elif dat_type == "test":
            print("Reading dataset file: {0}".format(dataset_file))
            dataset=h5py.File(dataset_file,'r')
            self.q_tot_test = dataset["q_tot_test"]
            self.q_tot_adv_test = dataset["q_adv_test"]
            self.theta_test = dataset["air_potential_temperature_test"]
            self.theta_adv_test = dataset["t_adv_test"]
            self.sw_toa_test = dataset["toa_incoming_shortwave_flux_test"]
            self.shf_test = dataset["surface_upward_sensible_heat_flux_test"]
            self.lhf_test = dataset["surface_upward_latent_heat_flux_test"]
            self.theta_phys_test = dataset["t_phys_test"]
            self.qphys_test = dataset["q_phys_test"] 
            
            self.npoints = int(self.q_tot_test.shape[0] * data_frac)
            self.x3data = [self.q_tot_test[:self.npoints], self.q_tot_adv_test[:self.npoints], self.theta_test[:self.npoints], self.theta_adv_test[:self.npoints]]
            # self.x3data = [self.q_tot_test[:self.npoints] + self.q_tot_adv_test[:self.npoints], self.theta_test[:self.npoints] + self.theta_adv_test[:self.npoints]]
            self.x2data = [self.sw_toa_test[:self.npoints], self.lhf_test[:self.npoints], self.shf_test[:self.npoints]]
            self.ydata = [self.qphys_test[:self.npoints], self.theta_phys_test[:self.npoints]]
            # self.ydata = [self.theta_phys_test[:self.npoints]]
            # self.ydata = [self.qphys_test[:self.npoints]]

    def __getitem__(self, i):
        """
        batch iterable
        """
        x3 = [d[i,:self.nlevs] for d in self.x3data]
        x2 = [d[i] for d in self.x2data]
        x =  np.concatenate((x3+x2))
        y = np.concatenate(([d[i,:self.nlevs] for d in self.ydata]))

        return (torch.from_numpy(x),torch.from_numpy(y))

    def __len__(self):
        return min(len(d) for d in self.x2data)
        