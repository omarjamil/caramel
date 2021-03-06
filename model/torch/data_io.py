"""
Read the data saved by model_input_data_process.py
and make available for model training and testing
"""

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import h5py
import torch
from torchvision import transforms

# locations={ "train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain"}

class NormalizersData(object):
    def __init__(self, location):
        print("Initialising normaliser to location: {0}".format(location))
        # self.qphys_normaliser_std = h5py.File('{0}/q_phys.hdf5'.format(location),'r')
        # self.tphys_normaliser_std = h5py.File('{0}/t_phys.hdf5'.format(location),'r')
        self.q_normaliser_std = h5py.File('{0}/q_tot.hdf5'.format(location),'r')
        # self.qv_normaliser_std = h5py.File('{0}/specific_humidity.hdf5'.format(location),'r')
        self.t_normaliser_std = h5py.File('{0}/air_potential_temperature.hdf5'.format(location),'r')
        # self.qadv_normaliser_std = h5py.File('{0}/q_adv.hdf5'.format(location),'r')
        # self.tadv_normaliser_std = h5py.File('{0}/t_adv.hdf5'.format(location),'r')
        self.sw_toa_normaliser_std = h5py.File('{0}/toa_incoming_shortwave_flux.hdf5'.format(location),'r')
        self.upshf_normaliser_std = h5py.File('{0}/surface_upward_sensible_heat_flux.hdf5'.format(location),'r')
        self.uplhf_normaliser_std = h5py.File('{0}/surface_upward_latent_heat_flux.hdf5'.format(location),'r')
        self.pressure_std = h5py.File('{0}/air_pressure.hdf5'.format(location),'r')
        self.rho_std = h5py.File('{0}/m01s00i253.hdf5'.format(location),'r')
        self.xwind_std = h5py.File('{0}/x_wind.hdf5'.format(location),'r')
        self.ywind_std = h5py.File('{0}/y_wind.hdf5'.format(location),'r')
        self.zwind_std = h5py.File('{0}/upward_air_velocity.hdf5'.format(location),'r')

        # self.qphys_mean = torch.tensor(self.qphys_normaliser_std['mean_'][:])
        # self.tphys_mean = torch.tensor(self.tphys_normaliser_std['mean_'][:])
        self.q_mean = torch.tensor(self.q_normaliser_std['mean_'][:])
        # self.qv_mean = torch.tensor(self.qv_normaliser_std['mean_'][:])
        self.t_mean = torch.tensor(self.t_normaliser_std['mean_'][:])
        # self.qadv_mean = torch.tensor(self.qadv_normaliser_std['mean_'][:])
        # self.tadv_mean = torch.tensor(self.tadv_normaliser_std['mean_'][:])
        self.sw_toa_mean = torch.tensor(self.sw_toa_normaliser_std['mean_'][:])
        self.upshf_mean = torch.tensor(self.uplhf_normaliser_std['mean_'][:])
        self.uplhf_mean = torch.tensor(self.uplhf_normaliser_std['mean_'][:])
        self.pressure_mean = torch.tensor(self.pressure_std['mean_'][:])
        self.rho_mean = torch.tensor(self.rho_std['mean_'][:])
        self.xwind_mean = torch.tensor(self.xwind_std['mean_'][:])
        self.ywind_mean = torch.tensor(self.ywind_std['mean_'][:])
        self.zwind_mean = torch.tensor(self.zwind_std['mean_'][:])

        # self.qphys_stdscale = torch.from_numpy(self.qphys_normaliser_std['scale_'][:])
        # self.tphys_stdscale = torch.from_numpy(self.tphys_normaliser_std['scale_'][:])
        self.q_stdscale = torch.from_numpy(self.q_normaliser_std['scale_'][:])
        # self.qv_stdscale = torch.from_numpy(self.qv_normaliser_std['scale_'][:])
        self.t_stdscale = torch.from_numpy(self.t_normaliser_std['scale_'][:])
        # self.qadv_stdscale = torch.from_numpy(self.qadv_normaliser_std['scale_'][:])
        # self.tadv_stdscale = torch.from_numpy(self.tadv_normaliser_std['scale_'][:])
        self.sw_toa_stdscale = torch.tensor(self.sw_toa_normaliser_std['scale_'][:])
        self.upshf_stdscale = torch.tensor(self.uplhf_normaliser_std['scale_'][:])
        self.uplhf_stdscale = torch.tensor(self.uplhf_normaliser_std['scale_'][:])
        self.pressure_stdscale = torch.tensor(self.pressure_std['scale_'][:])
        self.rho_stdscale = torch.tensor(self.rho_std['scale_'][:])
        self.xwind_stdscale = torch.tensor(self.xwind_std['scale_'][:])
        self.ywind_stdscale = torch.tensor(self.ywind_std['scale_'][:])
        self.zwind_stdscale = torch.tensor(self.zwind_std['scale_'][:])

        # self.qphys_mean_np = self.qphys_normaliser_std['mean_'][:]
        # self.tphys_mean_np = self.tphys_normaliser_std['mean_'][:]
        self.q_mean_np = self.q_normaliser_std['mean_'][:]
        # self.qv_mean_np = self.qv_normaliser_std['mean_'][:]
        self.t_mean_np = self.t_normaliser_std['mean_'][:]
        # self.qadv_mean_np = self.qadv_normaliser_std['mean_'][:]
        # self.tadv_mean_np = self.tadv_normaliser_std['mean_'][:]
        self.sw_toa_mean_np = self.sw_toa_normaliser_std['mean_'][:]
        self.upshf_mean_np = self.uplhf_normaliser_std['mean_'][:]
        self.uplhf_mean_np = self.uplhf_normaliser_std['mean_'][:]
        self.pressure_mean_np = self.pressure_std['mean_'][:]
        self.rho_mean_np = self.rho_std['mean_'][:]
        self.xwind_mean_np = self.xwind_std['mean_'][:]
        self.ywind_mean_np = self.ywind_std['mean_'][:]
        self.zwind_mean_np = self.zwind_std['mean_'][:]

        # self.qphys_stdscale_np = self.qphys_normaliser_std['scale_'][:]
        # self.tphys_stdscale_np = self.tphys_normaliser_std['scale_'][:]
        self.q_stdscale_np = self.q_normaliser_std['scale_'][:]
        # self.qv_stdscale_np = self.qv_normaliser_std['scale_'][:]
        self.t_stdscale_np = self.t_normaliser_std['scale_'][:]
        # self.qadv_stdscale_np = self.qadv_normaliser_std['scale_'][:]
        # self.tadv_stdscale_np = self.tadv_normaliser_std['scale_'][:]
        self.sw_toa_stdscale_np = self.sw_toa_normaliser_std['scale_'][:]
        self.upshf_stdscale_np = self.uplhf_normaliser_std['scale_'][:]
        self.uplhf_stdscale_np = self.uplhf_normaliser_std['scale_'][:]
        self.pressure_stdscale_np = self.pressure_std['scale_'][:]
        self.rho_stdscale_np = self.rho_std['scale_'][:]
        self.xwind_stdscale_np = self.xwind_std['scale_'][:]
        self.ywind_stdscale_np = self.ywind_std['scale_'][:]
        self.zwind_stdscale_np = self.zwind_std['scale_'][:]

    def normalise(self, data, mean, scale):
        return (data - mean) / scale
    
    def inverse_transform(self, data, mean, scale):
        return (data * scale) + mean

class Data_IO_validation(object):
    def __init__(self, region, nlevs, dataset_file, normaliser, 
                xvars=['qtot', 'qadv', 'theta', 'theta_adv', 'sw_toa', 'shf', 'lhf', 'p', 'rho', 'xwind', 'ywind', 'zwind'],
                yvars=['qtot_next', 'theta_next'],
                yvars2=['qtot', 'theta'], add_adv=False, no_norm=False, fmin=0, fmax=10):
        self.region = region
        self.nlevs = nlevs
        self.nn_norm = NormalizersData(normaliser)
        self.xdat = []
        self.xmean = []
        self.xstd = []
        self.ydat = []
        self.ymean = []
        self.ystd = []
        self.ydat2 = []
        self.ymean2 = []
        self.ystd2 = []
        self.xvars = xvars
        self.yvars = yvars
        self.yvars2 = yvars2
        self.fmin = fmin
        self.fmax = fmax
        # self.xvars = ['qtot', 'qadv', 'theta', 'theta_adv', 'sw_toa', 'shf', 'lhf']
        # self.yvars = ['qphys', 'theta_phys']
        # self.xvars = ['qtot', 'theta', 'sw_toa', 'shf', 'lhf']
        # self.yvars = ['qtot_next', 'theta_next']
        # self.yvars = ['qtot_next']
        # self.yvars2 = ['qphys', 'theta_phys']
        # self.yvars2 = ['qphys']

        self.xdata_idx = []
        self.ydata_idx = []
        self.ydata_idx2 = []
        self.add_adv = add_adv
        self.dat_type = "test"
        self.no_norm = no_norm

        print("Reading dataset file: {0}".format(dataset_file))
        dataset=h5py.File(dataset_file,'r')
        # self.qv_test = dataset["specific_humidity_test"]
        self.q_tot_test = dataset["q_tot_test"]
        # self.q_tot_diff_test = dataset["q_tot_diff_test"]
        # self.q_tot_adv_test = dataset["q_adv_test"]
        self.theta_test = dataset["air_potential_temperature_test"]
        # self.theta_diff_test = dataset["air_potential_temperature_diff_test"]
        # self.theta_adv_test = dataset["t_adv_test"]
        self.sw_toa_test = dataset["toa_incoming_shortwave_flux_test"]
        self.shf_test = dataset["surface_upward_sensible_heat_flux_test"]
        self.lhf_test = dataset["surface_upward_latent_heat_flux_test"]
        # self.theta_phys_test = dataset["t_phys_test"]
        # self.qphys_test = dataset["q_phys_test"]
        self.p_test = dataset['air_pressure_test']
        self.rho_test = dataset['m01s00i253_test']
        self.xwind_test = dataset['x_wind_test']
        self.ywind_test = dataset['y_wind_test']
        self.zwind_test = dataset['upward_air_velocity_test']

        self.npoints = int(self.q_tot_test.shape[0])
        self.norm_slc = slice(self.nlevs)
        self.xdata_and_norm = {
                                'qtot_test':[self.q_tot_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean, self.nn_norm.q_stdscale],
                                # 'qv_test':[self.qv_test[:self.npoints, :self.nlevs], self.nn_norm.qv_mean, self.nn_norm.qv_stdscale],
                                # 'qadv_test':[self.q_tot_adv_test[:self.npoints, :self.nlevs], self.nn_norm.qadv_mean, self.nn_norm.qadv_stdscale],
                                'theta_test':[self.theta_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean, self.nn_norm.t_stdscale],
                                # 'theta_adv_test':[self.theta_adv_test[:self.npoints, :self.nlevs], self.nn_norm.tadv_mean, self.nn_norm.tadv_stdscale],
                                'sw_toa_test':[self.sw_toa_test[:self.npoints], self.nn_norm.sw_toa_mean, self.nn_norm.sw_toa_stdscale],
                                'shf_test':[self.shf_test[:self.npoints], self.nn_norm.upshf_mean, self.nn_norm.upshf_stdscale],
                                'lhf_test':[self.lhf_test[:self.npoints], self.nn_norm.uplhf_mean, self.nn_norm.uplhf_stdscale],
                                'p_test':[self.p_test[:self.npoints, :self.nlevs], self.nn_norm.pressure_mean, self.nn_norm.pressure_stdscale],
                                'rho_test':[self.rho_test[:self.npoints, :self.nlevs], self.nn_norm.rho_mean, self.nn_norm.rho_stdscale],
                                'xwind_test':[self.xwind_test[:self.npoints, :self.nlevs], self.nn_norm.xwind_mean, self.nn_norm.xwind_stdscale],
                                'ywind_test':[self.ywind_test[:self.npoints, :self.nlevs], self.nn_norm.ywind_mean, self.nn_norm.ywind_stdscale],
                                'zwind_test':[self.zwind_test[:self.npoints, :self.nlevs], self.nn_norm.zwind_mean, self.nn_norm.zwind_stdscale],
                                }
        self.ydata_and_norm = {
                                # 'qphys_test':[self.qphys_test[:self.npoints, :self.nlevs], self.nn_norm.qphys_mean, self.nn_norm.qphys_stdscale],
                                'qtot_test':[self.q_tot_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean, self.nn_norm.q_stdscale],
                                # 'qv_test':[self.qv_test[:self.npoints, :self.nlevs], self.nn_norm.qv_mean, self.nn_norm.qv_stdscale],
                                'theta_test':[self.theta_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean, self.nn_norm.t_stdscale],
                                # 'qphys_test':[self.qphys_test[:self.npoints, :3], self.nn_norm.qphys_mean[norm_slc3], self.nn_norm.qphys_stdscale[norm_slc3]],
                                # 'theta_phys_test':[self.theta_phys_test[:self.npoints, :self.nlevs], self.nn_norm.tphys_mean, self.nn_norm.tphys_stdscale],
                                # 'qtot_next_test':[self.q_tot_test[:self.npoints, :self.nlevs]+self.q_tot_adv_test[:self.npoints, :self.nlevs]+self.qphys_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean, self.nn_norm.q_stdscale],
                                # 'qtot_next_test':[self.q_tot_test[:self.npoints-1, :self.nlevs]+self.q_tot_diff_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean, self.nn_norm.q_stdscale],
                                # 'qtot_next_test':[self.q_tot_test[:self.npoints, :1]+self.q_tot_adv_test[:self.npoints, :1]+self.qphys_test[:self.npoints, :1], self.nn_norm.q_mean[norm_slc1], self.nn_norm.q_stdscale[norm_slc1]],
                                # 'theta_next_test':[self.theta_test[:self.npoints, :self.nlevs]+self.theta_adv_test[:self.npoints, :self.nlevs]+self.theta_phys_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean, self.nn_norm.t_stdscale]
                                # 'theta_next_test':[self.theta_test[:self.npoints-1, :self.nlevs]+self.theta_diff_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean, self.nn_norm.t_stdscale]
                                }
        start_idx = 0
        for x in self.xvars:
            i = x+"_"+self.dat_type
            self.xdat.append(self.xdata_and_norm[i][0])
            mean = self.xdata_and_norm[i][1].reshape(-1)[self.norm_slc]
            std = self.xdata_and_norm[i][2].reshape(-1)[self.norm_slc]
            self.xmean.append(mean)
            self.xstd.append(std)
            end_idx = start_idx + self.xdata_and_norm[i][0].shape[1] #len(self.xdata_and_norm[i][1])
            self.xdata_idx.append((start_idx,end_idx))
            start_idx = end_idx

        start_idx = 0
        for y in self.yvars:
            j = y+"_"+self.dat_type
            self.ydat.append(self.ydata_and_norm[j][0])
            mean = self.ydata_and_norm[j][1].reshape(-1)[self.norm_slc]
            std = self.ydata_and_norm[j][2].reshape(-1)[self.norm_slc]
            self.ymean.append(mean)
            self.ystd.append(std)
            end_idx = start_idx + self.ydata_and_norm[j][0].shape[1]
            self.ydata_idx.append((start_idx,end_idx))
            start_idx = end_idx

        start_idx = 0
        for y2 in self.yvars2:
            k = y2+"_"+self.dat_type
            self.ydat2.append(self.ydata_and_norm[k][0])
            mean = self.ydata_and_norm[k][1].reshape(-1)[self.norm_slc]
            std = self.ydata_and_norm[k][2].reshape(-1)[self.norm_slc]
            self.ymean2.append(mean)
            self.ystd2.append(std)
            end_idx = start_idx + self.ydata_and_norm[k][0].shape[1]
            self.ydata_idx2.append((start_idx,end_idx))
            start_idx = end_idx

        self.xmean = torch.cat(self.xmean)
        self.ymean = torch.cat(self.ymean)
        self.ymean2 = torch.cat(self.ymean2)
        self.xstd = torch.cat(self.xstd)
        self.ystd = torch.cat(self.ystd)
        self.ystd2 = torch.cat(self.ystd2)
        
    
    def split_data(self, indata, xyz='x'):
        """
        Split x,y data into constituents
        """
        split_data = {}
        data_idx = {'x':self.xdata_idx,'y':self.ydata_idx,'y2':self.ydata_idx2}
        xyvars = {'x':self.xvars,'y':self.yvars,'y2':self.yvars2}
        for i,x in enumerate(xyvars[xyz]):
            l,h = data_idx[xyz][i]
            split_data[x] = indata[...,l:h]
        return split_data

    # def split_xdata(self, indata):
    #     split_data = {}
    #     for i,x in enumerate(self.xvars):
    #         l,h = self.xdata_idx[i]
    #         split_data[x] = indata[...,l:h]
    #     return split_data

    # def split_ydata(self, indata):
    #     split_data = {}
    #     for i,y in enumerate(self.yvars):
    #         split_data[y] = indata[...,self.ydata_idx[i][0]:self.ydata_idx[i][1]]
    #     return split_data

    def __get_test_vars__(self):
        """
        Return normalised variables
        """
        x_var_data = {}
        y_var_data = {}
        y_var_data2 = {}

        for x in self.xvars:
            i = x+"_test"
            if self.no_norm:
                x_var_data[i] = torch.from_numpy(self.xdata_and_norm[i][0][:])
            else:
                mean = self.xdata_and_norm[i][1].reshape(-1)[self.norm_slc]
                std = self.xdata_and_norm[i][2].reshape(-1)[self.norm_slc]
                x_var_data[i] = self._transform(torch.from_numpy(self.xdata_and_norm[i][0][:]), mean, std)
        
        for y in self.yvars:
            i = y+"_test"
            if self.no_norm:
                y_var_data[i] = torch.from_numpy(self.ydata_and_norm[i][0][:])
            else:
                mean = self.ydata_and_norm[i][1].reshape(-1)[self.norm_slc]
                std = self.ydata_and_norm[i][2].reshape(-1)[self.norm_slc]
                y_var_data[i] = self._transform(torch.from_numpy(self.ydata_and_norm[i][0][:]), mean, std)
        
        for y2 in self.yvars2:
            i = y2+"_test"
            if self.no_norm:
                y_var_data2[i] = torch.from_numpy(self.ydata_and_norm[i][0][:])
            else:
                mean = self.ydata_and_norm[i][1].reshape(-1)[self.norm_slc]
                std = self.ydata_and_norm[i][2].reshape(-1)[self.norm_slc]
                y_var_data2[i] = self._transform(torch.from_numpy(self.ydata_and_norm[i][0][:]), mean, std)

        return x_var_data, y_var_data, y_var_data2

    # def _transform(self, var, mean, std):
    #     """
    #     Normalise/standardisation   
    #     """
    #     return var.sub(mean).div(std)

    # def _inverse_transform(self, var, mean, std):
    #     """
    #     Inverse the normalisation/standardisation
    #     """
    #     return var.mul(std).add(mean)

    def _transform(self, var, mean, std):
        """
        Normalise/standardisation   
        """
        feature_min = self.fmin
        feature_max = self.fmax
        return var.sub(mean).div(std).mul(feature_max - feature_min).add(feature_min)


    def _inverse_transform(self, var, mean, std):
        """
        Inverse the normalisation/standardisation
        """
        feature_min = self.fmin
        feature_max = self.fmax
        invscaled = var.sub(feature_min).div(feature_max-feature_min)
        invvar = invscaled.mul(std).add(mean)
        return invvar

    def _inverse_transform_split(self, var, mean, std, xyz='x'):
        """
        Inverse the normalisation/standardisation on split data
        """
        inv_transformed_data = {}
        mean_split = self.split_data(mean, xyz=xyz)
        std_split = self.split_data(std, xyz=xyz)
        for v in var:
            # inv_transformed_data[v] = var[v].mul(std_split[v]).add(mean_split[v])
            inv_transformed_data[v] = self._inverse_transform(var[v], mean_split[v], std_split[v])
        return inv_transformed_data

    def get_data(self):
        x_var_data, y_var_data, y_var_data2 = self.__get_test_vars__()
        if self.add_adv == False:
            # x = torch.cat([x_var_data['qtot_test'], x_var_data['qadv_test'], x_var_data['theta_test'], x_var_data['theta_adv_test'], x_var_data['sw_toa_test'], x_var_data['shf_test'], x_var_data['lhf_test']], dim=1)
            # y = torch.cat([y_var_data['qphys_test'], y_var_data['theta_phys_test']], dim=1)
            x = torch.cat([x_var_data[k+'_test'] for k in self.xvars], dim=1)
            y = torch.cat([y_var_data[k+'_test'] for k in self.yvars], dim=1)
            y2 = torch.cat([y_var_data2[k+'_test'] for k in self.yvars2], dim=1)
        elif self.add_adv == True:
            x = torch.cat([x_var_data['qtot_test'] - x_var_data['qadv_test'], x_var_data['theta_test'] + x_var_data['theta_adv_test'], x_var_data['sw_toa_test'], x_var_data['shf_test'], x_var_data['lhf_test']], dim=1)
            y = torch.cat([y_var_data['qphys_test'], y_var_data['theta_phys_test']], dim=1)
            y2 = torch.cat([y_var_data2['qphys_test'], y_var_data2['theta_phys_test']], dim=1)

        return (x,y,y2, self.xmean, self.xstd, self.ymean, self.ystd, self.ymean2, self.ystd2)

class Data_IO_validation_diff(object):
    def __init__(self, region, nlevs, dataset_file, normaliser, 
                xvars=['qtot', 'qadv', 'theta', 'theta_adv', 'sw_toa', 'shf', 'lhf', 'p', 'rho', 'xwind', 'ywind', 'zwind'],
                yvars=['qtot_next', 'theta_next'],
                yvars2=['qtot_next', 'theta_next'], add_adv=False, no_norm=False):
        self.region = region
        self.nlevs = nlevs
        self.nn_norm = NormalizersData(normaliser)
        self.xdat = []
        self.xmean = []
        self.xstd = []
        self.ydat = []
        self.ymean = []
        self.ystd = []
        self.ydat2 = []
        self.ymean2 = []
        self.ystd2 = []
        self.xvars = xvars
        self.yvars = yvars
        self.yvars2 = yvars2
        # self.xvars = ['qtot', 'qadv', 'theta', 'theta_adv', 'sw_toa', 'shf', 'lhf']
        # self.yvars = ['qphys', 'theta_phys']
        # self.xvars = ['qtot', 'theta', 'sw_toa', 'shf', 'lhf']
        # self.yvars = ['qtot_next', 'theta_next']
        # self.yvars = ['qtot_next']
        # self.yvars2 = ['qphys', 'theta_phys']
        # self.yvars2 = ['qphys']

        self.xdata_idx = []
        self.ydata_idx = []
        self.ydata_idx2 = []
        self.add_adv = add_adv
        self.dat_type = "test"
        self.no_norm = no_norm

        print("Reading dataset file: {0}".format(dataset_file))
        dataset=h5py.File(dataset_file,'r')
        self.q_tot_test = dataset["q_tot_test"]
        self.q_tot_adv_test = dataset["q_adv_test"]
        self.theta_test = dataset["air_potential_temperature_test"]
        self.theta_adv_test = dataset["t_adv_test"]
        self.sw_toa_test = dataset["toa_incoming_shortwave_flux_test"]
        self.shf_test = dataset["surface_upward_sensible_heat_flux_test"]
        self.lhf_test = dataset["surface_upward_latent_heat_flux_test"]
        # self.theta_phys_test = dataset["t_phys_test"]
        # self.qphys_test = dataset["q_phys_test"]
        self.p_test = dataset['air_pressure_test']
        self.rho_test = dataset['m01s00i253_test']
        self.xwind_test = dataset['x_wind_test']
        self.ywind_test = dataset['y_wind_test']
        self.zwind_test = dataset['upward_air_velocity_test']

        self.npoints = int(self.q_tot_test.shape[0])
        self.norm_slc = slice(self.nlevs)
        self.xdata_and_norm = {
                                'qtot_test':[self.q_tot_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean, self.nn_norm.q_stdscale],
                                'qadv_test':[self.q_tot_adv_test[:self.npoints, :self.nlevs], self.nn_norm.qadv_mean, self.nn_norm.qadv_stdscale],
                                'theta_test':[self.theta_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean, self.nn_norm.t_stdscale],
                                'theta_adv_test':[self.theta_adv_test[:self.npoints, :self.nlevs], self.nn_norm.tadv_mean, self.nn_norm.tadv_stdscale],
                                'sw_toa_test':[self.sw_toa_test[:self.npoints], self.nn_norm.sw_toa_mean, self.nn_norm.sw_toa_stdscale],
                                'shf_test':[self.shf_test[:self.npoints], self.nn_norm.upshf_mean, self.nn_norm.upshf_stdscale],
                                'lhf_test':[self.lhf_test[:self.npoints], self.nn_norm.uplhf_mean, self.nn_norm.uplhf_stdscale],
                                'p_test':[self.p_test[:self.npoints, :self.nlevs], self.nn_norm.pressure_mean, self.nn_norm.pressure_stdscale],
                                'rho_test':[self.rho_test[:self.npoints, :self.nlevs], self.nn_norm.rho_mean, self.nn_norm.rho_stdscale],
                                'xwind_test':[self.xwind_test[:self.npoints, :self.nlevs], self.nn_norm.xwind_mean, self.nn_norm.xwind_stdscale],
                                'ywind_test':[self.ywind_test[:self.npoints, :self.nlevs], self.nn_norm.ywind_mean, self.nn_norm.ywind_stdscale],
                                'zwind_test':[self.zwind_test[:self.npoints, :self.nlevs], self.nn_norm.zwind_mean, self.nn_norm.zwind_stdscale],
                                }
        self.ydata_and_norm = {
                                # 'qphys_test':[self.qphys_test[:self.npoints, :self.nlevs], self.nn_norm.qphys_mean, self.nn_norm.qphys_stdscale],
                                'qtot_test':[self.q_tot_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean, self.nn_norm.q_stdscale],
                                'qtot_next_test':[self.q_tot_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean, self.nn_norm.q_stdscale],
                                'theta_test':[self.theta_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean, self.nn_norm.t_stdscale],
                                'theta_next_test':[self.theta_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean, self.nn_norm.t_stdscale],
                                # 'qphys_test':[self.qphys_test[:self.npoints, :3], self.nn_norm.qphys_mean[norm_slc3], self.nn_norm.qphys_stdscale[norm_slc3]],
                                # 'theta_phys_test':[self.theta_phys_test[:self.npoints, :self.nlevs], self.nn_norm.tphys_mean, self.nn_norm.tphys_stdscale],
                                # 'qtot_next_test':[self.q_tot_test[:self.npoints, :self.nlevs]+self.q_tot_adv_test[:self.npoints, :self.nlevs]+self.qphys_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean, self.nn_norm.q_stdscale],
                                # 'qtot_next_test':[self.q_tot_test[:self.npoints-1, :self.nlevs]+self.q_tot_diff_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean, self.nn_norm.q_stdscale],
                                # 'qtot_next_test':[self.q_tot_test[:self.npoints, :1]+self.q_tot_adv_test[:self.npoints, :1]+self.qphys_test[:self.npoints, :1], self.nn_norm.q_mean[norm_slc1], self.nn_norm.q_stdscale[norm_slc1]],
                                # 'theta_next_test':[self.theta_test[:self.npoints, :self.nlevs]+self.theta_adv_test[:self.npoints, :self.nlevs]+self.theta_phys_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean, self.nn_norm.t_stdscale]
                                # 'theta_next_test':[self.theta_test[:self.npoints-1, :self.nlevs]+self.theta_diff_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean, self.nn_norm.t_stdscale]
                                }
        start_idx = 0
        for x in self.xvars:
            i = x+"_"+self.dat_type
            self.xdat.append(self.xdata_and_norm[i][0])
            mean = self.xdata_and_norm[i][1].reshape(-1)[self.norm_slc]
            std = self.xdata_and_norm[i][2].reshape(-1)[self.norm_slc]
            self.xmean.append(mean)
            self.xstd.append(std)
            end_idx = start_idx + self.xdata_and_norm[i][0].shape[1] #len(self.xdata_and_norm[i][1])
            self.xdata_idx.append((start_idx,end_idx))
            start_idx = end_idx

        start_idx = 0
        for y in self.yvars:
            j = y+"_"+self.dat_type
            self.ydat.append(self.ydata_and_norm[j][0])
            mean = self.ydata_and_norm[j][1].reshape(-1)[self.norm_slc]
            std = self.ydata_and_norm[j][2].reshape(-1)[self.norm_slc]
            self.ymean.append(mean)
            self.ystd.append(std)
            end_idx = start_idx + self.ydata_and_norm[j][0].shape[1]
            self.ydata_idx.append((start_idx,end_idx))
            start_idx = end_idx

        start_idx = 0
        for y2 in self.yvars2:
            k = y2+"_"+self.dat_type
            self.ydat2.append(self.ydata_and_norm[k][0])
            mean = self.ydata_and_norm[k][1].reshape(-1)[self.norm_slc]
            std = self.ydata_and_norm[k][2].reshape(-1)[self.norm_slc]
            self.ymean2.append(mean)
            self.ystd2.append(std)
            end_idx = start_idx + self.ydata_and_norm[k][0].shape[1]
            self.ydata_idx2.append((start_idx,end_idx))
            start_idx = end_idx

        self.xmean = torch.cat(self.xmean)
        self.ymean = torch.cat(self.ymean)
        self.ymean2 = torch.cat(self.ymean2)
        self.xstd = torch.cat(self.xstd)
        self.ystd = torch.cat(self.ystd)
        self.ystd2 = torch.cat(self.ystd2)
        
    
    def split_data(self, indata, xyz='x'):
        """
        Split x,y data into constituents
        """
        split_data = {}
        data_idx = {'x':self.xdata_idx,'y':self.ydata_idx,'y2':self.ydata_idx2}
        xyvars = {'x':self.xvars,'y':self.yvars,'y2':self.yvars2}
        for i,x in enumerate(xyvars[xyz]):
            l,h = data_idx[xyz][i]
            split_data[x] = indata[...,l:h]
        return split_data

    # def split_xdata(self, indata):
    #     split_data = {}
    #     for i,x in enumerate(self.xvars):
    #         l,h = self.xdata_idx[i]
    #         split_data[x] = indata[...,l:h]
    #     return split_data

    # def split_ydata(self, indata):
    #     split_data = {}
    #     for i,y in enumerate(self.yvars):
    #         split_data[y] = indata[...,self.ydata_idx[i][0]:self.ydata_idx[i][1]]
    #     return split_data

    def __get_test_vars__(self):
        """
        Return normalised variables
        """
        x_var_data = {}
        y_var_data = {}
        y_var_data2 = {}

        for x in self.xvars:
            i = x+"_test"
            if self.no_norm:
                x_var_data[i] = torch.from_numpy(self.xdata_and_norm[i][0][:])
            else:
                mean = self.xdata_and_norm[i][1].reshape(-1)[self.norm_slc]
                std = self.xdata_and_norm[i][2].reshape(-1)[self.norm_slc]
                x_var_data[i] = self._transform(torch.from_numpy(self.xdata_and_norm[i][0][:]), mean, std)
        
        for y in self.yvars:
            i = y+"_test"
            if self.no_norm:
                y_var_data[i] = torch.from_numpy(self.ydata_and_norm[i][0][:])
            else:
                mean = self.ydata_and_norm[i][1].reshape(-1)[self.norm_slc]
                std = self.ydata_and_norm[i][2].reshape(-1)[self.norm_slc]
                y_var_data[i] = self._transform(torch.from_numpy(self.ydata_and_norm[i][0][:]), mean, std)
        
        for y2 in self.yvars2:
            i = y2+"_test"
            if self.no_norm:
                y_var_data2[i] = torch.from_numpy(self.ydata_and_norm[i][0][:])
            else:
                mean = self.ydata_and_norm[i][1].reshape(-1)[self.norm_slc]
                std = self.ydata_and_norm[i][2].reshape(-1)[self.norm_slc]
                y_var_data2[i] = self._transform(torch.from_numpy(self.ydata_and_norm[i][0][:]), mean, std)

        return x_var_data, y_var_data, y_var_data2

    def _transform(self, var, mean, std, feature_min=0, feature_max=1):
        """
        Normalise/standardisation   
        """
        return var.sub(mean).div(std).mul(feature_max - feature_min).add(feature_min)


    def _inverse_transform(self, var, mean, std, feature_min=0, feature_max=1):
        """
        Inverse the normalisation/standardisation
        """
        invscaled = var.sub(feature_min).div(feature_max-feature_min)
        invvar = invscaled.mul(std).add(mean)
        return invvar

    def _inverse_transform_split(self, var, mean, std, xyz='x'):
        """
        Inverse the normalisation/standardisation on split data
        """
        inv_transformed_data = {}
        mean_split = self.split_data(mean, xyz=xyz)
        std_split = self.split_data(std, xyz=xyz)
        for v in var:
            inv_transformed_data[v] = var[v].mul(std_split[v]).add(mean_split[v])
        return inv_transformed_data

    def get_data(self):
        x_var_data, y_var_data, y_var_data2 = self.__get_test_vars__()
        if self.add_adv == False:
            # x = torch.cat([x_var_data['qtot_test'], x_var_data['qadv_test'], x_var_data['theta_test'], x_var_data['theta_adv_test'], x_var_data['sw_toa_test'], x_var_data['shf_test'], x_var_data['lhf_test']], dim=1)
            # y = torch.cat([y_var_data['qphys_test'], y_var_data['theta_phys_test']], dim=1)
            x = torch.cat([x_var_data[k+'_test'] for k in self.xvars], dim=1)
            y = torch.cat([y_var_data[k+'_test'] for k in self.yvars], dim=1)
            y2 = torch.cat([y_var_data2[k+'_test'] for k in self.yvars2], dim=1)
        elif self.add_adv == True:
            x = torch.cat([x_var_data['qtot_test'] - x_var_data['qadv_test'], x_var_data['theta_test'] + x_var_data['theta_adv_test'], x_var_data['sw_toa_test'], x_var_data['shf_test'], x_var_data['lhf_test']], dim=1)
            y = torch.cat([y_var_data['qphys_test'], y_var_data['theta_phys_test']], dim=1)
            y2 = torch.cat([y_var_data2['qphys_test'], y_var_data2['theta_phys_test']], dim=1)

        return (x,y,y2, self.xmean, self.xstd, self.ymean, self.ystd, self.ymean2, self.ystd2)
        

class Data_IO_validation_x2(object):
    def __init__(self, region, nlevs, dataset_file, normaliser, 
                xvars=['qtot', 'theta', 'sw_toa', 'shf', 'lhf', 'p', 'rho', 'xwind', 'ywind', 'zwind'],
                yvars=['qtot_next', 'theta_next'],
                xvars2=['qadv', 'theta_adv'], no_norm=False):
        self.region = region
        self.nlevs = nlevs
        self.nn_norm = NormalizersData(normaliser)
        self.xdat = []
        self.xmean = []
        self.xstd = []
        self.ydat = []
        self.ymean = []
        self.ystd = []
        self.xdat2 = []
        self.xmean2 = []
        self.xstd2 = []
        self.xvars = xvars
        self.yvars = yvars
        self.xvars2 = xvars2
        # self.xvars = ['qtot', 'qadv', 'theta', 'theta_adv', 'sw_toa', 'shf', 'lhf']
        # self.yvars = ['qphys', 'theta_phys']
        # self.xvars = ['qtot', 'theta', 'sw_toa', 'shf', 'lhf']
        # self.yvars = ['qtot_next', 'theta_next']
        # self.yvars = ['qtot_next']
        # self.yvars2 = ['qphys', 'theta_phys']
        # self.yvars2 = ['qphys']

        self.xdata_idx = []
        self.ydata_idx = []
        self.xdata_idx2 = []
        self.dat_type = "test"
        self.no_norm = no_norm

        print("Reading dataset file: {0}".format(dataset_file))
        dataset=h5py.File(dataset_file,'r')
        self.q_tot_test = dataset["q_tot_test"]
        # self.q_tot_diff_test = dataset["q_tot_diff_test"]
        self.q_tot_adv_test = dataset["q_adv_test"]
        self.theta_test = dataset["air_potential_temperature_test"]
        # self.theta_diff_test = dataset["air_potential_temperature_diff_test"]
        self.theta_adv_test = dataset["t_adv_test"]
        self.sw_toa_test = dataset["toa_incoming_shortwave_flux_test"]
        self.shf_test = dataset["surface_upward_sensible_heat_flux_test"]
        self.lhf_test = dataset["surface_upward_latent_heat_flux_test"]
        self.theta_phys_test = dataset["t_phys_test"]
        self.qphys_test = dataset["q_phys_test"]
        self.p_test = dataset['air_pressure_test']
        self.rho_test = dataset['m01s00i253_test']
        self.xwind_test = dataset['x_wind_test']
        self.ywind_test = dataset['y_wind_test']
        self.zwind_test = dataset['upward_air_velocity_test']

        self.npoints = int(self.q_tot_test.shape[0])
        self.norm_slc = slice(self.nlevs)
        self.xdata_and_norm = {
                                'qtot_test':[self.q_tot_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean, self.nn_norm.q_stdscale],
                                'qadv_test':[self.q_tot_adv_test[:self.npoints, :self.nlevs], self.nn_norm.qadv_mean, self.nn_norm.qadv_stdscale],
                                'theta_test':[self.theta_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean, self.nn_norm.t_stdscale],
                                'theta_adv_test':[self.theta_adv_test[:self.npoints, :self.nlevs], self.nn_norm.tadv_mean, self.nn_norm.tadv_stdscale],
                                'sw_toa_test':[self.sw_toa_test[:self.npoints], self.nn_norm.sw_toa_mean, self.nn_norm.sw_toa_stdscale],
                                'shf_test':[self.shf_test[:self.npoints], self.nn_norm.upshf_mean, self.nn_norm.upshf_stdscale],
                                'lhf_test':[self.lhf_test[:self.npoints], self.nn_norm.uplhf_mean, self.nn_norm.uplhf_stdscale],
                                'p_test':[self.p_test[:self.npoints, :self.nlevs], self.nn_norm.pressure_mean, self.nn_norm.pressure_stdscale],
                                'rho_test':[self.rho_test[:self.npoints, :self.nlevs], self.nn_norm.rho_mean, self.nn_norm.rho_stdscale],
                                'xwind_test':[self.xwind_test[:self.npoints, :self.nlevs], self.nn_norm.xwind_mean, self.nn_norm.xwind_stdscale],
                                'ywind_test':[self.ywind_test[:self.npoints, :self.nlevs], self.nn_norm.ywind_mean, self.nn_norm.ywind_stdscale],
                                'zwind_test':[self.zwind_test[:self.npoints, :self.nlevs], self.nn_norm.zwind_mean, self.nn_norm.zwind_stdscale],
                                }
        self.ydata_and_norm = {
                                'qphys_test':[self.qphys_test[:self.npoints, :self.nlevs], self.nn_norm.qphys_mean, self.nn_norm.qphys_stdscale],
                                'qtot_test':[self.q_tot_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean, self.nn_norm.q_stdscale],
                                'theta_test':[self.theta_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean, self.nn_norm.t_stdscale],
                                # 'qphys_test':[self.qphys_test[:self.npoints, :3], self.nn_norm.qphys_mean[norm_slc3], self.nn_norm.qphys_stdscale[norm_slc3]],
                                'theta_phys_test':[self.theta_phys_test[:self.npoints, :self.nlevs], self.nn_norm.tphys_mean, self.nn_norm.tphys_stdscale],
                                # 'qtot_next_test':[self.q_tot_test[:self.npoints, :self.nlevs]+self.q_tot_adv_test[:self.npoints, :self.nlevs]+self.qphys_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean, self.nn_norm.q_stdscale],
                                # 'qtot_next_test':[self.q_tot_test[:self.npoints-1, :self.nlevs]+self.q_tot_diff_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean, self.nn_norm.q_stdscale],
                                # 'qtot_next_test':[self.q_tot_test[:self.npoints, :1]+self.q_tot_adv_test[:self.npoints, :1]+self.qphys_test[:self.npoints, :1], self.nn_norm.q_mean[norm_slc1], self.nn_norm.q_stdscale[norm_slc1]],
                                # 'theta_next_test':[self.theta_test[:self.npoints, :self.nlevs]+self.theta_adv_test[:self.npoints, :self.nlevs]+self.theta_phys_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean, self.nn_norm.t_stdscale]
                                # 'theta_next_test':[self.theta_test[:self.npoints-1, :self.nlevs]+self.theta_diff_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean, self.nn_norm.t_stdscale]
                                }
        start_idx = 0
        for x in self.xvars:
            i = x+"_"+self.dat_type
            self.xdat.append(self.xdata_and_norm[i][0])
            mean = self.xdata_and_norm[i][1].reshape(-1)[self.norm_slc]
            std = self.xdata_and_norm[i][2].reshape(-1)[self.norm_slc]
            self.xmean.append(mean)
            self.xstd.append(std)
            end_idx = start_idx + self.xdata_and_norm[i][0].shape[1] #len(self.xdata_and_norm[i][1])
            self.xdata_idx.append((start_idx,end_idx))
            start_idx = end_idx

        start_idx = 0
        for y in self.yvars:
            j = y+"_"+self.dat_type
            self.ydat.append(self.ydata_and_norm[j][0])
            mean = self.ydata_and_norm[j][1].reshape(-1)[self.norm_slc]
            std = self.ydata_and_norm[j][2].reshape(-1)[self.norm_slc]
            self.ymean.append(mean)
            self.ystd.append(std)
            end_idx = start_idx + self.ydata_and_norm[j][0].shape[1]
            self.ydata_idx.append((start_idx,end_idx))
            start_idx = end_idx

        start_idx = 0
        for x2 in self.xvars2:
            k = x2+"_"+self.dat_type
            self.xdat2.append(self.xdata_and_norm[k][0])
            mean = self.xdata_and_norm[k][1].reshape(-1)[self.norm_slc]
            std = self.xdata_and_norm[k][2].reshape(-1)[self.norm_slc]
            self.xmean2.append(mean)
            self.xstd2.append(std)
            end_idx = start_idx + self.xdata_and_norm[k][0].shape[1]
            self.xdata_idx2.append((start_idx,end_idx))
            start_idx = end_idx

        self.xmean = torch.cat(self.xmean)
        self.ymean = torch.cat(self.ymean)
        self.xmean2 = torch.cat(self.xmean2)
        self.xstd = torch.cat(self.xstd)
        self.ystd = torch.cat(self.ystd)
        self.xstd2 = torch.cat(self.xstd2)
        
    
    def split_data(self, indata, xyz='x'):
        """
        Split x,y data into constituents
        """
        split_data = {}
        data_idx = {'x':self.xdata_idx,'y':self.ydata_idx,'x2':self.xdata_idx2}
        xyvars = {'x':self.xvars,'y':self.yvars,'x2':self.xvars2}
        for i,x in enumerate(xyvars[xyz]):
            l,h = data_idx[xyz][i]
            split_data[x] = indata[...,l:h]
        return split_data

    # def split_xdata(self, indata):
    #     split_data = {}
    #     for i,x in enumerate(self.xvars):
    #         l,h = self.xdata_idx[i]
    #         split_data[x] = indata[...,l:h]
    #     return split_data

    # def split_ydata(self, indata):
    #     split_data = {}
    #     for i,y in enumerate(self.yvars):
    #         split_data[y] = indata[...,self.ydata_idx[i][0]:self.ydata_idx[i][1]]
    #     return split_data

    def __get_test_vars__(self):
        """
        Return normalised variables
        """
        x_var_data = {}
        y_var_data = {}
        x_var_data2 = {}

        for x in self.xvars:
            i = x+"_test"
            if self.no_norm:
                x_var_data[i] = torch.from_numpy(self.xdata_and_norm[i][0][:])
            else:
                mean = self.xdata_and_norm[i][1].reshape(-1)[self.norm_slc]
                std = self.xdata_and_norm[i][2].reshape(-1)[self.norm_slc]
                x_var_data[i] = self._transform(torch.from_numpy(self.xdata_and_norm[i][0][:]), mean, std)
        
        for y in self.yvars:
            i = y+"_test"
            if self.no_norm:
                y_var_data[i] = torch.from_numpy(self.ydata_and_norm[i][0][:])
            else:
                mean = self.ydata_and_norm[i][1].reshape(-1)[self.norm_slc]
                std = self.ydata_and_norm[i][2].reshape(-1)[self.norm_slc]
                y_var_data[i] = self._transform(torch.from_numpy(self.ydata_and_norm[i][0][:]), mean, std)
        
        for x2 in self.xvars2:
            i = x2+"_test"
            if self.no_norm:
                x_var_data2[i] = torch.from_numpy(self.xdata_and_norm[i][0][:])
            else:
                mean = self.xdata_and_norm[i][1].reshape(-1)[self.norm_slc]
                std = self.xdata_and_norm[i][2].reshape(-1)[self.norm_slc]
                x_var_data2[i] = self._transform(torch.from_numpy(self.xdata_and_norm[i][0][:]), mean, std)

        return x_var_data, y_var_data, x_var_data2

    def _transform(self, var, mean, std):
        """
        Normalise/standardisation   
        """
        return var.sub(mean).div(std)

    def _inverse_transform(self, var, mean, std):
        """
        Inverse the normalisation/standardisation
        """
        return var.mul(std).add(mean)

    def _inverse_transform_split(self, var, mean, std, xyz='x'):
        """
        Inverse the normalisation/standardisation on split data
        """
        inv_transformed_data = {}
        mean_split = self.split_data(mean, xyz=xyz)
        std_split = self.split_data(std, xyz=xyz)
        for v in var:
            inv_transformed_data[v] = var[v].mul(std_split[v]).add(mean_split[v])
        return inv_transformed_data

    def get_data(self):
        x_var_data, y_var_data, x_var_data2 = self.__get_test_vars__()
        x = torch.cat([x_var_data[k+'_test'] for k in self.xvars], dim=1)
        y = torch.cat([y_var_data[k+'_test'] for k in self.yvars], dim=1)
        x2 = torch.cat([x_var_data2[k+'_test'] for k in self.xvars2], dim=1)
        return (x,y,x2, self.xmean, self.xstd, self.ymean, self.ystd, self.xmean2, self.xstd2)

        # x = torch.cat([torch.from_numpy(d) for d in self.xdat], dim=1)
        # y = torch.cat([torch.from_numpy(d) for d in self.ydat], dim=1)
        # x_ = self._transform(x, self.xmean, self.xstd)
        # y_ = self._transform(y, self.ymean, self.ystd)
        # return (x_,y_, self.xmean, self.xstd, self.ymean, self.ystd)

class Data_IO_validation_CNN(object):
    def __init__(self, region, nlevs, dataset_file, normaliser, 
                xvars=['qtot', 'qadv', 'theta', 'theta_adv'],
                xvars2=['sw_toa', 'shf', 'lhf'],
                yvars=['qtot_next', 'theta_next'],
                yvars2=['qphys', 'theta_phys']):
        self.region = region
        self.nlevs = nlevs
        self.nn_norm = NormalizersData(normaliser)
        self.xdat = []
        self.xdat2 = []
        self.xmean = []
        self.xmean2 = []
        self.xstd = []
        self.xstd2 = []
        self.ydat = []
        self.ymean = []
        self.ystd = []
        self.ydat2 = []
        self.ymean2 = []
        self.ystd2 = []
        self.xvars = xvars
        self.xvars2 = xvars2
        self.yvars = yvars
        self.yvars2 = yvars2

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
        self.npoints = int(self.q_tot_test.shape[0])
        self.xdata_and_norm = {
                                'qtot_test':[self.q_tot_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean[0,:self.nlevs], self.nn_norm.q_stdscale[0,:self.nlevs]],
                                'qadv_test':[self.q_tot_adv_test[:self.npoints, :self.nlevs], self.nn_norm.qadv_mean[0,:self.nlevs], self.nn_norm.qadv_stdscale[0,:self.nlevs]],
                                'theta_test':[self.theta_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean[0,:self.nlevs], self.nn_norm.t_stdscale[0,:self.nlevs]],
                                'theta_adv_test':[self.theta_adv_test[:self.npoints, :self.nlevs], self.nn_norm.tadv_mean[0,:self.nlevs], self.nn_norm.tadv_stdscale[0,:self.nlevs]],
                                'sw_toa_test':[self.sw_toa_test[:self.npoints], self.nn_norm.sw_toa_mean, self.nn_norm.sw_toa_stdscale],
                                'shf_test':[self.shf_test[:self.npoints], self.nn_norm.upshf_mean, self.nn_norm.upshf_stdscale],
                                'lhf_test':[self.lhf_test[:self.npoints], self.nn_norm.uplhf_mean, self.nn_norm.uplhf_stdscale]
                                }
        self.ydata_and_norm = {
                                'qphys_test':[self.qphys_test[:self.npoints, :self.nlevs], self.nn_norm.qphys_mean[0,:self.nlevs], self.nn_norm.qphys_stdscale[0,:self.nlevs]],
                                # 'qphys_test':[self.qphys_test[:self.npoints, :3], self.nn_norm.qphys_mean[0,:3], self.nn_norm.qphys_stdscale[0,:3]],
                                'theta_phys_test':[self.theta_phys_test[:self.npoints, :self.nlevs], self.nn_norm.tphys_mean[0,:self.nlevs], self.nn_norm.tphys_stdscale[0,:self.nlevs]],
                                'qtot_next_test':[self.q_tot_test[:self.npoints, :self.nlevs]+self.q_tot_adv_test[:self.npoints, :self.nlevs]+self.qphys_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean[0,:self.nlevs], self.nn_norm.q_stdscale[0,:self.nlevs]],
                                # 'qtot_next_test':[self.q_tot_test[:self.npoints, :1]+self.q_tot_adv_test[:self.npoints, :1]+self.qphys_test[:self.npoints, :1], self.nn_norm.q_mean[0,:1], self.nn_norm.q_stdscale[0,:1]],
                                'theta_next_test':[self.theta_test[:self.npoints, :self.nlevs]+self.theta_adv_test[:self.npoints, :self.nlevs]+self.theta_phys_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean[0,:self.nlevs], self.nn_norm.t_stdscale[0,:self.nlevs]]
                                }
        for x in self.xvars:
            i = x+"_test"
            self.xdat.append(self.xdata_and_norm[i][0])
            self.xmean.append(self.xdata_and_norm[i][1])
            self.xstd.append(self.xdata_and_norm[i][2])

        for x2 in self.xvars2:
            i = x2+"_test"
            self.xdat2.append(self.xdata_and_norm[i][0])
            self.xmean2.append(self.xdata_and_norm[i][1])
            self.xstd2.append(self.xdata_and_norm[i][2])

        for y in self.yvars:
            j = y+"_test"
            self.ydat.append(self.ydata_and_norm[j][0])
            self.ymean.append(self.ydata_and_norm[j][1])
            self.ystd.append(self.ydata_and_norm[j][2])
   
        for y2 in self.yvars2:
            k = y2+"_test"
            self.ydat2.append(self.ydata_and_norm[k][0])
            self.ymean2.append(self.ydata_and_norm[k][1])
            self.ystd2.append(self.ydata_and_norm[k][2])

        self.xmean2 = torch.cat(self.xmean2)
        self.xmean = torch.stack(self.xmean)
        self.ymean = torch.cat(self.ymean)
        self.ymean2 = torch.cat(self.ymean2)
        self.xstd2 = torch.cat(self.xstd2)
        self.xstd = torch.stack(self.xstd)
        self.ystd = torch.cat(self.ystd)
        self.ystd2 = torch.cat(self.ystd2)
        
    
    def __get_test_vars__(self):
        """
        Return normalised variables
        """
        x_var_data = {}
        x_var_data2 = {}
        y_var_data = {}
        y_var_data2 = {}

        for x in self.xvars:
            i = x+"_test"
            x_var_data[i] = self._transform(torch.from_numpy(self.xdata_and_norm[i][0][:]), self.xdata_and_norm[i][1], self.xdata_and_norm[i][2])
        
        for x2 in self.xvars2:
            i = x2+"_test"
            x_var_data2[i] = self._transform(torch.from_numpy(self.xdata_and_norm[i][0][:]), self.xdata_and_norm[i][1], self.xdata_and_norm[i][2])

        for y in self.yvars:
            i = y+"_test"
            y_var_data[i] = self._transform(torch.from_numpy(self.ydata_and_norm[i][0][:]), self.ydata_and_norm[i][1], self.ydata_and_norm[i][2])
        
        for y2 in self.yvars2:
            i = y2+"_test"
            y_var_data2[i] = self._transform(torch.from_numpy(self.ydata_and_norm[i][0][:]), self.ydata_and_norm[i][1], self.ydata_and_norm[i][2])

        return x_var_data, x_var_data2, y_var_data, y_var_data2

    def _transform(self, var, mean, std):
        """
        Normalise/standardisation   
        """
        return var.sub(mean).div(std)

    def _inverse_transform(self, var, mean, std):
        """
        Inverse the normalisation/standardisation
        """
        return var.mul(std).add(mean)

    def get_data(self):
        x_var_data, x_var_data2, y_var_data, y_var_data2 = self.__get_test_vars__()
        x = torch.stack([x_var_data[k+'_test'] for k in self.xvars], dim=1)
        x2 = torch.cat([x_var_data2[k+'_test'] for k in self.xvars2], dim=1)
        y = torch.cat([y_var_data[k+'_test'] for k in self.yvars], dim=1)
        y2 = torch.cat([y_var_data2[k+'_test'] for k in self.yvars2], dim=1)

        return (x,x2,y,y2, self.xmean, self.xstd, self.xmean2, self.xstd2, self.ymean, self.ystd, self.ymean2, self.ystd2)


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, dat_type, nlevs, dataset_file, normaliser, data_frac=1., 
                xvars=['qtot', 'qadv', 'theta', 'theta_adv', 'sw_toa', 'shf', 'lhf', 'p', 'rho', 'xwind', 'ywind', 'zwind'],
                yvars=['qtot_next', 'theta_next'],
                yvars2=['qphys', 'theta_phys'], add_adv=False):
        super().__init__()
        self.dat_type = dat_type
        self.dataset_file = dataset_file
        self.nlevs = nlevs
        self.nn_norm = NormalizersData(normaliser)
        self.xdat = []
        self.xmean = []
        self.xstd = []
        self.ydat = []
        self.ymean = []
        self.ystd = []
        self.ydat2 = []
        self.ymean2 = []
        self.ystd2 = []
        self.xvars = xvars
        self.yvars = yvars
        self.yvars2 = yvars2
        # self.xvars = ['qtot', 'qadv', 'theta', 'theta_adv', 'sw_toa', 'shf', 'lhf']
        # self.yvars = ['qphys', 'theta_phys']
        # self.xvars = ['qtot', 'qadv', 'theta', 'theta_adv', 'sw_toa', 'shf', 'lhf']
        # self.xvars = ['qtot', 'theta', 'sw_toa', 'shf', 'lhf']
        # self.yvars = ['qtot_next', 'theta_next']
        # self.yvars = ['qtot_next']
        # self.yvars2 = ['qphys', 'theta_phys']
        # self.yvars2 = ['qphys']

        self.xdata_idx = []
        self.ydata_idx = []
        self.ydata_idx2 = []
        self.add_adv = add_adv

        if self.dat_type == "train":
            print("Reading dataset file: {0}".format(dataset_file))
            dataset=h5py.File(dataset_file,'r')
            self.q_tot_train = dataset["q_tot_train"]
            self.q_tot_diff_train = dataset["q_tot_diff_train"]
            self.q_tot_adv_train = dataset["q_adv_train"]
            self.theta_train = dataset["air_potential_temperature_train"]
            self.theta_diff_train = dataset["air_potential_temperature_diff_train"]
            self.theta_adv_train = dataset["t_adv_train"]
            self.sw_toa_train = dataset["toa_incoming_shortwave_flux_train"]
            self.shf_train = dataset["surface_upward_sensible_heat_flux_train"]
            self.lhf_train = dataset["surface_upward_latent_heat_flux_train"]
            self.theta_phys_train = dataset["t_phys_train"]
            self.qphys_train = dataset["q_phys_train"]
            self.p_train = dataset['air_pressure_train']
            self.rho_train = dataset['m01s00i253_train']
            self.xwind_train = dataset['x_wind_train']
            self.ywind_train = dataset['y_wind_train']
            self.zwind_train = dataset['upward_air_velocity_train']
            self.npoints = int(self.q_tot_train.shape[0] * data_frac)
            self.xdata_and_norm = {
                                    'qtot_train':[self.q_tot_train[:self.npoints, :self.nlevs], self.nn_norm.q_mean[0,:self.nlevs], self.nn_norm.q_stdscale[0,:self.nlevs]],
                                    'qadv_train':[self.q_tot_adv_train[:self.npoints, :self.nlevs], self.nn_norm.qadv_mean[0,:self.nlevs], self.nn_norm.qadv_stdscale[0,:self.nlevs]],
                                    'theta_train':[self.theta_train[:self.npoints, :self.nlevs], self.nn_norm.t_mean[0,:self.nlevs], self.nn_norm.t_stdscale[0,:self.nlevs]],
                                    'theta_adv_train':[self.theta_adv_train[:self.npoints, :self.nlevs], self.nn_norm.tadv_mean[0,:self.nlevs], self.nn_norm.tadv_stdscale[0,:self.nlevs]],
                                    'sw_toa_train':[self.sw_toa_train[:self.npoints], self.nn_norm.sw_toa_mean[0,:], self.nn_norm.sw_toa_stdscale[0,:]],
                                    'shf_train':[self.shf_train[:self.npoints], self.nn_norm.upshf_mean[0,:], self.nn_norm.upshf_stdscale[0,:]],
                                    'lhf_train':[self.lhf_train[:self.npoints], self.nn_norm.uplhf_mean[0,:], self.nn_norm.uplhf_stdscale[0,:]],
                                    'p_train':[self.p_train[:self.npoints, :self.nlevs], self.nn_norm.pressure_mean[0,:self.nlevs], self.nn_norm.pressure_stdscale[0,:self.nlevs]],
                                    'rho_train':[self.rho_train[:self.npoints, :self.nlevs], self.nn_norm.rho_mean[0,:self.nlevs], self.nn_norm.rho_stdscale[0,:self.nlevs]],
                                    'xwind_train':[self.xwind_train[:self.npoints, :self.nlevs], self.nn_norm.xwind_mean[0,:self.nlevs], self.nn_norm.xwind_stdscale[0,:self.nlevs]],
                                    'ywind_train':[self.ywind_train[:self.npoints, :self.nlevs], self.nn_norm.ywind_mean[0,:self.nlevs], self.nn_norm.ywind_stdscale[0,:self.nlevs]],
                                    'zwind_train':[self.zwind_train[:self.npoints, :self.nlevs], self.nn_norm.zwind_mean[0,:self.nlevs], self.nn_norm.zwind_stdscale[0,:self.nlevs]],
                                    }
            self.ydata_and_norm = {
                                    'qphys_train':[self.qphys_train[:self.npoints, :self.nlevs], self.nn_norm.qphys_mean[0,:self.nlevs], self.nn_norm.qphys_stdscale[0,:self.nlevs]],
                                    # 'qphys_train':[self.qphys_train[:self.npoints, :3], self.nn_norm.qphys_mean[0,:3], self.nn_norm.qphys_stdscale[0,:3]],
                                    'theta_phys_train':[self.theta_phys_train[:self.npoints, :self.nlevs], self.nn_norm.tphys_mean[0,:self.nlevs], self.nn_norm.tphys_stdscale[0,:self.nlevs]],
                                    # 'qtot_next_train':[self.q_tot_train[:self.npoints, :self.nlevs]+self.q_tot_adv_train[:self.npoints, :self.nlevs]+self.qphys_train[:self.npoints, :self.nlevs], self.nn_norm.q_mean[0,:self.nlevs], self.nn_norm.q_stdscale[0,:self.nlevs]],
                                    # 'qtot_next_train':[self.q_tot_train[:self.npoints, :self.nlevs]+self.q_tot_adv_train[:self.npoints, :self.nlevs]+self.qphys_train[:self.npoints, :self.nlevs], self.nn_norm.q_mean[0,:self.nlevs], self.nn_norm.q_stdscale[0,:self.nlevs]],
                                    'qtot_next_train':[self.q_tot_train[:self.npoints, :self.nlevs]+self.q_tot_diff_train[:self.npoints, :self.nlevs], self.nn_norm.q_mean[0,:self.nlevs], self.nn_norm.q_stdscale[0,:self.nlevs]],

                                    # 'qtot_next_train':[self.q_tot_train[:self.npoints, :1]+self.q_tot_adv_train[:self.npoints, :1]+self.qphys_train[:self.npoints, :1], self.nn_norm.q_mean[0,:1], self.nn_norm.q_stdscale[0,:1]],
                                    # 'theta_next_train':[self.theta_train[:self.npoints, :self.nlevs]+self.theta_adv_train[:self.npoints, :self.nlevs]+self.theta_phys_train[:self.npoints, :self.nlevs], self.nn_norm.t_mean[0,:self.nlevs], self.nn_norm.t_stdscale[0,:self.nlevs]]
                                    'theta_next_train':[self.theta_train[:self.npoints, :self.nlevs]+self.theta_diff_train[:self.npoints, :self.nlevs], self.nn_norm.t_mean[0,:self.nlevs], self.nn_norm.t_stdscale[0,:self.nlevs]]


                                    }
            start_idx = 0
            for x in self.xvars:
                i = x+"_train"
                self.xdat.append(self.xdata_and_norm[i][0])
                self.xmean.append(self.xdata_and_norm[i][1])
                self.xstd.append(self.xdata_and_norm[i][2])
                end_idx = start_idx + len(self.xdata_and_norm[i][1])
                self.xdata_idx.append((start_idx,end_idx))
                start_idx = end_idx

            start_idx = 0
            for y in self.yvars:
                j = y+"_train"
                self.ydat.append(self.ydata_and_norm[j][0])
                self.ymean.append(self.ydata_and_norm[j][1])
                self.ystd.append(self.ydata_and_norm[j][2])
                end_idx = start_idx + len(self.ydata_and_norm[j][1])
                self.ydata_idx.append((start_idx,end_idx))
                start_idx = end_idx

            start_idx = 0
            for y2 in self.yvars2:
                k = y2+"_train"
                self.ydat2.append(self.ydata_and_norm[k][0])
                self.ymean2.append(self.ydata_and_norm[k][1])
                self.ystd2.append(self.ydata_and_norm[k][2])
                end_idx = start_idx + len(self.ydata_and_norm[k][1])
                self.ydata_idx2.append((start_idx,end_idx))
                start_idx = end_idx

        elif self.dat_type == "test":
            print("Reading dataset file: {0}".format(dataset_file))
            dataset=h5py.File(dataset_file,'r')
            self.q_tot_test = dataset["q_tot_test"]
            self.q_tot_diff_test = dataset["q_tot_diff_test"]
            self.q_tot_adv_test = dataset["q_adv_test"]
            self.theta_test = dataset["air_potential_temperature_test"]
            self.theta_diff_test = dataset["air_potential_temperature_diff_test"]
            self.theta_adv_test = dataset["t_adv_test"]
            self.sw_toa_test = dataset["toa_incoming_shortwave_flux_test"]
            self.shf_test = dataset["surface_upward_sensible_heat_flux_test"]
            self.lhf_test = dataset["surface_upward_latent_heat_flux_test"]
            self.theta_phys_test = dataset["t_phys_test"]
            self.qphys_test = dataset["q_phys_test"] 
            self.p_test = dataset['air_pressure_test']
            self.rho_test = dataset['m01s00i253_test']
            self.xwind_test = dataset['x_wind_test']
            self.ywind_test = dataset['y_wind_test']
            self.zwind_test = dataset['upward_air_velocity_test']
            self.npoints = int(self.q_tot_test.shape[0] * data_frac)
            self.xdata_and_norm = {
                                    'qtot_test':[self.q_tot_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean[0,:self.nlevs], self.nn_norm.q_stdscale[0,:self.nlevs]],
                                    'qadv_test':[self.q_tot_adv_test[:self.npoints, :self.nlevs], self.nn_norm.qadv_mean[0,:self.nlevs], self.nn_norm.qadv_stdscale[0,:self.nlevs]],
                                    'theta_test':[self.theta_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean[0,:self.nlevs], self.nn_norm.t_stdscale[0,:self.nlevs]],
                                    'theta_adv_test':[self.theta_adv_test[:self.npoints, :self.nlevs], self.nn_norm.tadv_mean[0,:self.nlevs], self.nn_norm.tadv_stdscale[0,:self.nlevs]],
                                    'sw_toa_test':[self.sw_toa_test[:self.npoints], self.nn_norm.sw_toa_mean[0,:], self.nn_norm.sw_toa_stdscale[0,:]],
                                    'shf_test':[self.shf_test[:self.npoints], self.nn_norm.upshf_mean[0,:], self.nn_norm.upshf_stdscale[0,:]],
                                    'lhf_test':[self.lhf_test[:self.npoints], self.nn_norm.uplhf_mean[0,:], self.nn_norm.uplhf_stdscale[0,:]],
                                    'p_test':[self.p_test[:self.npoints, :self.nlevs], self.nn_norm.pressure_mean[0,:self.nlevs], self.nn_norm.pressure_stdscale[0,:self.nlevs]],
                                    'rho_test':[self.rho_test[:self.npoints, :self.nlevs], self.nn_norm.rho_mean[0,:self.nlevs], self.nn_norm.rho_stdscale[0,:self.nlevs]],
                                    'xwind_test':[self.xwind_test[:self.npoints, :self.nlevs], self.nn_norm.xwind_mean[0,:self.nlevs], self.nn_norm.xwind_stdscale[0,:self.nlevs]],
                                    'ywind_test':[self.ywind_test[:self.npoints, :self.nlevs], self.nn_norm.ywind_mean[0,:self.nlevs], self.nn_norm.ywind_stdscale[0,:self.nlevs]],
                                    'zwind_test':[self.zwind_test[:self.npoints, :self.nlevs], self.nn_norm.zwind_mean[0,:self.nlevs], self.nn_norm.zwind_stdscale[0,:self.nlevs]],
                                    }
            self.ydata_and_norm = {
                                    'qphys_test':[self.qphys_test[:self.npoints, :self.nlevs], self.nn_norm.qphys_mean[0,:self.nlevs], self.nn_norm.qphys_stdscale[0,:self.nlevs]],
                                    # 'qphys_test':[self.qphys_test[:self.npoints, :3], self.nn_norm.qphys_mean[0,:3], self.nn_norm.qphys_stdscale[0,:3]],
                                    'theta_phys_test':[self.theta_phys_test[:self.npoints, :self.nlevs], self.nn_norm.tphys_mean[0,:self.nlevs], self.nn_norm.tphys_stdscale[0,:self.nlevs]],
                                    # 'qtot_next_test':[self.q_tot_test[:self.npoints, :self.nlevs]+self.q_tot_adv_test[:self.npoints, :self.nlevs]+self.qphys_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean[0,:self.nlevs], self.nn_norm.q_stdscale[0,:self.nlevs]],
                                    'qtot_next_test':[self.q_tot_test[:self.npoints, :self.nlevs]+self.q_tot_diff_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean[0,:self.nlevs], self.nn_norm.q_stdscale[0,:self.nlevs]],
                                    # 'qtot_next_test':[self.q_tot_test[:self.npoints, :1]+self.q_tot_adv_test[:self.npoints, :1]+self.qphys_test[:self.npoints, :1], self.nn_norm.q_mean[0,:1], self.nn_norm.q_stdscale[0,:1]],
                                    # 'theta_next_test':[self.theta_test[:self.npoints, :self.nlevs]+self.theta_adv_test[:self.npoints, :self.nlevs]+self.theta_phys_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean[0,:self.nlevs], self.nn_norm.t_stdscale[0,:self.nlevs]]
                                    'theta_next_test':[self.theta_test[:self.npoints, :self.nlevs]+self.theta_diff_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean[0,:self.nlevs], self.nn_norm.t_stdscale[0,:self.nlevs]]

                                    }
         
            for x in self.xvars:
                i = x+"_test"
                self.xdat.append(self.xdata_and_norm[i][0])
                self.xmean.append(self.xdata_and_norm[i][1])
                self.xstd.append(self.xdata_and_norm[i][2])

            for y in self.yvars:
                j = y+"_test"
                self.ydat.append(self.ydata_and_norm[j][0])
                self.ymean.append(self.ydata_and_norm[j][1])
                self.ystd.append(self.ydata_and_norm[j][2])
            
            for y2 in self.yvars2:
                k = y2+"_test"
                self.ydat2.append(self.ydata_and_norm[k][0])
                self.ymean2.append(self.ydata_and_norm[k][1])
                self.ystd2.append(self.ydata_and_norm[k][2])

        self.ymean = torch.cat(self.ymean)
        self.ystd = torch.cat(self.ystd)
        self.ymean2 = torch.cat(self.ymean2)
        self.ystd2 = torch.cat(self.ystd2)
        self.xmean = torch.cat(self.xmean)
        self.xstd = torch.cat(self.xstd)


    def __transform(self, var, mean, std):
        """
        Normalise/standardisation
        """
        return var.sub(mean).div(std)

    def __inverse_transform(self, var, mean, std):
        """
        Inverse the normalisation/standardisation
        """
        return var.mul(std).add(mean)

    def __get_train_vars__(self,indx):
        """
        Return normalised variables
        """
        x_var_data = {}
        y_var_data = {}
        y_var_data2 = {}

        for x in self.xvars:
            if self.dat_type == "train":
                i = x+"_train"
            elif self.dat_type == "test":
                i = x+"_test"
            x_var_data[i] = self.__transform(torch.from_numpy(self.xdata_and_norm[i][0][indx]), self.xdata_and_norm[i][1], self.xdata_and_norm[i][2])
        
        for y in self.yvars:
            if self.dat_type == "train":
                i = y+"_train"
            elif self.dat_type == "test":
                i = y+"_test"
            y_var_data[i] = self.__transform(torch.from_numpy(self.ydata_and_norm[i][0][indx]), self.ydata_and_norm[i][1], self.ydata_and_norm[i][2])

        for y2 in self.yvars2:
            if self.dat_type == "train":
                i = y2+"_train"
            elif self.dat_type == "test":
                i = y2+"_test"
            y_var_data2[i] = self.__transform(torch.from_numpy(self.ydata_and_norm[i][0][indx]), self.ydata_and_norm[i][1], self.ydata_and_norm[i][2])

        return x_var_data, y_var_data, y_var_data2
        
    def __getitem__(self, i):
        """
        batch iterable
        """
        x_var_data, y_var_data, y_var_data2 = self.__get_train_vars__(i)
        if self.add_adv == False:
            if self.dat_type == "train":
                x = torch.cat([x_var_data[k+'_train'] for k in self.xvars])
                y = torch.cat([y_var_data[k+'_train'] for k in self.yvars])
                y2 = torch.cat([y_var_data2[k+'_train'] for k in self.yvars2])
            elif self.dat_type == "test":
                x = torch.cat([x_var_data[k+'_test'] for k in self.xvars])
                y = torch.cat([y_var_data[k+'_test'] for k in self.yvars])
                y2 = torch.cat([y_var_data2[k+'_test'] for k in self.yvars2])
        elif self.add_adv == True:
            if self.dat_type == "train":
                x = torch.cat([x_var_data['qtot_train'] - x_var_data['qadv_train'], x_var_data['theta_train'] + x_var_data['theta_adv_train'], x_var_data['sw_toa_train'], x_var_data['shf_train'], x_var_data['lhf_train']])
                y = torch.cat([y_var_data['qphys_train'],y_var_data['theta_phys_train']])
            elif self.dat_type == "test":
                x = torch.cat([x_var_data['qtot_test'] - x_var_data['qadv_test'], x_var_data['theta_test'] + x_var_data['theta_adv_test'], x_var_data['sw_toa_test'], x_var_data['shf_test'], x_var_data['lhf_test']])
                y = torch.cat([y_var_data['qphys_test'], y_var_data['theta_phys_test']])

        return x,y,y2

    def __len__(self):
        return min(len(d) for d in self.xdat)

    def split_data(self, indata, xyz='x'):
        """
        Split x,y data into constituents
        """
        split_data = {}
        data_idx = {'x':self.xdata_idx,'y':self.ydata_idx,'y2':self.ydata_idx2}
        xyvars = {'x':self.xvars,'y':self.yvars,'y2':self.yvars2}
        for i,x in enumerate(xyvars[xyz]):
            l,h = data_idx[xyz][i]
            split_data[x] = indata[...,l:h]
        return split_data

class ConcatDatasetCNN(torch.utils.data.Dataset):
    def __init__(self, dat_type, nlevs, dataset_file, normaliser, data_frac=1., 
                xvars=['qtot', 'qadv', 'theta', 'theta_adv'],
                xvars2=['sw_toa', 'shf', 'lhf'],
                yvars=['qtot_next', 'theta_next'],
                yvars2=['qphys', 'theta_phys']):
        super().__init__()
        self.dat_type = dat_type
        self.dataset_file = dataset_file
        self.nlevs = nlevs
        self.nn_norm = NormalizersData(normaliser)
        self.xdat = []
        self.xdat2 = []
        self.xmean = []
        self.xmean2 = []
        self.xstd = []
        self.xstd2 = []
        self.ydat = []
        self.ymean = []
        self.ystd = []
        self.ydat2 = []
        self.ymean2 = []
        self.ystd2 = []
        self.xvars = xvars
        self.xvars2 = xvars2
        self.yvars = yvars
        self.yvars2 = yvars2

        if self.dat_type == "train":
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
            self.xdata_and_norm = {
                                    'qtot_train':[self.q_tot_train[:self.npoints, :self.nlevs], self.nn_norm.q_mean[0,:self.nlevs], self.nn_norm.q_stdscale[0,:self.nlevs]],
                                    'qadv_train':[self.q_tot_adv_train[:self.npoints, :self.nlevs], self.nn_norm.qadv_mean[0,:self.nlevs], self.nn_norm.qadv_stdscale[0,:self.nlevs]],
                                    'theta_train':[self.theta_train[:self.npoints, :self.nlevs], self.nn_norm.t_mean[0,:self.nlevs], self.nn_norm.t_stdscale[0,:self.nlevs]],
                                    'theta_adv_train':[self.theta_adv_train[:self.npoints, :self.nlevs], self.nn_norm.tadv_mean[0,:self.nlevs], self.nn_norm.tadv_stdscale[0,:self.nlevs]],
                                    'sw_toa_train':[self.sw_toa_train[:self.npoints], self.nn_norm.sw_toa_mean, self.nn_norm.sw_toa_stdscale],
                                    'shf_train':[self.shf_train[:self.npoints], self.nn_norm.upshf_mean, self.nn_norm.upshf_stdscale],
                                    'lhf_train':[self.lhf_train[:self.npoints], self.nn_norm.uplhf_mean, self.nn_norm.uplhf_stdscale]
                                    }
            self.ydata_and_norm = {
                                    'qphys_train':[self.qphys_train[:self.npoints, :self.nlevs], self.nn_norm.qphys_mean[0,:self.nlevs], self.nn_norm.qphys_stdscale[0,:self.nlevs]],
                                    # 'qphys_train':[self.qphys_train[:self.npoints, :3], self.nn_norm.qphys_mean[0,:3], self.nn_norm.qphys_stdscale[0,:3]],
                                    'theta_phys_train':[self.theta_phys_train[:self.npoints, :self.nlevs], self.nn_norm.tphys_mean[0,:self.nlevs], self.nn_norm.tphys_stdscale[0,:self.nlevs]],
                                    'qtot_next_train':[self.q_tot_train[:self.npoints, :self.nlevs]+self.q_tot_adv_train[:self.npoints, :self.nlevs]+self.qphys_train[:self.npoints, :self.nlevs], self.nn_norm.q_mean[0,:self.nlevs], self.nn_norm.q_stdscale[0,:self.nlevs]],
                                    # 'qtot_next_train':[self.q_tot_train[:self.npoints, :1]+self.q_tot_adv_train[:self.npoints, :1]+self.qphys_train[:self.npoints, :1], self.nn_norm.q_mean[0,:1], self.nn_norm.q_stdscale[0,:1]],
                                    'theta_next_train':[self.theta_train[:self.npoints, :self.nlevs]+self.theta_adv_train[:self.npoints, :self.nlevs]+self.theta_phys_train[:self.npoints, :self.nlevs], self.nn_norm.t_mean[0,:self.nlevs], self.nn_norm.t_stdscale[0,:self.nlevs]]

                                    }
            for x in self.xvars:
                i = x+"_train"
                self.xdat.append(self.xdata_and_norm[i][0])
                self.xmean.append(self.xdata_and_norm[i][1])
                self.xstd.append(self.xdata_and_norm[i][2])

            for x2 in self.xvars2:
                i = x2+"_train"
                self.xdat2.append(self.xdata_and_norm[i][0])
                self.xmean2.append(self.xdata_and_norm[i][1])
                self.xstd2.append(self.xdata_and_norm[i][2])

            for y in self.yvars:
                j = y+"_train"
                self.ydat.append(self.ydata_and_norm[j][0])
                self.ymean.append(self.ydata_and_norm[j][1])
                self.ystd.append(self.ydata_and_norm[j][2])

            for y2 in self.yvars2:
                k = y2+"_train"
                self.ydat2.append(self.ydata_and_norm[k][0])
                self.ymean2.append(self.ydata_and_norm[k][1])
                self.ystd2.append(self.ydata_and_norm[k][2])

        elif self.dat_type == "test":
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
            self.xdata_and_norm = {
                                    'qtot_test':[self.q_tot_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean[0,:self.nlevs], self.nn_norm.q_stdscale[0,:self.nlevs]],
                                    'qadv_test':[self.q_tot_adv_test[:self.npoints, :self.nlevs], self.nn_norm.qadv_mean[0,:self.nlevs], self.nn_norm.qadv_stdscale[0,:self.nlevs]],
                                    'theta_test':[self.theta_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean[0,:self.nlevs], self.nn_norm.t_stdscale[0,:self.nlevs]],
                                    'theta_adv_test':[self.theta_adv_test[:self.npoints, :self.nlevs], self.nn_norm.tadv_mean[0,:self.nlevs], self.nn_norm.tadv_stdscale[0,:self.nlevs]],
                                    'sw_toa_test':[self.sw_toa_test[:self.npoints], self.nn_norm.sw_toa_mean, self.nn_norm.sw_toa_stdscale],
                                    'shf_test':[self.shf_test[:self.npoints], self.nn_norm.upshf_mean, self.nn_norm.upshf_stdscale],
                                    'lhf_test':[self.lhf_test[:self.npoints], self.nn_norm.uplhf_mean, self.nn_norm.uplhf_stdscale]
                                    }
            self.ydata_and_norm = {
                                    'qphys_test':[self.qphys_test[:self.npoints, :self.nlevs], self.nn_norm.qphys_mean[0,:self.nlevs], self.nn_norm.qphys_stdscale[0,:self.nlevs]],
                                    # 'qphys_test':[self.qphys_test[:self.npoints, :3], self.nn_norm.qphys_mean[0,:3], self.nn_norm.qphys_stdscale[0,:3]],
                                    'theta_phys_test':[self.theta_phys_test[:self.npoints, :self.nlevs], self.nn_norm.tphys_mean[0,:self.nlevs], self.nn_norm.tphys_stdscale[0,:self.nlevs]],
                                    'qtot_next_test':[self.q_tot_test[:self.npoints, :self.nlevs]+self.q_tot_adv_test[:self.npoints, :self.nlevs]+self.qphys_test[:self.npoints, :self.nlevs], self.nn_norm.q_mean[0,:self.nlevs], self.nn_norm.q_stdscale[0,:self.nlevs]],
                                    # 'qtot_next_test':[self.q_tot_test[:self.npoints, :1]+self.q_tot_adv_test[:self.npoints, :1]+self.qphys_test[:self.npoints, :1], self.nn_norm.q_mean[0,:1], self.nn_norm.q_stdscale[0,:1]],
                                    'theta_next_test':[self.theta_test[:self.npoints, :self.nlevs]+self.theta_adv_test[:self.npoints, :self.nlevs]+self.theta_phys_test[:self.npoints, :self.nlevs], self.nn_norm.t_mean[0,:self.nlevs], self.nn_norm.t_stdscale[0,:self.nlevs]]
                                    }
         
            for x in self.xvars:
                i = x+"_test"
                self.xdat.append(self.xdata_and_norm[i][0])
                self.xmean.append(self.xdata_and_norm[i][1])
                self.xstd.append(self.xdata_and_norm[i][2])

            for x2 in self.xvars2:
                i = x2+"_test"
                self.xdat2.append(self.xdata_and_norm[i][0])
                self.xmean2.append(self.xdata_and_norm[i][1])
                self.xstd2.append(self.xdata_and_norm[i][2])

            for y in self.yvars:
                j = y+"_test"
                self.ydat.append(self.ydata_and_norm[j][0])
                self.ymean.append(self.ydata_and_norm[j][1])
                self.ystd.append(self.ydata_and_norm[j][2])
            
            for y2 in self.yvars2:
                k = y2+"_test"
                self.ydat2.append(self.ydata_and_norm[k][0])
                self.ymean2.append(self.ydata_and_norm[k][1])
                self.ystd2.append(self.ydata_and_norm[k][2])

        self.ymean = torch.cat(self.ymean)
        self.ystd = torch.cat(self.ystd)
        self.ymean2 = torch.cat(self.ymean2)
        self.ystd2 = torch.cat(self.ystd2)
        self.xmean = torch.stack(self.xmean)
        self.xstd = torch.stack(self.xstd)
        self.xmean2 = torch.cat(self.xmean2)
        self.xstd2 = torch.cat(self.xstd2)

    def __transform(self, var, mean, std):
        """
        Normalise/standardisation
        """
        return var.sub(mean).div(std)

    def __inverse_transform(self, var, mean, std):
        """
        Inverse the normalisation/standardisation
        """
        return var.mul(std).add(mean)

    def __get_train_vars__(self,indx):
        """
        Return normalised variables
        """
        x_var_data = {}
        x_var_data2 = {}
        y_var_data = {}
        y_var_data2 = {}

        for x in self.xvars:
            if self.dat_type == "train":
                i = x+"_train"
            elif self.dat_type == "test":
                i = x+"_test"
            x_var_data[i] = self.__transform(torch.from_numpy(self.xdata_and_norm[i][0][indx]), self.xdata_and_norm[i][1], self.xdata_and_norm[i][2])
        
        for x2 in self.xvars2:
            if self.dat_type == "train":
                i = x2+"_train"
            elif self.dat_type == "test":
                i = x2+"_test"
            x_var_data2[i] = self.__transform(torch.from_numpy(self.xdata_and_norm[i][0][indx]), self.xdata_and_norm[i][1], self.xdata_and_norm[i][2])

        for y in self.yvars:
            if self.dat_type == "train":
                i = y+"_train"
            elif self.dat_type == "test":
                i = y+"_test"
            y_var_data[i] = self.__transform(torch.from_numpy(self.ydata_and_norm[i][0][indx]), self.ydata_and_norm[i][1], self.ydata_and_norm[i][2])

        for y2 in self.yvars2:
            if self.dat_type == "train":
                i = y2+"_train"
            elif self.dat_type == "test":
                i = y2+"_test"
            y_var_data2[i] = self.__transform(torch.from_numpy(self.ydata_and_norm[i][0][indx]), self.ydata_and_norm[i][1], self.ydata_and_norm[i][2])

        return x_var_data, x_var_data2, y_var_data, y_var_data2
        
    def __getitem__(self, i):
        """
        batch iterable
        """
        x_var_data, x_var_data2, y_var_data, y_var_data2 = self.__get_train_vars__(i)
        if self.dat_type == "train":
            x = torch.stack([x_var_data[k+'_train'] for k in self.xvars])
            x2 = torch.cat([x_var_data2[k+'_train'] for k in self.xvars2])
            y = torch.cat([y_var_data[k+'_train'] for k in self.yvars])
            y2 = torch.cat([y_var_data2[k+'_train'] for k in self.yvars2])
        elif self.dat_type == "test":
            x = torch.stack([x_var_data[k+'_test'] for k in self.xvars])
            x2 = torch.cat([x_var_data2[k+'_test'] for k in self.xvars2])
            y = torch.cat([y_var_data[k+'_test'] for k in self.yvars])
            y2 = torch.cat([y_var_data2[k+'_test'] for k in self.yvars2])

        return x,x2,y,y2

    def __len__(self):
        return min(len(d) for d in self.xdat)

class ConcatDatasetCNN2D(torch.utils.data.Dataset):
    def __init__(self, dat_type, nlevs, dataset_file, normaliser, data_frac=1., 
                xvars=['qtot', 'qadv', 'theta', 'theta_adv'],
                xvars2=['sw_toa', 'shf', 'lhf'],
                yvars=['qtot_next', 'theta_next'],
                yvars2=['qphys', 'theta_phys']):
        super().__init__()
        self.dat_type = dat_type
        self.dataset_file = dataset_file
        self.nlevs = nlevs
        self.nn_norm = NormalizersData(normaliser)
        self.xdat = []
        self.xdat2 = []
        self.xmean = []
        self.xmean2 = []
        self.xstd = []
        self.xstd2 = []
        self.ydat = []
        self.ymean = []
        self.ystd = []
        self.ydat2 = []
        self.ymean2 = []
        self.ystd2 = []
        self.xvars = xvars
        self.xvars2 = xvars2
        self.yvars = yvars
        self.yvars2 = yvars2
        # self.norm = transforms.Normalize(mean=(0.5,0.5,0.5,0.5), std=(0.25,0.25,0.25,0.25))


        if self.dat_type == "train":
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
            self.xdata_and_norm = {
                                    'qtot_train':[self.q_tot_train[:self.npoints, :-1, :self.nlevs], self.nn_norm.q_mean[:,:self.nlevs], self.nn_norm.q_stdscale[:,:self.nlevs]],
                                    'qadv_train':[self.q_tot_adv_train[:self.npoints, :-1, :self.nlevs], self.nn_norm.qadv_mean[:,:self.nlevs], self.nn_norm.qadv_stdscale[:,:self.nlevs]],
                                    'theta_train':[self.theta_train[:self.npoints, :-1, :self.nlevs], self.nn_norm.t_mean[:,:self.nlevs], self.nn_norm.t_stdscale[:,:self.nlevs]],
                                    'theta_adv_train':[self.theta_adv_train[:self.npoints, :-1, :self.nlevs], self.nn_norm.tadv_mean[:,:self.nlevs], self.nn_norm.tadv_stdscale[:,:self.nlevs]],
                                    'sw_toa_train':[self.sw_toa_train[:self.npoints,:-1,0], self.nn_norm.sw_toa_mean, self.nn_norm.sw_toa_stdscale],
                                    'shf_train':[self.shf_train[:self.npoints,:-1,0], self.nn_norm.upshf_mean, self.nn_norm.upshf_stdscale],
                                    'lhf_train':[self.lhf_train[:self.npoints,:-1,0], self.nn_norm.uplhf_mean, self.nn_norm.uplhf_stdscale]
                                    }
            self.ydata_and_norm = {
                                    'qphys_train':[self.qphys_train[:self.npoints, -1, :self.nlevs], self.nn_norm.qphys_mean[:,:self.nlevs], self.nn_norm.qphys_stdscale[:,:self.nlevs]],
                                    'theta_phys_train':[self.theta_phys_train[:self.npoints, -1, :self.nlevs], self.nn_norm.tphys_mean[:,:self.nlevs], self.nn_norm.tphys_stdscale[:,:self.nlevs]],
                                    'qtot_next_train':[self.q_tot_train[:self.npoints, -1, :self.nlevs], self.nn_norm.q_mean[:,:self.nlevs], self.nn_norm.q_stdscale[:,:self.nlevs]],
                                    'theta_next_train':[self.theta_train[:self.npoints, -1, :self.nlevs], self.nn_norm.t_mean[:,:self.nlevs], self.nn_norm.t_stdscale[:,:self.nlevs]]

                                    }
            for x in self.xvars:
                i = x+"_train"
                self.xdat.append(self.xdata_and_norm[i][0])
                self.xmean.append(self.xdata_and_norm[i][1])
                self.xstd.append(self.xdata_and_norm[i][2])

            for x2 in self.xvars2:
                i = x2+"_train"
                self.xdat2.append(self.xdata_and_norm[i][0])
                self.xmean2.append(self.xdata_and_norm[i][1])
                self.xstd2.append(self.xdata_and_norm[i][2])

            for y in self.yvars:
                j = y+"_train"
                self.ydat.append(self.ydata_and_norm[j][0])
                self.ymean.append(self.ydata_and_norm[j][1])
                self.ystd.append(self.ydata_and_norm[j][2])

            for y2 in self.yvars2:
                k = y2+"_train"
                self.ydat2.append(self.ydata_and_norm[k][0])
                self.ymean2.append(self.ydata_and_norm[k][1])
                self.ystd2.append(self.ydata_and_norm[k][2])

        elif self.dat_type == "test":
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
            self.xdata_and_norm = {
                                    'qtot_test':[self.q_tot_test[:self.npoints, :-1, :self.nlevs], self.nn_norm.q_mean[:,:self.nlevs], self.nn_norm.q_stdscale[:,:self.nlevs]],
                                    'qadv_test':[self.q_tot_adv_test[:self.npoints, :-1, :self.nlevs], self.nn_norm.qadv_mean[:,:self.nlevs], self.nn_norm.qadv_stdscale[:,:self.nlevs]],
                                    'theta_test':[self.theta_test[:self.npoints, :-1, :self.nlevs], self.nn_norm.t_mean[:,:self.nlevs], self.nn_norm.t_stdscale[:,:self.nlevs]],
                                    'theta_adv_test':[self.theta_adv_test[:self.npoints, :-1, :self.nlevs], self.nn_norm.tadv_mean[:,:self.nlevs], self.nn_norm.tadv_stdscale[:,:self.nlevs]],
                                    'sw_toa_test':[self.sw_toa_test[:self.npoints, :-1,0], self.nn_norm.sw_toa_mean, self.nn_norm.sw_toa_stdscale],
                                    'shf_test':[self.shf_test[:self.npoints, :-1,0], self.nn_norm.upshf_mean, self.nn_norm.upshf_stdscale],
                                    'lhf_test':[self.lhf_test[:self.npoints, :-1,0], self.nn_norm.uplhf_mean, self.nn_norm.uplhf_stdscale]
                                    }
            self.ydata_and_norm = {
                                    'qphys_test':[self.qphys_test[:self.npoints, -1, :self.nlevs], self.nn_norm.qphys_mean[:,:self.nlevs], self.nn_norm.qphys_stdscale[:,:self.nlevs]],
                                    'theta_phys_test':[self.theta_phys_test[:self.npoints, -1, :self.nlevs], self.nn_norm.tphys_mean[:,:self.nlevs], self.nn_norm.tphys_stdscale[:,:self.nlevs]],
                                    'qtot_next_test':[self.q_tot_test[:self.npoints, -1, :self.nlevs], self.nn_norm.q_mean[:,:self.nlevs], self.nn_norm.q_stdscale[:,:self.nlevs]],
                                    'theta_next_test':[self.theta_test[:self.npoints, -1, :self.nlevs], self.nn_norm.t_mean[:,:self.nlevs], self.nn_norm.t_stdscale[:,:self.nlevs]]
                                    }
         
            for x in self.xvars:
                i = x+"_test"
                self.xdat.append(self.xdata_and_norm[i][0])
                self.xmean.append(self.xdata_and_norm[i][1])
                self.xstd.append(self.xdata_and_norm[i][2])

            for x2 in self.xvars2:
                i = x2+"_test"
                self.xdat2.append(self.xdata_and_norm[i][0])
                self.xmean2.append(self.xdata_and_norm[i][1])
                self.xstd2.append(self.xdata_and_norm[i][2])

            for y in self.yvars:
                j = y+"_test"
                self.ydat.append(self.ydata_and_norm[j][0])
                self.ymean.append(self.ydata_and_norm[j][1])
                self.ystd.append(self.ydata_and_norm[j][2])
            
            for y2 in self.yvars2:
                k = y2+"_test"
                self.ydat2.append(self.ydata_and_norm[k][0])
                self.ymean2.append(self.ydata_and_norm[k][1])
                self.ystd2.append(self.ydata_and_norm[k][2])

        self.ymean = torch.cat(self.ymean)
        self.ystd = torch.cat(self.ystd)
        self.ymean2 = torch.cat(self.ymean2)
        self.ystd2 = torch.cat(self.ystd2)
        self.xmean = torch.stack(self.xmean)
        self.xstd = torch.stack(self.xstd)
        self.xmean2 = torch.cat(self.xmean2)
        self.xstd2 = torch.cat(self.xstd2)

    def __transform(self, var, mean, std):
        """
        Normalise/standardisation
        """
        if len(var.shape) > 1:
            mean = mean.reshape(1, self.nlevs)
            std = std.reshape(1, self.nlevs)
        # print("var:", var.shape, var[:,-1])
        # print("mean, std:", mean.shape, std.shape, mean[:,-1], std[:,-1])
        
        return var.sub(mean).div(std)

    def __inverse_transform(self, var, mean, std):
        """
        Inverse the normalisation/standardisation
        """
        if len(var.shape) > 1:
            mean = mean.reshape(1, self.nlevs)
            std = std.reshape(1, self.nlevs)
        return var.mul(std).add(mean)

    def __get_train_vars__(self,indx):
        """
        Return normalised variables
        """
        x_var_data = {}
        x_var_data2 = {}
        y_var_data = {}
        y_var_data2 = {}

        for x in self.xvars:
            if self.dat_type == "train":
                i = x+"_train"
            elif self.dat_type == "test":
                i = x+"_test"
            x_var_data[i] = self.__transform(torch.from_numpy(self.xdata_and_norm[i][0][indx]), self.xdata_and_norm[i][1], self.xdata_and_norm[i][2])
        
        for x2 in self.xvars2:
            if self.dat_type == "train":
                i = x2+"_train"
            elif self.dat_type == "test":
                i = x2+"_test"
            x_var_data2[i] = self.__transform(torch.from_numpy(self.xdata_and_norm[i][0][indx]), self.xdata_and_norm[i][1], self.xdata_and_norm[i][2])

        for y in self.yvars:
            if self.dat_type == "train":
                i = y+"_train"
            elif self.dat_type == "test":
                i = y+"_test"
            y_var_data[i] = self.__transform(torch.from_numpy(self.ydata_and_norm[i][0][indx]), self.ydata_and_norm[i][1], self.ydata_and_norm[i][2])

        for y2 in self.yvars2:
            if self.dat_type == "train":
                i = y2+"_train"
            elif self.dat_type == "test":
                i = y2+"_test"
            y_var_data2[i] = self.__transform(torch.from_numpy(self.ydata_and_norm[i][0][indx]), self.ydata_and_norm[i][1], self.ydata_and_norm[i][2])

        return x_var_data, x_var_data2, y_var_data, y_var_data2
        
    def __getitem__(self, i):
        """
        batch iterable
        """
        x_var_data, x_var_data2, y_var_data, y_var_data2 = self.__get_train_vars__(i)
        if self.dat_type == "train":
            x = torch.stack([x_var_data[k+'_train'] for k in self.xvars], dim=0)
            # print(x.shape)
            # print("pre", x[2,0,:])
            # x = self.norm(x)
            # print("pos", x[2,0,:])
            x2 = torch.cat([x_var_data2[k+'_train'] for k in self.xvars2])
            y = torch.cat([y_var_data[k+'_train'] for k in self.yvars])
            y2 = torch.cat([y_var_data2[k+'_train'] for k in self.yvars2])
        elif self.dat_type == "test":
            x = torch.stack([x_var_data[k+'_test'] for k in self.xvars], dim=0)
            x2 = torch.cat([x_var_data2[k+'_test'] for k in self.xvars2])
            y = torch.cat([y_var_data[k+'_test'] for k in self.yvars])
            y2 = torch.cat([y_var_data2[k+'_test'] for k in self.yvars2])

        return x,x2,y.reshape(self.nlevs*len(self.yvars)),y2

    def __len__(self):
        return min(len(d) for d in self.xdat)
