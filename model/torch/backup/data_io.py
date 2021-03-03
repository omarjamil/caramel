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
        self.qphys_normaliser_std = h5py.File('{0}/q_phys.hdf5'.format(location),'r')
        self.tphys_normaliser_std = h5py.File('{0}/t_phys.hdf5'.format(location),'r')
        self.q_normaliser_std = h5py.File('{0}/q_tot.hdf5'.format(location),'r')
        self.t_normaliser_std = h5py.File('{0}/air_potential_temperature.hdf5'.format(location),'r')
        self.qadv_normaliser_std = h5py.File('{0}/q_adv.hdf5'.format(location),'r')
        self.tadv_normaliser_std = h5py.File('{0}/t_adv.hdf5'.format(location),'r')
        self.sw_toa_normaliser_std = h5py.File('{0}/toa_incoming_shortwave_flux.hdf5'.format(location),'r')
        self.upshf_normaliser_std = h5py.File('{0}/surface_upward_sensible_heat_flux.hdf5'.format(location),'r')
        self.uplhf_normaliser_std = h5py.File('{0}/surface_upward_latent_heat_flux.hdf5'.format(location),'r')

        self.qphys_mean = torch.tensor(self.qphys_normaliser_std['mean_'][:])
        self.tphys_mean = torch.tensor(self.tphys_normaliser_std['mean_'][:])
        self.q_mean = torch.tensor(self.q_normaliser_std['mean_'][:])
        self.t_mean = torch.tensor(self.t_normaliser_std['mean_'][:])
        self.qadv_mean = torch.tensor(self.qadv_normaliser_std['mean_'][:])
        self.tadv_mean = torch.tensor(self.tadv_normaliser_std['mean_'][:])
        self.sw_toa_mean = torch.tensor(self.sw_toa_normaliser_std['mean_'][:])
        self.upshf_mean = torch.tensor(self.uplhf_normaliser_std['mean_'][:])
        self.uplhf_mean = torch.tensor(self.uplhf_normaliser_std['mean_'][:])

        self.qphys_stdscale = torch.from_numpy(self.qphys_normaliser_std['scale_'][:])
        self.tphys_stdscale = torch.from_numpy(self.tphys_normaliser_std['scale_'][:])
        self.q_stdscale = torch.from_numpy(self.q_normaliser_std['scale_'][:])
        self.t_stdscale = torch.from_numpy(self.t_normaliser_std['scale_'][:])
        self.qadv_stdscale = torch.from_numpy(self.qadv_normaliser_std['scale_'][:])
        self.tadv_stdscale = torch.from_numpy(self.tadv_normaliser_std['scale_'][:])
        self.sw_toa_stdscale = torch.tensor(self.sw_toa_normaliser_std['scale_'][:])
        self.upshf_stdscale = torch.tensor(self.uplhf_normaliser_std['scale_'][:])
        self.uplhf_stdscale = torch.tensor(self.uplhf_normaliser_std['scale_'][:])

        self.qphys_mean_np = self.qphys_normaliser_std['mean_'][:]
        self.tphys_mean_np = self.tphys_normaliser_std['mean_'][:]
        self.q_mean_np = self.q_normaliser_std['mean_'][:]
        self.t_mean_np = self.t_normaliser_std['mean_'][:]
        self.qadv_mean_np = self.qadv_normaliser_std['mean_'][:]
        self.tadv_mean_np = self.tadv_normaliser_std['mean_'][:]
        self.sw_toa_mean_np = self.sw_toa_normaliser_std['mean_'][:]
        self.upshf_mean_np = self.uplhf_normaliser_std['mean_'][:]
        self.uplhf_mean_np = self.uplhf_normaliser_std['mean_'][:]

        self.qphys_stdscale_np = self.qphys_normaliser_std['scale_'][:]
        self.tphys_stdscale_np = self.tphys_normaliser_std['scale_'][:]
        self.q_stdscale_np = self.q_normaliser_std['scale_'][:]
        self.t_stdscale_np = self.t_normaliser_std['scale_'][:]
        self.qadv_stdscale_np = self.qadv_normaliser_std['scale_'][:]
        self.tadv_stdscale_np = self.tadv_normaliser_std['scale_'][:]
        self.sw_toa_stdscale_np = self.sw_toa_normaliser_std['scale_'][:]
        self.upshf_stdscale_np = self.uplhf_normaliser_std['scale_'][:]
        self.uplhf_stdscale_np = self.uplhf_normaliser_std['scale_'][:]

    def normalise(self, data, mean, scale):
        return (data - mean) / scale
    
    def inverse_transform(self, data, mean, scale):
        return (data * scale) + mean

class Data_IO_validation(object):
    def __init__(self, region, nlevs, dataset_file, normaliser, 
                xvars=['qtot', 'qadv', 'theta', 'theta_adv', 'sw_toa', 'shf', 'lhf'],
                yvars=['qtot_next', 'theta_next'],
                yvars2=['qphys', 'theta_phys'],add_adv=False):
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
        start_idx = 0
        for x in self.xvars:
            i = x+"_test"
            self.xdat.append(self.xdata_and_norm[i][0])
            self.xmean.append(self.xdata_and_norm[i][1])
            self.xstd.append(self.xdata_and_norm[i][2])
            end_idx = start_idx + len(self.xdata_and_norm[i][1])
            self.xdata_idx.append((start_idx,end_idx))
            start_idx = end_idx

        start_idx = 0
        for y in self.yvars:
            j = y+"_test"
            self.ydat.append(self.ydata_and_norm[j][0])
            self.ymean.append(self.ydata_and_norm[j][1])
            self.ystd.append(self.ydata_and_norm[j][2])
            end_idx = start_idx + len(self.ydata_and_norm[j][1])
            self.ydata_idx.append((start_idx,end_idx))
            start_idx = end_idx
        
        start_idx = 0
        for y2 in self.yvars2:
            k = y2+"_test"
            self.ydat2.append(self.ydata_and_norm[k][0])
            self.ymean2.append(self.ydata_and_norm[k][1])
            self.ystd2.append(self.ydata_and_norm[k][2])
            end_idx = start_idx + len(self.ydata_and_norm[k][1])
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
            x_var_data[i] = self._transform(torch.from_numpy(self.xdata_and_norm[i][0][:]), self.xdata_and_norm[i][1], self.xdata_and_norm[i][2])
        
        for y in self.yvars:
            i = y+"_test"
            y_var_data[i] = self._transform(torch.from_numpy(self.ydata_and_norm[i][0][:]), self.ydata_and_norm[i][1], self.ydata_and_norm[i][2])
        
        for y2 in self.yvars2:
            i = y2+"_test"
            y_var_data2[i] = self._transform(torch.from_numpy(self.ydata_and_norm[i][0][:]), self.ydata_and_norm[i][1], self.ydata_and_norm[i][2])

        return x_var_data, y_var_data, y_var_data2

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

        # x = torch.cat([torch.from_numpy(d) for d in self.xdat], dim=1)
        # y = torch.cat([torch.from_numpy(d) for d in self.ydat], dim=1)
        # x_ = self._transform(x, self.xmean, self.xstd)
        # y_ = self._transform(y, self.ymean, self.ystd)
        # return (x_,y_, self.xmean, self.xstd, self.ymean, self.ystd)

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, dat_type, nlevs, dataset_file, normaliser, data_frac=1., 
                xvars=['qtot', 'qadv', 'theta', 'theta_adv', 'sw_toa', 'shf', 'lhf'],
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