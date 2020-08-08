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
        self.qphys_normaliser_std = h5py.File("{0}/q_phys.hdf5".format(location),"r")
        self.tphys_normaliser_std = h5py.File("{0}/t_phys.hdf5".format(location),"r")
        self.q_normaliser_std = h5py.File("{0}/q_tot.hdf5".format(location),"r")
        self.t_normaliser_std = h5py.File("{0}/air_potential_temperature.hdf5".format(location),"r")
        self.qadv_normaliser_std = h5py.File("{0}/q_adv.hdf5".format(location),"r")
        self.tadv_normaliser_std = h5py.File("{0}/t_adv.hdf5".format(location),"r")
        self.sw_toa_normaliser_std = h5py.File("{0}/toa_incoming_shortwave_flux.hdf5".format(location),"r")
        self.upshf_normaliser_std = h5py.File("{0}/surface_upward_sensible_heat_flux.hdf5".format(location),"r")
        self.uplhf_normaliser_std = h5py.File("{0}/surface_upward_latent_heat_flux.hdf5".format(location),"r")
        self.pressure_std = h5py.File("{0}/air_pressure.hdf5".format(location),"r")
        self.rho_std = h5py.File("{0}/m01s00i253.hdf5".format(location),"r")
        self.xwind_std = h5py.File("{0}/x_wind.hdf5".format(location),"r")
        self.ywind_std = h5py.File("{0}/y_wind.hdf5".format(location),"r")
        self.zwind_std = h5py.File("{0}/upward_air_velocity.hdf5".format(location),"r")

        self.qphys_mean = torch.tensor(self.qphys_normaliser_std["mean_"][:])
        self.tphys_mean = torch.tensor(self.tphys_normaliser_std["mean_"][:])
        self.q_mean = torch.tensor(self.q_normaliser_std["mean_"][:])
        self.t_mean = torch.tensor(self.t_normaliser_std["mean_"][:])
        self.qadv_mean = torch.tensor(self.qadv_normaliser_std["mean_"][:])
        self.tadv_mean = torch.tensor(self.tadv_normaliser_std["mean_"][:])
        self.sw_toa_mean = torch.tensor(self.sw_toa_normaliser_std["mean_"][:])
        self.upshf_mean = torch.tensor(self.uplhf_normaliser_std["mean_"][:])
        self.uplhf_mean = torch.tensor(self.uplhf_normaliser_std["mean_"][:])
        self.pressure_mean = torch.tensor(self.pressure_std["mean_"][:])
        self.rho_mean = torch.tensor(self.rho_std["mean_"][:])
        self.xwind_mean = torch.tensor(self.xwind_std["mean_"][:])
        self.ywind_mean = torch.tensor(self.ywind_std["mean_"][:])
        self.zwind_mean = torch.tensor(self.zwind_std["mean_"][:])

        self.qphys_stdscale = torch.from_numpy(self.qphys_normaliser_std["scale_"][:])
        self.tphys_stdscale = torch.from_numpy(self.tphys_normaliser_std["scale_"][:])
        self.q_stdscale = torch.from_numpy(self.q_normaliser_std["scale_"][:])
        self.t_stdscale = torch.from_numpy(self.t_normaliser_std["scale_"][:])
        self.qadv_stdscale = torch.from_numpy(self.qadv_normaliser_std["scale_"][:])
        self.tadv_stdscale = torch.from_numpy(self.tadv_normaliser_std["scale_"][:])
        self.sw_toa_stdscale = torch.tensor(self.sw_toa_normaliser_std["scale_"][:])
        self.upshf_stdscale = torch.tensor(self.uplhf_normaliser_std["scale_"][:])
        self.uplhf_stdscale = torch.tensor(self.uplhf_normaliser_std["scale_"][:])
        self.pressure_stdscale = torch.tensor(self.pressure_std["scale_"][:])
        self.rho_stdscale = torch.tensor(self.rho_std["scale_"][:])
        self.xwind_stdscale = torch.tensor(self.xwind_std["scale_"][:])
        self.ywind_stdscale = torch.tensor(self.ywind_std["scale_"][:])
        self.zwind_stdscale = torch.tensor(self.zwind_std["scale_"][:])

        self.qphys_mean_np = self.qphys_normaliser_std["mean_"][:]
        self.tphys_mean_np = self.tphys_normaliser_std["mean_"][:]
        self.q_mean_np = self.q_normaliser_std["mean_"][:]
        self.t_mean_np = self.t_normaliser_std["mean_"][:]
        self.qadv_mean_np = self.qadv_normaliser_std["mean_"][:]
        self.tadv_mean_np = self.tadv_normaliser_std["mean_"][:]
        self.sw_toa_mean_np = self.sw_toa_normaliser_std["mean_"][:]
        self.upshf_mean_np = self.uplhf_normaliser_std["mean_"][:]
        self.uplhf_mean_np = self.uplhf_normaliser_std["mean_"][:]
        self.pressure_mean_np = self.pressure_std["mean_"][:]
        self.rho_mean_np = self.rho_std["mean_"][:]
        self.xwind_mean_np = self.xwind_std["mean_"][:]
        self.ywind_mean_np = self.ywind_std["mean_"][:]
        self.zwind_mean_np = self.zwind_std["mean_"][:]

        self.qphys_stdscale_np = self.qphys_normaliser_std["scale_"][:]
        self.tphys_stdscale_np = self.tphys_normaliser_std["scale_"][:]
        self.q_stdscale_np = self.q_normaliser_std["scale_"][:]
        self.t_stdscale_np = self.t_normaliser_std["scale_"][:]
        self.qadv_stdscale_np = self.qadv_normaliser_std["scale_"][:]
        self.tadv_stdscale_np = self.tadv_normaliser_std["scale_"][:]
        self.sw_toa_stdscale_np = self.sw_toa_normaliser_std["scale_"][:]
        self.upshf_stdscale_np = self.uplhf_normaliser_std["scale_"][:]
        self.uplhf_stdscale_np = self.uplhf_normaliser_std["scale_"][:]
        self.pressure_stdscale_np = self.pressure_std["scale_"][:]
        self.rho_stdscale_np = self.rho_std["scale_"][:]
        self.xwind_stdscale_np = self.xwind_std["scale_"][:]
        self.ywind_stdscale_np = self.ywind_std["scale_"][:]
        self.zwind_stdscale_np = self.zwind_std["scale_"][:]

    def normalise(self, data, mean, scale):
        return (data - mean) / scale
    
    def inverse_transform(self, data, mean, scale):
        return (data * scale) + mean

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, dat_type, nlevs, dataset_file, normaliser, batch_size, samples_frac=1.,data_frac=1., 
                xvars=["qtot", "qadv", "theta", "theta_adv", "sw_toa", "shf", "lhf", "p", "rho", "xwind", "ywind", "zwind"],
                yvars=["qtot_next", "theta_next"],
                yvars2=["qphys", "theta_phys"], no_norm=False):
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

        self.xdata_idx = []
        self.ydata_idx = []
        self.ydata_idx2 = []
        self.no_norm = no_norm
        self.batch_size = batch_size

        print("Reading dataset file: {0}".format(dataset_file))
        dataset=h5py.File(dataset_file,"r")
        self.q_tot_train = dataset["q_tot_"+self.dat_type]
        self.q_tot_diff_train = dataset["q_tot_diff_"+self.dat_type]
        self.q_tot_adv_train = dataset["q_adv_"+self.dat_type]
        self.theta_train = dataset["air_potential_temperature_"+self.dat_type]
        self.theta_diff_train = dataset["air_potential_temperature_diff_"+self.dat_type]
        self.theta_adv_train = dataset["t_adv_"+self.dat_type]
        self.sw_toa_train = dataset["toa_incoming_shortwave_flux_"+self.dat_type]
        self.shf_train = dataset["surface_upward_sensible_heat_flux_"+self.dat_type]
        self.lhf_train = dataset["surface_upward_latent_heat_flux_"+self.dat_type]
        self.theta_phys_train = dataset["t_phys_"+self.dat_type]
        self.qphys_train = dataset["q_phys_"+self.dat_type]
        self.p_train = dataset["air_pressure_"+self.dat_type]
        self.rho_train = dataset["m01s00i253_"+self.dat_type]
        self.xwind_train = dataset["x_wind_"+self.dat_type]
        self.ywind_train = dataset["y_wind_"+self.dat_type]
        self.zwind_train = dataset["upward_air_velocity_"+self.dat_type]
        self.npoints = int(self.q_tot_train.shape[1] * data_frac)
        if samples_frac > 1.:
            self.nsamples = int(samples_frac)
        else:
            self.nsamples = int(self.q_tot_train.shape[0] * samples_frac)
        print("N_samples: {0}".format(self.nsamples))
        print("N_points: {0}".format(self.npoints))

        self.end_indx = self.npoints  - self.npoints%self.batch_size
        self.nbatches = self.end_indx//self.batch_size
       
        # Build an index map to have nsample and npoints accessible via a single index call
        # print("Creating index map ... ")
        # self.index_map = {}
        # indx = 0
        # for i in range(self.nsamples * self.nbatches):
            # for j in range(self.batch_size):
                # self.index_map[indx] = (i,j)
                # print(i,j,indx, end='\n')
                # indx += 1
        self.total_points = self.nsamples * self.nbatches
        print("Batches {0}; Data points {1}".format(self.total_points, self.total_points*self.batch_size))
        self.v_slc1 = slice(0,self.nsamples)
        self.v_slc2 = slice(0,self.end_indx)
        self.v_slc3 = slice(0,self.nlevs)
        self.norm_slc = slice(self.nlevs)
        var3shape = (self.nsamples *self.nbatches, self.batch_size, self.nlevs)
        var2shape = (self.nsamples *self.nbatches, self.batch_size, 1)
        self.xdata_and_norm = {
                                "qtot_"+self.dat_type:[self.q_tot_train[self.v_slc1, self.v_slc2, self.v_slc3].reshape(var3shape), self.nn_norm.q_mean, self.nn_norm.q_stdscale],
                                "qadv_"+self.dat_type:[self.q_tot_adv_train[self.v_slc1, self.v_slc2, self.v_slc3].reshape(var3shape), self.nn_norm.qadv_mean, self.nn_norm.qadv_stdscale],
                                "theta_"+self.dat_type:[self.theta_train[self.v_slc1, self.v_slc2, self.v_slc3].reshape(var3shape), self.nn_norm.t_mean, self.nn_norm.t_stdscale],
                                "theta_adv_"+self.dat_type:[self.theta_adv_train[self.v_slc1, self.v_slc2, self.v_slc3].reshape(var3shape), self.nn_norm.tadv_mean, self.nn_norm.tadv_stdscale],
                                "sw_toa_"+self.dat_type:[self.sw_toa_train[self.v_slc1, self.v_slc2, :].reshape(var2shape), self.nn_norm.sw_toa_mean, self.nn_norm.sw_toa_stdscale],
                                "p_"+self.dat_type:[self.p_train[self.v_slc1, self.v_slc2, self.v_slc3].reshape(var3shape), self.nn_norm.pressure_mean, self.nn_norm.pressure_stdscale],
                                "rho_"+self.dat_type:[self.rho_train[self.v_slc1, self.v_slc2, self.v_slc3].reshape(var3shape), self.nn_norm.rho_mean, self.nn_norm.rho_stdscale],
                                "xwind_"+self.dat_type:[self.xwind_train[self.v_slc1, self.v_slc2, self.v_slc3].reshape(var3shape), self.nn_norm.xwind_mean, self.nn_norm.xwind_stdscale],
                                "ywind_"+self.dat_type:[self.ywind_train[self.v_slc1, self.v_slc2, self.v_slc3].reshape(var3shape), self.nn_norm.ywind_mean, self.nn_norm.ywind_stdscale],
                                "zwind_"+self.dat_type:[self.zwind_train[self.v_slc1, self.v_slc2, self.v_slc3].reshape(var3shape), self.nn_norm.zwind_mean, self.nn_norm.zwind_stdscale],
                                "shf_"+self.dat_type:[self.shf_train[self.v_slc1, self.v_slc2, :].reshape(var2shape), self.nn_norm.upshf_mean, self.nn_norm.upshf_stdscale],
                                "lhf_"+self.dat_type:[self.lhf_train[self.v_slc1, self.v_slc2, :].reshape(var2shape), self.nn_norm.uplhf_mean, self.nn_norm.uplhf_stdscale]
                                }
        self.ydata_and_norm = {
                                "qphys_"+self.dat_type:[self.qphys_train[self.v_slc1, self.v_slc2, self.v_slc3].reshape(var3shape), self.nn_norm.qphys_mean, self.nn_norm.qphys_stdscale],
                                # "qphys_"+self.dat_type:[self.qphys_train[:self.npoints, :3], self.nn_norm.qphys_mean[0,:3], self.nn_norm.qphys_stdscale[0,:3]],
                                "theta_phys_"+self.dat_type:[self.theta_phys_train[self.v_slc1, self.v_slc2, self.v_slc3].reshape(var3shape), self.nn_norm.tphys_mean, self.nn_norm.tphys_stdscale],
                                # "qtot_next_"+self.dat_type:[self.q_tot_train[:self.npoints, :self.nlevs]+self.q_tot_adv_train[:self.npoints, :self.nlevs]+self.qphys_train[:self.npoints, :self.nlevs], self.nn_norm.q_mean, self.nn_norm.q_stdscale],
                                # "qtot_next_"+self.dat_type:[self.q_tot_train[:self.npoints, :self.nlevs]+self.q_tot_adv_train[:self.npoints, :self.nlevs]+self.qphys_train[:self.npoints, :self.nlevs], self.nn_norm.q_mean, self.nn_norm.q_stdscale],
                                "qtot_"+self.dat_type:[self.q_tot_train[self.v_slc1, self.v_slc2, self.v_slc3].reshape(var3shape), self.nn_norm.q_mean, self.nn_norm.q_stdscale],
                                "theta_"+self.dat_type:[self.theta_train[self.v_slc1, self.v_slc2, self.v_slc3].reshape(var3shape), self.nn_norm.t_mean, self.nn_norm.t_stdscale],
                                "qtot_next_"+self.dat_type:[(self.q_tot_train[self.v_slc1, self.v_slc2, self.v_slc3]+self.q_tot_diff_train[self.v_slc1, self.v_slc2, self.v_slc3]).reshape(var3shape), self.nn_norm.q_mean, self.nn_norm.q_stdscale],
                                # "qtot_diff_"+self.dat_type:[self.q_tot_diff_train[self.v_slc1, self.v_slc2, self.v_slc3].reshape(var3shape), self.nn_norm.q_mean, self.nn_norm.q_stdscale],

                                # "qtot_next_"+self.dat_type:[self.q_tot_train[:self.npoints, :1]+self.q_tot_adv_train[:self.npoints, :1]+self.qphys_train[:self.npoints, :1], self.nn_norm.q_mean[0,:1], self.nn_norm.q_stdscale[0,:1]],
                                # "theta_next_"+self.dat_type:[self.theta_train[:self.npoints, :self.nlevs]+self.theta_adv_train[:self.npoints, :self.nlevs]+self.theta_phys_train[:self.npoints, :self.nlevs], self.nn_norm.t_mean, self.nn_norm.t_stdscale]
                                "theta_next_"+self.dat_type:[(self.theta_train[self.v_slc1, self.v_slc2, self.v_slc3]+self.theta_diff_train[self.v_slc1, self.v_slc2, self.v_slc3]).reshape(var3shape), self.nn_norm.t_mean, self.nn_norm.t_stdscale]
                                # "theta_diff_"+self.dat_type:[self.theta_diff_train[self.v_slc1, self.v_slc2, self.v_slc3].reshape(var3shape), self.nn_norm.t_mean, self.nn_norm.t_stdscale]
                                }
        start_idx = 0
        for x in self.xvars:
            i = x+"_"+self.dat_type
            self.xdat.append(self.xdata_and_norm[i][0])
            mean = self.xdata_and_norm[i][1].reshape(-1)[self.norm_slc]
            std = self.xdata_and_norm[i][2].reshape(-1)[self.norm_slc]
            self.xmean.append(mean)
            self.xstd.append(std)
            end_idx = start_idx + self.xdata_and_norm[i][0].shape[2] #len(self.xdata_and_norm[i][1])
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
            end_idx = start_idx + len(self.ydata_and_norm[j][1])
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
            end_idx = start_idx + len(self.ydata_and_norm[k][1])
            self.ydata_idx2.append((start_idx,end_idx))
            start_idx = end_idx
      
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
        # isamples, ipoints = self.index_map[indx]
        x_var_data = {}
        y_var_data = {}
        y_var_data2 = {}

        for x in self.xvars:
            if self.dat_type == "train":
                i = x+"_train"
            elif self.dat_type == "test":
                i = x+"_test"
            if self.no_norm:
                x_var_data[i] = torch.from_numpy(self.xdata_and_norm[i][0][indx, :,:])
            else:
                mean = self.xdata_and_norm[i][1].reshape(-1)[self.norm_slc]
                std = self.xdata_and_norm[i][2].reshape(-1)[self.norm_slc]
                x_var_data[i] = self.__transform(torch.from_numpy(self.xdata_and_norm[i][0][indx, :,:]), mean, std)
        
        for y in self.yvars:
            if self.dat_type == "train":
                i = y+"_train"
            elif self.dat_type == "test":
                i = y+"_test"
            if self.no_norm:
                y_var_data[i] = torch.from_numpy(self.ydata_and_norm[i][0][indx, :,:])
            else:
                mean = self.ydata_and_norm[i][1].reshape(-1)[self.norm_slc]
                std = self.ydata_and_norm[i][2].reshape(-1)[self.norm_slc]
                y_var_data[i] = self.__transform(torch.from_numpy(self.ydata_and_norm[i][0][indx, :,:]), mean, std)

        for y2 in self.yvars2:
            if self.dat_type == "train":
                i = y2+"_train"
            elif self.dat_type == "test":
                i = y2+"_test"
            if self.no_norm:
                y_var_data2[i] = torch.from_numpy(self.ydata_and_norm[i][0][indx,:,:])
            else:
                mean = self.ydata_and_norm[i][1].reshape(-1)[self.norm_slc]
                std = self.ydata_and_norm[i][2].reshape(-1)[self.norm_slc]
                y_var_data2[i] = self.__transform(torch.from_numpy(self.ydata_and_norm[i][0][indx,:,:]), mean, std)

        return x_var_data, y_var_data, y_var_data2
        
    def __getitem__(self, i):
        """
        batch iterable
        """
        # print("index {0}".format(i))
        x_var_data, y_var_data, y_var_data2 = self.__get_train_vars__(i)
        x = []
        y = []
        y2 = []
        if self.dat_type == "train":
            # x = torch.cat([x_var_data[k+"_train"] for k in self.xvars], dim=1)
            # y = torch.cat([y_var_data[k+"_train"] for k in self.yvars], dim=1)
            # y2 = torch.cat([y_var_data2[k+"_train"] for k in self.yvars2], dim=1)
            for xv in self.xvars:
                x.append(x_var_data[xv+"_train"])
            for yv in self.yvars:
                y.append(y_var_data[yv+"_train"])
            for y2v in self.yvars2:
                y2.append(y_var_data2[y2v+"_train"])
        elif self.dat_type == "test":
            # x = torch.cat([x_var_data[k+"_test"] for k in self.xvars], dim=1)
            # y = torch.cat([y_var_data[k+"_test"] for k in self.yvars], dim=1)
            # y2 = torch.cat([y_var_data2[k+"_test"] for k in self.yvars2], dim=1)
            for xv in self.xvars:
                x.append(x_var_data[xv+"_test"])
            for yv in self.yvars:
                y.append(y_var_data[yv+"_test"])
            for y2v in self.yvars2:
                y2.append(y_var_data2[y2v+"_test"])
    

        return x,y,y2

    def __len__(self):
        return self.total_points
        # return min(len(d) for d in self.xdat)

    def split_data(self, indata, xyz="x"):
        """
        Split x,y data into constituents
        """
        split_data = {}
        data_idx = {"x":self.xdata_idx,"y":self.ydata_idx,"y2":self.ydata_idx2}
        xyvars = {"x":self.xvars,"y":self.yvars,"y2":self.yvars2}
        for i,x in enumerate(xyvars[xyz]):
            l,h = data_idx[xyz][i]
            split_data[x] = indata[...,l:h]
        return split_data
