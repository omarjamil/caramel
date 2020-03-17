import torch
import h5py

class Normalizers(object):
    def __init__(self, locations):
        print("Initialising normaliser to location: {0}".format(locations['normaliser_loc']))
        self.qphys_normaliser_std = h5py.File('{0}/q_phys.hdf5'.format(locations['normaliser_loc']),'r')
        self.tphys_normaliser_std = h5py.File('{0}/t_phys.hdf5'.format(locations['normaliser_loc']),'r')
        self.q_normaliser_std = h5py.File('{0}/q_tot.hdf5'.format(locations['normaliser_loc']),'r')
        self.t_normaliser_std = h5py.File('{0}/air_potential_temperature.hdf5'.format(locations['normaliser_loc']),'r')
        self.qadv_normaliser_std = h5py.File('{0}/q_adv.hdf5'.format(locations['normaliser_loc']),'r')
        self.tadv_normaliser_std = h5py.File('{0}/t_adv.hdf5'.format(locations['normaliser_loc']),'r')

        self.qphys_mean = torch.tensor(self.qphys_normaliser_std['mean_'][:])
        self.tphys_mean = torch.tensor(self.tphys_normaliser_std['mean_'][:])
        self.q_mean = torch.tensor(self.q_normaliser_std['mean_'][:])
        self.t_mean = torch.tensor(self.t_normaliser_std['mean_'][:])
        self.qadv_mean = torch.tensor(self.qadv_normaliser_std['mean_'][:])
        self.tadv_mean = torch.tensor(self.tadv_normaliser_std['mean_'][:])
        

        self.qphys_stdscale = torch.from_numpy(self.qphys_normaliser_std['scale_'][:])
        self.tphys_stdscale = torch.from_numpy(self.tphys_normaliser_std['scale_'][:])
        self.q_stdscale = torch.from_numpy(self.q_normaliser_std['scale_'][:])
        self.t_stdscale = torch.from_numpy(self.t_normaliser_std['scale_'][:])
        self.qadv_stdscale = torch.from_numpy(self.qadv_normaliser_std['scale_'][:])
        self.tadv_stdscale = torch.from_numpy(self.tadv_normaliser_std['scale_'][:])

        # # Data normaliser - std
        # # self.qphys_normaliser_std = h5py.File('{0}/std_qphystot.hdf5'.format(locations['normaliser_loc']),'r')
        # self.qphys_dot_normaliser_std = h5py.File('{0}/std_qphysdot.hdf5'.format(locations['normaliser_loc']),'r')
        # self.qphys_dot_normaliser_std_s = h5py.File('{0}/std_qphysdot_s.hdf5'.format(locations['normaliser_loc']),'r')
        # self.q_normaliser_std = h5py.File('{0}/std_qtot.hdf5'.format(locations['normaliser_loc']),'r')
        # self.q_normaliser_std_s = h5py.File('{0}/std_qtot_s.hdf5'.format(locations['normaliser_loc']),'r')
        # self.qadd_normaliser_std = h5py.File('{0}/std_qadd_dot.hdf5'.format(locations['normaliser_loc']),'r')
        # self.qadd_normaliser_std_s = h5py.File('{0}/std_qadd_dot_s.hdf5'.format(locations['normaliser_loc']),'r')
        # self.qadv_dot_normaliser_std = h5py.File('{0}/std_qadv_dot.hdf5'.format(locations['normaliser_loc']),'r')
        # self.qadv_dot_normaliser_std_s = h5py.File('{0}/std_qadv_dot_s.hdf5'.format(locations['normaliser_loc']),'r')
        # # self.qadv_normaliser_std = h5py.File('{0}/std_qadvtot.hdf5'.format(locations['normaliser_loc']),'r')
        # self.tadv_dot_normaliser_std = h5py.File('{0}/std_tadv_dot.hdf5'.format(locations['normaliser_loc']),'r')

        # # self.qphys_mean = torch.tensor(self.qphys_normaliser_std['mean_'][:])
        # self.qphys_dot_mean = torch.tensor(self.qphys_dot_normaliser_std['mean_'][:])
        # self.qphys_dot_mean_s = torch.tensor(self.qphys_dot_normaliser_std_s['mean_'][:])
        # self.qadd_mean  = torch.tensor(self.qadd_normaliser_std['mean_'][:])
        # self.qadd_mean_s  = torch.tensor(self.qadd_normaliser_std_s['mean_'][:])
        # self.q_mean = torch.tensor(self.q_normaliser_std['mean_'][:])
        # self.q_mean_s = torch.tensor(self.q_normaliser_std_s['mean_'][:])
        # self.qadv_dot_mean = torch.tensor(self.qadv_dot_normaliser_std['mean_'][:])
        # self.qadv_dot_mean_s = torch.tensor(self.qadv_dot_normaliser_std_s['mean_'][:])
        # # self.qadv_mean = torch.tensor(self.qadv_normaliser_std['mean_'][:])
        # self.tadv_mean = torch.tensor(self.tadv_dot_normaliser_std['mean_'][:])
        

        # # self.qphys_stdscale = torch.from_numpy(self.qphys_normaliser_std['scale_'][:])
        # self.qphys_dot_stdscale = torch.from_numpy(self.qphys_dot_normaliser_std['scale_'][:])
        # self.qphys_dot_stdscale_s = torch.from_numpy(self.qphys_dot_normaliser_std_s['scale_'][:])
        # self.qadd_stdscale = torch.from_numpy(self.qadd_normaliser_std['scale_'][:])
        # self.qadd_stdscale_s = torch.from_numpy(self.qadd_normaliser_std_s['scale_'][:])
        # self.q_stdscale = torch.from_numpy(self.q_normaliser_std['scale_'][:])
        # self.q_stdscale_s = torch.from_numpy(self.q_normaliser_std_s['scale_'][:])
        # self.qadv_dot_stdscale = torch.from_numpy(self.qadv_dot_normaliser_std['scale_'][:])
        # self.qadv_dot_stdscale_s = torch.from_numpy(self.qadv_dot_normaliser_std_s['scale_'][:])
        # # self.qadv_stdscale = torch.from_numpy(self.qadv_normaliser_std['scale_'][:])
        # self.tadv_stdscale = torch.from_numpy(self.tadv_dot_normaliser_std['scale_'][:])


    def inverse_std(self, input_vals, scale, mean):
        """
        Inverse transform standardised data
        """
        output_tensor = (input_vals*scale) + mean
        return output_tensor

    def std(self, input_vals, scale, mean):
        """
        Standard scaling data
        """
        output_tensor = (input_vals - mean)/scale
        return output_tensor

    def inverse_minmax(self, input_vals, scale, feature_min, feature_max, data_min):
        """
        Inverse min max scaled tensor
        """
        range_min, range_max = feature_min, feature_max 
        output_tensor = (input_vals - range_min)/scale + data_min 
        return output_tensor

    def minmax(self, input_vals, scale, feature_min, feature_max, data_min):
        """
        Minmax scaling based on scikit-learn minmaxscaler
        """
        # scale = (range_max - range_min) / (torch.max(input_tensor,0)[0] - torch.min(input_tensor,0)[0])
        range_min, range_max = feature_min, feature_max
        output_tensor = scale * input_vals + range_min - data_min * scale #torch.min(input_tensor, 0)[0] * scale
        return output_tensor