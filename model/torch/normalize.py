import torch
import h5py

class Normalizers(object):
    def __init__(self, locations):
        # Data normalisers - minmax
        self.qphys_normaliser_mm = h5py.File('{0}/minmax_qphystot.hdf5'.format(locations['normaliser_loc']),'r')
        self.q_normaliser_mm = h5py.File('{0}/minmax_qtot.hdf5'.format(locations['normaliser_loc']),'r')
        self.qadd_normaliser_mm = h5py.File('{0}/minmax_qadd_dot.hdf5'.format(locations['normaliser_loc']),'r')
        self.qphys_feature_min = torch.tensor(self.qphys_normaliser_mm['feature_range'][0])
        self.qphys_feature_max = torch.tensor(self.qphys_normaliser_mm['feature_range'][1])
        self.qadd_feature_min  = torch.tensor(self.qadd_normaliser_mm['feature_range'][0])
        self.q_feature_max = torch.tensor(self.q_normaliser_mm['feature_range'][1])
        self.q_feature_min  = torch.tensor(self.q_normaliser_mm['feature_range'][0])
        self.qadd_feature_max = torch.tensor(self.qadd_normaliser_mm['feature_range'][1])
        self.qphys_mmscale = torch.from_numpy(self.qphys_normaliser_mm['scale_'][:])
        self.qadd_mmscale = torch.from_numpy(self.qadd_normaliser_mm['scale_'][:])
        self.q_mmscale = torch.from_numpy(self.q_normaliser_mm['scale_'][:])
        self.qphys_data_min = torch.from_numpy(self.qphys_normaliser_mm['data_min_'][:])
        self.qadd_data_min = torch.from_numpy(self.qadd_normaliser_mm['data_min_'][:])
        self.q_data_min = torch.from_numpy(self.q_normaliser_mm['data_min_'][:])
        # Data normaliser - std
        self.qphys_normaliser_std = h5py.File('{0}/std_qphystot.hdf5'.format(locations['normaliser_loc']),'r')
        self.qphys_dot_normaliser_std = h5py.File('{0}/std_qphysdot.hdf5'.format(locations['normaliser_loc']),'r')
        self.qphys_dot_normaliser_std_s = h5py.File('{0}/std_qphysdot_s.hdf5'.format(locations['normaliser_loc']),'r')
        self.q_normaliser_std = h5py.File('{0}/std_qtot.hdf5'.format(locations['normaliser_loc']),'r')
        self.q_normaliser_std_s = h5py.File('{0}/std_qtot_s.hdf5'.format(locations['normaliser_loc']),'r')
        self.qadd_normaliser_std = h5py.File('{0}/std_qadd_dot.hdf5'.format(locations['normaliser_loc']),'r')
        self.qadd_normaliser_std_s = h5py.File('{0}/std_qadd_dot_s.hdf5'.format(locations['normaliser_loc']),'r')
        self.qadv_dot_normaliser_std = h5py.File('{0}/std_qadv_dot.hdf5'.format(locations['normaliser_loc']),'r')
        self.qadv_dot_normaliser_std_s = h5py.File('{0}/std_qadv_dot_s.hdf5'.format(locations['normaliser_loc']),'r')
        self.qadv_normaliser_std = h5py.File('{0}/std_qadvtot.hdf5'.format(locations['normaliser_loc']),'r')

        self.qphys_mean = torch.tensor(self.qphys_normaliser_std['mean_'][:])
        self.qphys_dot_mean = torch.tensor(self.qphys_dot_normaliser_std['mean_'][:])
        self.qphys_dot_mean_s = torch.tensor(self.qphys_dot_normaliser_std_s['mean_'][:])
        self.qadd_mean  = torch.tensor(self.qadd_normaliser_std['mean_'][:])
        self.qadd_mean_s  = torch.tensor(self.qadd_normaliser_std_s['mean_'][:])
        self.q_mean = torch.tensor(self.q_normaliser_std['mean_'][:])
        self.q_mean_s = torch.tensor(self.q_normaliser_std_s['mean_'][:])
        self.qadv_dot_mean = torch.tensor(self.qadv_dot_normaliser_std['mean_'][:])
        self.qadv_dot_mean_s = torch.tensor(self.qadv_dot_normaliser_std_s['mean_'][:])
        self.qadv_mean = torch.tensor(self.qadv_normaliser_std['mean_'][:])
        

        self.qphys_stdscale = torch.from_numpy(self.qphys_normaliser_std['scale_'][:])
        self.qphys_dot_stdscale = torch.from_numpy(self.qphys_dot_normaliser_std['scale_'][:])
        self.qphys_dot_stdscale_s = torch.from_numpy(self.qphys_dot_normaliser_std_s['scale_'][:])
        self.qadd_stdscale = torch.from_numpy(self.qadd_normaliser_std['scale_'][:])
        self.qadd_stdscale_s = torch.from_numpy(self.qadd_normaliser_std_s['scale_'][:])
        self.q_stdscale = torch.from_numpy(self.q_normaliser_std['scale_'][:])
        self.q_stdscale_s = torch.from_numpy(self.q_normaliser_std_s['scale_'][:])
        self.qadv_dot_stdscale = torch.from_numpy(self.qadv_dot_normaliser_std['scale_'][:])
        self.qadv_dot_stdscale_s = torch.from_numpy(self.qadv_dot_normaliser_std_s['scale_'][:])
        self.qadv_stdscale = torch.from_numpy(self.qadv_normaliser_std['scale_'][:])


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