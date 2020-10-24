import numpy as np 
import h5py

class NormalizersData(object):
    def __init__(self, location, nlevs):
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

        self.qphys_mean = self.qphys_normaliser_std['mean_'][0,:nlevs]
        self.tphys_mean = self.tphys_normaliser_std['mean_'][0,:nlevs]
        self.q_mean = self.q_normaliser_std['mean_'][0,:nlevs]
        self.t_mean = self.t_normaliser_std['mean_'][0,:nlevs]
        self.qadv_mean = self.qadv_normaliser_std['mean_'][0,:nlevs]
        self.tadv_mean = self.tadv_normaliser_std['mean_'][0,:nlevs]
        self.sw_toa_mean = self.sw_toa_normaliser_std['mean_'][:]
        self.upshf_mean = self.uplhf_normaliser_std['mean_'][:]
        self.uplhf_mean = self.uplhf_normaliser_std['mean_'][:]

        self.qphys_stdscale = self.qphys_normaliser_std['scale_'][0,:nlevs]
        self.tphys_stdscale = self.tphys_normaliser_std['scale_'][0,:nlevs]
        self.q_stdscale = self.q_normaliser_std['scale_'][0,:nlevs]
        self.t_stdscale = self.t_normaliser_std['scale_'][0,:nlevs]
        self.qadv_stdscale = self.qadv_normaliser_std['scale_'][0,:nlevs]
        self.tadv_stdscale = self.tadv_normaliser_std['scale_'][0,:nlevs]
        self.sw_toa_stdscale = self.sw_toa_normaliser_std['scale_'][:]
        self.upshf_stdscale = self.uplhf_normaliser_std['scale_'][:]
        self.uplhf_stdscale = self.uplhf_normaliser_std['scale_'][:]

    def normalise(self, data, mean, scale):
        return (data - mean) / scale
    
    def inverse_transform(self, data, mean, scale):
        return (data * scale) + mean


def get_data(dataset_file, data_frac, nlevs, normaliser=None):
    print("Reading dataset file: {0}".format(dataset_file))
    dataset=h5py.File(dataset_file, 'r')
    q_tot_tr = dataset["q_tot_train"]
    q_tot_tst = dataset["q_tot_test"]

    npoints_train = int(q_tot_tr.shape[0] * data_frac)
    npoints_test = int(q_tot_tst.shape[0] * data_frac)

    q_tot_train = dataset["q_tot_train"][:npoints_train,:nlevs]
    q_tot_adv_train = dataset["q_adv_train"][:npoints_train, :nlevs]
    theta_train = dataset["air_potential_temperature_train"][:npoints_train, :nlevs]
    theta_adv_train = dataset["t_adv_train"][:npoints_train, :nlevs]
    sw_toa_train = dataset["toa_incoming_shortwave_flux_train"][:npoints_train]
    shf_train = dataset["surface_upward_sensible_heat_flux_train"][:npoints_train]
    lhf_train = dataset["surface_upward_latent_heat_flux_train"][:npoints_train]
    theta_phys_train = dataset["t_phys_train"][:npoints_train, :nlevs]
    qphys_train = dataset["q_phys_train"][:npoints_train, :nlevs]
    qnext_train = q_tot_train + q_tot_adv_train + qphys_train
    theta_next_train = theta_train + theta_adv_train + theta_phys_train

    q_tot_test = dataset["q_tot_test"][:npoints_test, :nlevs]
    q_tot_adv_test = dataset["q_adv_test"][:npoints_test, :nlevs]
    theta_test = dataset["air_potential_temperature_test"][:npoints_test, :nlevs]
    theta_adv_test = dataset["t_adv_test"][:npoints_test, :nlevs]
    sw_toa_test = dataset["toa_incoming_shortwave_flux_test"][:npoints_test]
    shf_test = dataset["surface_upward_sensible_heat_flux_test"][:npoints_test]
    lhf_test = dataset["surface_upward_latent_heat_flux_test"][:npoints_test]
    theta_phys_test = dataset["t_phys_test"][:npoints_test, :nlevs]
    qphys_test = dataset["q_phys_test"][:npoints_test, :nlevs]
    qnext_test = q_tot_test + q_tot_adv_test + qphys_test
    theta_next_test = theta_test + theta_adv_test + theta_phys_test

    if normaliser is not None:
        norm = NormalizersData(normaliser, nlevs)
        q_tot_train = norm.normalise(q_tot_train, norm.q_mean, norm.q_stdscale)
        q_tot_adv_train = norm.normalise(q_tot_adv_train, norm.qadv_mean, norm.qadv_stdscale)
        theta_train = norm.normalise(theta_train, norm.t_mean, norm.t_stdscale)
        theta_adv_train = norm.normalise(theta_adv_train, norm.tadv_mean, norm.tadv_stdscale)
        sw_toa_train = norm.normalise(sw_toa_train, norm.sw_toa_mean, norm.sw_toa_stdscale)
        shf_train = norm.normalise(shf_train, norm.upshf_mean, norm.upshf_stdscale)
        lhf_train = norm.normalise(lhf_train, norm.uplhf_mean, norm.uplhf_stdscale)
        qnext_train = norm.normalise(qnext_train, norm.q_mean, norm.q_stdscale)
        theta_next_train = norm.normalise(theta_next_train, norm.t_mean, norm.t_stdscale)
        
        q_tot_test = norm.normalise(q_tot_test, norm.q_mean, norm.q_stdscale)
        q_tot_adv_test = norm.normalise(q_tot_adv_test, norm.qadv_mean, norm.qadv_stdscale)
        theta_test = norm.normalise(theta_test, norm.t_mean, norm.t_stdscale)
        theta_adv_test = norm.normalise(theta_adv_test, norm.tadv_mean, norm.tadv_stdscale)
        sw_toa_test = norm.normalise(sw_toa_test, norm.sw_toa_mean, norm.sw_toa_stdscale)
        shf_test = norm.normalise(shf_test, norm.upshf_mean, norm.upshf_stdscale)
        lhf_test = norm.normalise(lhf_test, norm.uplhf_mean, norm.uplhf_stdscale)
        qnext_test = norm.normalise(qnext_test, norm.q_mean, norm.q_stdscale)
        theta_next_test = norm.normalise(theta_next_test, norm.t_mean, norm.t_stdscale)



    x_train = np.concatenate([q_tot_train, q_tot_adv_train, theta_train, theta_adv_train, sw_toa_train, shf_train, lhf_train], axis=1)
    # y_train = np.concatenate([qphys_train, theta_phys_train], axis=1)
    # y_train = np.concatenate([qnext_train, theta_next_train], axis=1)
    y_train = qnext_train
    # y_train = theta_next_train

    x_test = np.concatenate([q_tot_test, q_tot_adv_test, theta_test, theta_adv_test, sw_toa_test, shf_test, lhf_test], axis=1)
    # y_test = np.concatenate([qphys_test, theta_phys_test], axis=1)
    # y_test = np.concatenate([qnext_test, theta_next_test], axis=1)
    y_test = qnext_test
    # y_test = theta_next_test

    return x_train, y_train, x_test, y_test

def get_val_data(dataset_file, nlevs, normaliser=None):
    print("Reading dataset file: {0}".format(dataset_file))
    dataset=h5py.File(dataset_file, 'r')

    q_tot_test = dataset["q_tot_test"][:, :nlevs]
    q_tot_adv_test = dataset["q_adv_test"][:, :nlevs]
    theta_test = dataset["air_potential_temperature_test"][:, :nlevs]
    theta_adv_test = dataset["t_adv_test"][:, :nlevs]
    sw_toa_test = dataset["toa_incoming_shortwave_flux_test"][:]
    shf_test = dataset["surface_upward_sensible_heat_flux_test"][:]
    lhf_test = dataset["surface_upward_latent_heat_flux_test"][:]
    theta_phys_test = dataset["t_phys_test"][:, :nlevs]
    qphys_test = dataset["q_phys_test"][:, :nlevs]
    qnext_test = q_tot_test + q_tot_adv_test + qphys_test
    theta_next_test = theta_test + theta_adv_test + theta_phys_test

    if normaliser is not None:
        norm = NormalizersData(normaliser, nlevs)
        q_tot_test = norm.normalise(q_tot_test, norm.q_mean, norm.q_stdscale)
        q_tot_adv_test = norm.normalise(q_tot_adv_test, norm.qadv_mean, norm.qadv_stdscale)
        theta_test = norm.normalise(theta_test, norm.t_mean, norm.t_stdscale)
        theta_adv_test = norm.normalise(theta_adv_test, norm.tadv_mean, norm.tadv_stdscale)
        sw_toa_test = norm.normalise(sw_toa_test, norm.sw_toa_mean, norm.sw_toa_stdscale)
        shf_test = norm.normalise(shf_test, norm.upshf_mean, norm.upshf_stdscale)
        lhf_test = norm.normalise(lhf_test, norm.uplhf_mean, norm.uplhf_stdscale)
        qnext_test = norm.normalise(qnext_test, norm.q_mean, norm.q_stdscale)
        theta_next_test = norm.normalise(theta_next_test, norm.t_mean, norm.t_stdscale)
    # x_test = np.concatenate([q_tot_test, q_tot_adv_test, theta_test, theta_adv_test, sw_toa_test, shf_test, lhf_test], axis=1)
    # y_test = np.concatenate([qphys_test, theta_phys_test], axis=1)
    # y_test = np.concatenate([qnext_test, theta_next_test], axis=1)

    return [q_tot_test, q_tot_adv_test, theta_test, theta_adv_test, sw_toa_test, shf_test, lhf_test], [qnext_test, theta_next_test]