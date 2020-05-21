import h5py
import matplotlib.pyplot as plt


def standardised_data(datafile, npoints):
    dataset = h5py.File(datafile, 'r')
    qphys_test_norm = dataset['q_phys_train'][:npoints]
    theta_phys_test_norm = dataset['t_phys_train'][:npoints]
    qtot_test_norm = dataset['q_tot_train'][:npoints]
    qadv_test_norm = dataset['q_adv_train'][:npoints]
    theta_test_norm = dataset['air_potential_temperature_train'][:npoints]
    theta_adv_test_norm = dataset['t_adv_train'][:npoints]
    sw_toa = dataset['toa_incoming_shortwave_flux_train'][:npoints]
    shf = dataset['surface_upward_sensible_heat_flux_train'][:npoints]
    lhf = dataset['surface_upward_latent_heat_flux_train'][:npoints]

    level = 0
    fig, axs = plt.subplots(3,3,figsize=(14, 10),sharex=True)
    ax = axs[0,0]
    ax.plot(qphys_test_norm[:,level],'.-',label='qphys')
    ax.legend()

    ax = axs[1,0]
    ax.plot(theta_phys_test_norm[:,level],'.-',label='tphys')
    ax.legend()
    
    ax = axs[2,0]
    ax.plot(qadv_test_norm[:,level],'.-',label='qadv')
    ax.legend()

    ax = axs[0,1]
    ax.plot(sw_toa[:],'.-',label='sw_toa')
    ax.legend()

    ax = axs[1,1]
    ax.plot(theta_adv_test_norm[:,level],'.-',label='tadv')
    ax.legend()
    
    ax = axs[2,1]
    ax.plot(theta_test_norm[:,level],'.-',label='theta')
    ax.legend()
    
    ax = axs[0,2]
    ax.plot(qtot_test_norm[:,level],'.-',label='qtot')
    ax.legend()

    ax = axs[1,2]
    ax.plot(shf[:],'.-',label='shf')
    ax.legend()

    ax = axs[2,2]
    ax.plot(lhf[:],'.-',label='lhf')
    ax.legend()

    ax.set_title('Level {0}'.format(level))
    ax.legend()
    plt.show()

if __name__ == "__main__":
    datafile="/project/spice/radiation/ML/CRM/data/models/archive_datain/train_data_021501AQ_std.hdf5"
    standardised_data(datafile, 10000)