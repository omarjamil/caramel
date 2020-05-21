import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import data_io


def plot_variables(dataset_file,nn_norm, data_frac, nlevs):
    print("Reading dataset file: {0}".format(dataset_file))
    dataset=h5py.File(dataset_file,'r')
    q_tot_train = dataset["q_tot_train"]
    q_tot_adv_train = dataset["q_adv_train"]
    theta_train = dataset["air_potential_temperature_train"]
    theta_adv_train = dataset["t_adv_train"]
    sw_toa_train = dataset["toa_incoming_shortwave_flux_train"]
    shf_train = dataset["surface_upward_sensible_heat_flux_train"]
    lhf_train = dataset["surface_upward_latent_heat_flux_train"]
    theta_phys_train = dataset["t_phys_train"]
    qphys_train = dataset["q_phys_train"]
    npoints = int(q_tot_train.shape[0] * data_frac)
    xdata_and_norm = {
                            'qtot_train':[q_tot_train[:npoints, :nlevs], nn_norm.q_mean_np[0,:nlevs], nn_norm.q_stdscale_np[0,:nlevs]],
                            'qadv_train':[q_tot_adv_train[:npoints, :nlevs], nn_norm.qadv_mean_np[0,:nlevs], nn_norm.qadv_stdscale_np[0,:nlevs]],
                            'theta_train':[theta_train[:npoints, :nlevs], nn_norm.t_mean_np[0,:nlevs], nn_norm.t_stdscale_np[0,:nlevs]],
                            'theta_adv_train':[theta_adv_train[:npoints, :nlevs], nn_norm.tadv_mean_np[0,:nlevs], nn_norm.tadv_stdscale_np[0,:nlevs]],
                            'sw_toa_train':[sw_toa_train[:npoints], nn_norm.sw_toa_mean_np, nn_norm.sw_toa_stdscale_np],
                            'shf_train':[shf_train[:npoints], nn_norm.upshf_mean_np, nn_norm.upshf_stdscale_np],
                            'lhf_train':[lhf_train[:npoints], nn_norm.uplhf_mean_np, nn_norm.uplhf_stdscale_np]
                            }
    ydata_and_norm = {
                            'qphys_train':[qphys_train[:npoints, :nlevs], nn_norm.qphys_mean_np[0,:nlevs], nn_norm.qphys_stdscale_np[0,:nlevs]],
                            # 'qphys_train':[qphys_train[:npoints, :3], nn_norm.qphys_mean_np[0,:3], nn_norm.qphys_stdscale_np[0,:3]],
                            'theta_phys_train':[theta_phys_train[:npoints, :nlevs], nn_norm.tphys_mean_np[0,:nlevs], nn_norm.tphys_stdscale_np[0,:nlevs]],
                            'qtot_next_train':[q_tot_train[:npoints, :nlevs]+q_tot_adv_train[:npoints, :nlevs]+qphys_train[:npoints, :nlevs], nn_norm.q_mean_np[0,:nlevs], nn_norm.q_stdscale_np[0,:nlevs]],
                            # 'qtot_next_train':[q_tot_train[:npoints, :3]+q_tot_adv_train[:npoints, :3]+qphys_train[:npoints, :3], nn_norm.q_mean_np[0,:3], nn_norm.q_stdscale_np[0,:3]],
                            'theta_next_train':[theta_train[:npoints, :nlevs]+theta_adv_train[:npoints, :nlevs]+theta_phys_train[:npoints, :nlevs], nn_norm.t_mean_np[0,:nlevs], nn_norm.t_stdscale_np[0,:nlevs]]

                            }

    qtot = xdata_and_norm['qtot_train'][0]
    qtot_mean = xdata_and_norm['qtot_train'][1]
    qtot_std = xdata_and_norm['qtot_train'][2]
    qtot_norm = nn_norm.normalise(qtot, qtot_mean, qtot_std)
    qadv = xdata_and_norm['qadv_train'][0]
    qadv_mean = xdata_and_norm['qadv_train'][1]
    qadv_std = xdata_and_norm['qadv_train'][2]
    qadv_norm = nn_norm.normalise(qadv, qadv_mean, qadv_std)
    theta = xdata_and_norm['theta_train'][0]
    theta_mean = xdata_and_norm['theta_train'][1]
    theta_std = xdata_and_norm['theta_train'][2]
    theta_norm = nn_norm.normalise(theta, theta_mean, theta_std)
    theta_adv = xdata_and_norm['theta_adv_train'][0]
    theta_adv_mean = xdata_and_norm['theta_adv_train'][1]
    theta_adv_std = xdata_and_norm['theta_adv_train'][2]
    theta_adv_norm = nn_norm.normalise(theta_adv, theta_adv_mean, theta_adv_std)

    fig, axs = plt.subplots(2,4,sharex=True)
    ax = axs[0,0]
    ax.plot(np.arange(nlevs), qtot[0,:],'k-',label='qtot')
    ax.plot(np.arange(nlevs), qtot_mean,'k.',label='qtot_mean')
    ax.fill_between(np.arange(nlevs),qtot[0,:]-qtot_std,qtot[0,:]+qtot_std)
    ax.legend()

    ax = axs[1,0]
    ax.plot(np.arange(nlevs), qtot_norm[0,:],'k-',label='qtot_norm')
    ax.legend()

    ax = axs[0,1]
    ax.plot(np.arange(nlevs), theta[0,:],'k-',label='theta')
    ax.plot(np.arange(nlevs), theta_mean,'k.',label='theta_mean')
    ax.fill_between(np.arange(nlevs),theta[0,:]-theta_std,theta[0,:]+theta_std)
    ax.legend()

    ax = axs[1,1]
    ax.plot(np.arange(nlevs), theta_norm[0,:],'k-',label='theta_norm')
    ax.legend()

    ax = axs[0,2]
    ax.plot(np.arange(nlevs), qadv[0,:],'k-',label='qadv')
    ax.plot(np.arange(nlevs), qadv_mean,'k.',label='qadv_mean')
    # ax.plot(np.arange(nlevs), qadv[0,:]/qtot_mean,'r.-',label='qadv/qtot')
    ax.fill_between(np.arange(nlevs),qadv[0,:]-qadv_std,qadv[0,:]+qadv_std)
    ax.legend()

    ax = axs[1,2]
    ax.plot(np.arange(nlevs), qadv_norm[0,:],'k-',label='qadv_norm')
    ax.legend()

    ax = axs[0,3]
    ax.plot(np.arange(nlevs), theta_adv[0,:],'k-',label='theta_adv')
    ax.plot(np.arange(nlevs), theta_adv_mean,'k.',label='theta_adv_mean')
    ax.fill_between(np.arange(nlevs),theta_adv[0,:]-theta_adv_std,theta_adv[0,:]+theta_adv_std)
    ax.legend()

    ax = axs[1,3]
    ax.plot(np.arange(nlevs), theta_adv_norm[0,:],'k-',label='theta_adv_norm')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    training_data_file = "/project/spice/radiation/ML/CRM/data/models/datain/train_data_023001AQ3HT.hdf5"
    # normaliser = "/project/spice/radiation/ML/CRM/data/models/normaliser/021501AQ3H_normalise/"
    normaliser = "/project/spice/radiation/ML/CRM/data/models/normaliser/163001AQ3HT_normalise/"
    data_norm = data_io.NormalizersData(normaliser)
    data_frac = 0.01
    nlevs = 45
    plot_variables(training_data_file,data_norm, data_frac, nlevs)