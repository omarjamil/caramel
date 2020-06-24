"""
Save PCA components for use later
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import data_io
from scipy import stats
from sklearn.decomposition import PCA
from numpy.testing import assert_array_almost_equal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def validation_data(dataset_file, nn_norm, nlevs):
    print("Reading dataset file: {0}".format(dataset_file))
    dataset=h5py.File(dataset_file,'r')
    q_tot_test = dataset["q_tot_test"]
    q_tot_adv_test = dataset["q_adv_test"]
    theta_test = dataset["air_potential_temperature_test"]
    theta_adv_test = dataset["t_adv_test"]
    sw_toa_test = dataset["toa_incoming_shortwave_flux_test"]
    shf_test = dataset["surface_upward_sensible_heat_flux_test"]
    lhf_test = dataset["surface_upward_latent_heat_flux_test"]
    theta_phys_test = dataset["t_phys_test"]
    qphys_test = dataset["q_phys_test"]
    npoints = int(q_tot_test.shape[0])
    xdata_and_norm = {
                            'qtot_test':[q_tot_test[:npoints, :nlevs], nn_norm.q_mean_np[0,:nlevs], nn_norm.q_stdscale_np[0,:nlevs]],
                            'qadv_test':[q_tot_adv_test[:npoints, :nlevs], nn_norm.qadv_mean_np[0,:nlevs], nn_norm.qadv_stdscale_np[0,:nlevs]],
                            'theta_test':[theta_test[:npoints, :nlevs], nn_norm.t_mean_np[0,:nlevs], nn_norm.t_stdscale_np[0,:nlevs]],
                            'theta_adv_test':[theta_adv_test[:npoints, :nlevs], nn_norm.tadv_mean_np[0,:nlevs], nn_norm.tadv_stdscale_np[0,:nlevs]],
                            'sw_toa_test':[sw_toa_test[:npoints], nn_norm.sw_toa_mean_np, nn_norm.sw_toa_stdscale_np],
                            'shf_test':[shf_test[:npoints], nn_norm.upshf_mean_np, nn_norm.upshf_stdscale_np],
                            'lhf_test':[lhf_test[:npoints], nn_norm.uplhf_mean_np, nn_norm.uplhf_stdscale_np]
                            }
    ydata_and_norm = {
                            'qphys_test':[qphys_test[:npoints, :nlevs], nn_norm.qphys_mean_np[0,:nlevs], nn_norm.qphys_stdscale_np[0,:nlevs]],
                            # 'qphys_test':[qphys_test[:npoints, :3], nn_norm.qphys_mean_np[0,:3], nn_norm.qphys_stdscale_np[0,:3]],
                            'theta_phys_test':[theta_phys_test[:npoints, :nlevs], nn_norm.tphys_mean_np[0,:nlevs], nn_norm.tphys_stdscale_np[0,:nlevs]],
                            'qtot_next_test':[q_tot_test[:npoints, :nlevs]+q_tot_adv_test[:npoints, :nlevs]+qphys_test[:npoints, :nlevs], nn_norm.q_mean_np[0,:nlevs], nn_norm.q_stdscale_np[0,:nlevs]],
                            # 'qtot_next_test':[q_tot_test[:npoints, :3]+q_tot_adv_test[:npoints, :3]+qphys_test[:npoints, :3], nn_norm.q_mean_np[0,:3], nn_norm.q_stdscale_np[0,:3]],
                            'theta_next_test':[theta_test[:npoints, :nlevs]+theta_adv_test[:npoints, :nlevs]+theta_phys_test[:npoints, :nlevs], nn_norm.t_mean_np[0,:nlevs], nn_norm.t_stdscale_np[0,:nlevs]]
                            } 

    qtot = xdata_and_norm['qtot_test'][0]
    qtot_mean = xdata_and_norm['qtot_test'][1]
    qtot_std = xdata_and_norm['qtot_test'][2]
    qtot_norm = nn_norm.normalise(qtot, qtot_mean, qtot_std)
    qadv = xdata_and_norm['qadv_test'][0]
    qadv_mean = xdata_and_norm['qadv_test'][1]
    qadv_std = xdata_and_norm['qadv_test'][2]
    qadv_norm = nn_norm.normalise(qadv, qadv_mean, qadv_std)
    theta = xdata_and_norm['theta_test'][0]
    theta_mean = xdata_and_norm['theta_test'][1]
    theta_std = xdata_and_norm['theta_test'][2]
    theta_norm = nn_norm.normalise(theta, theta_mean, theta_std)
    theta_adv = xdata_and_norm['theta_adv_test'][0]
    theta_adv_mean = xdata_and_norm['theta_adv_test'][1]
    theta_adv_std = xdata_and_norm['theta_adv_test'][2]
    theta_adv_norm = nn_norm.normalise(theta_adv, theta_adv_mean, theta_adv_std)
    sw_toa = xdata_and_norm['sw_toa_test'][0]
    sw_toa_mean = xdata_and_norm['sw_toa_test'][1]
    sw_toa_std = xdata_and_norm['sw_toa_test'][2]
    sw_toa_norm = nn_norm.normalise(sw_toa, sw_toa_mean, sw_toa_std)
    shf = xdata_and_norm['shf_test'][0]
    shf_mean = xdata_and_norm['shf_test'][1]
    shf_std = xdata_and_norm['shf_test'][2]
    shf_norm = nn_norm.normalise(shf, shf_mean, shf_std)
    lhf = xdata_and_norm['lhf_test'][0]
    lhf_mean = xdata_and_norm['lhf_test'][1]
    lhf_std = xdata_and_norm['lhf_test'][2]
    lhf_norm = nn_norm.normalise(lhf, lhf_mean, lhf_std)

    qphys = ydata_and_norm['qphys_test'][0]
    qphys_mean = ydata_and_norm['qphys_test'][1]
    qphys_std = ydata_and_norm['qphys_test'][2]
    qphys_norm = nn_norm.normalise(qphys, qphys_mean, qphys_std)
    theta_phys = ydata_and_norm['theta_phys_test'][0]
    theta_phys_mean = ydata_and_norm['theta_phys_test'][1]
    theta_phys_std = ydata_and_norm['theta_phys_test'][2]
    theta_phys_norm = nn_norm.normalise(theta_phys, theta_phys_mean, theta_phys_std)

    xvars = [qtot_norm, qadv_norm, theta_norm, theta_adv_norm, sw_toa_norm, shf_norm, lhf_norm]
    yvars = [qphys_norm]

    X_validate = np.hstack(xvars)
    y_validate = np.hstack(yvars)

    return X_validate, y_validate

def load_dataset(dataset_file,nn_norm, data_frac, nlevs):
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
    sw_toa = xdata_and_norm['sw_toa_train'][0]
    sw_toa_mean = xdata_and_norm['sw_toa_train'][1]
    sw_toa_std = xdata_and_norm['sw_toa_train'][2]
    sw_toa_norm = nn_norm.normalise(sw_toa, sw_toa_mean, sw_toa_std)
    shf = xdata_and_norm['shf_train'][0]
    shf_mean = xdata_and_norm['shf_train'][1]
    shf_std = xdata_and_norm['shf_train'][2]
    shf_norm = nn_norm.normalise(shf, shf_mean, shf_std)
    lhf = xdata_and_norm['lhf_train'][0]
    lhf_mean = xdata_and_norm['lhf_train'][1]
    lhf_std = xdata_and_norm['lhf_train'][2]
    lhf_norm = nn_norm.normalise(lhf, lhf_mean, lhf_std)

    qphys = ydata_and_norm['qphys_train'][0]
    qphys_mean = ydata_and_norm['qphys_train'][1]
    qphys_std = ydata_and_norm['qphys_train'][2]
    qphys_norm = nn_norm.normalise(qphys, qphys_mean, qphys_std)
    theta_phys = ydata_and_norm['theta_phys_train'][0]
    theta_phys_mean = ydata_and_norm['theta_phys_train'][1]
    theta_phys_std = ydata_and_norm['theta_phys_train'][2]
    theta_phys_norm = nn_norm.normalise(theta_phys, theta_phys_mean, theta_phys_std)
    qtot_next = ydata_and_norm['qtot_next_train'][0]
    qtot_next_mean = ydata_and_norm['qtot_next_train'][1]
    qtot_next_std = ydata_and_norm['qtot_next_train'][2]
    qtot_next_norm = nn_norm.normalise(qtot_next, qtot_next_mean, qtot_next_std)
    theta_next = ydata_and_norm['theta_next_train'][0]
    theta_next_mean = ydata_and_norm['theta_next_train'][1]
    theta_next_std = ydata_and_norm['theta_next_train'][2]
    theta_next_norm = nn_norm.normalise(theta_next, theta_next_mean, theta_next_std)

    xvars = [qtot_norm, qadv_norm, theta_norm, theta_adv_norm, sw_toa_norm, shf_norm, lhf_norm]
    # yvars = [qphys_norm]
    yvars = [theta_next_norm]

    X_train = np.hstack(xvars)
    y_train = np.hstack(yvars)

    return X_train, y_train    

def calc_PCA(X, y, location, nlevs):
    # print(xtrain.shape)
    # print(ytrain.shape)
    # xpca = PCA(n_components=200)
    xpca = PCA(n_components=0.99)
    # xpca.fit(X)
    # ypca = PCA(n_components=70)
    ypca = PCA(n_components=0.99)
    ypca.fit(y)
    # U, S, VT = np.linalg.svd(X_train - X_train.mean(0))
    # assert_array_almost_equal(VT[:30], pca.components_)

    # X_pca = xpca.transform(X)
    # y_pca = ypca.transform(y)
    # print(xpca.mean_.shape, xpca.components_.shape)
    print(ypca.mean_.shape, ypca.components_.shape)
    # X_train_pca2 = (X_train - pca.mean_).dot(pca.components_.T)
    # assert_array_almost_equal(X_train_pca, X_train_pca2)
    # print(X_train_pca.shape)
    # X_projected = xpca.inverse_transform(X_pca)
    # y_projected = ypca.inverse_transform(y_pca)
    # X_projected2 = X_train_pca.dot(pca.components_) + pca.mean_
    # assert_array_almost_equal(X_projected, X_projected2)
    # print(X_train.shape, X_projected.shape)
    print("Saving PCA models")
    # joblib.dump(xpca, location+"xpca_vars_{0}_nlevs_{1}_pcs_{2}.joblib".format(X.shape[1],nlevs, xpca.components_.shape[0]))
    joblib.dump(ypca, location+"ypca_tnext_{0}_nlevs_{1}_pcs_{2}.joblib".format(y.shape[1],nlevs, ypca.components_.shape[0]))

    return (0,0,0,0) #X_pca, y_pca, xpca, ypca

    # for lev in range(nlevs):
    # # lev = 0
    #     fig, axs = plt.subplots(3,1,figsize=(14,10), sharex=True)
    #     ax = axs[0]
    #     # ax.plot(X_train[:,lev], label='qtot')
    #     ax.hist(X_train[:,lev], 150, label='qtot')
    #     ax.legend()

    #     ax = axs[1]
    #     # ax.plot(X_projected[:,lev], label='qtot recon')
    #     ax.hist(X_projected[:,lev],150, label='qtot recon')
    #     ax.legend()

    #     ax = axs[2]
    #     # ax.plot(X_projected[:,lev]- X_train[:,lev], label='recon - real')
    #     ax.hist(X_train_pca[lev,:], 150, label='PCA')
    #     ax.legend()
    #     plt.show()

def gp_fit(X_train, y_train, X_test, y_test):
    # Instantiate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, copy_X_train=False)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X_train, y_train)

    
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(X_test, return_std=True)

    return y_pred, sigma

if __name__ == "__main__":
    training_data_file = "/project/spice/radiation/ML/CRM/data/models/datain/train_data_023001AQT.hdf5"
    testing_data_file = "/project/spice/radiation/ML/CRM/data/models/datain/validation_0N100W/validation_data_0N100W_015.hdf5"
    # normaliser = "/project/spice/radiation/ML/CRM/data/models/normaliser/021501AQ3H_normalise/"
    normaliser = "/project/spice/radiation/ML/CRM/data/models/normaliser/023001AQ_standardise_mx/"
    # normaliser = "/project/spice/radiation/ML/CRM/data/models/normaliser/023001AQ_normalise/"
    data_norm = data_io.NormalizersData(normaliser)
    data_frac = 1.
    nlevs = 70
    X_train, y_train = load_dataset(training_data_file,data_norm, data_frac, nlevs)
    X_test, y_test = validation_data(testing_data_file, data_norm, nlevs)
    pca_file_location = "/project/spice/radiation/ML/CRM/data/models/normaliser/023001AQ_pca/"
    X_train_pca, y_train_pca, xpca, ypca = calc_PCA(X_train, y_train, pca_file_location, nlevs)
    # print("X_train_pca", X_train_pca.shape)
    # X_test_pca, y_test_pca = xpca.transform(X_test), ypca.transform(y_test)
    # y_pred, sigma = gp_fit(X_train_pca, y_train_pca, X_test_pca, y_test_pca)
    # y_pred_inv = ypca.inverse_transform(y_pred)
    # pickle.dump(y_pred_inv, open("ypred.p","wb"))
    # for lev in range(nlevs):
    # # lev = 0
    #     fig, axs = plt.subplots(3,1,figsize=(14,10), sharex=True)
    #     ax = axs[0]
    #     ax.plot(y_test[:,lev], label='qphys')
    #     # ax.hist(X_train[:,lev], 150, label='qtot')
    #     ax.legend()

    #     ax = axs[1]
    #     ax.plot(y_pred_inv[:,lev], label='qphys pred')
    #     # ax.hist(X_projected[:,lev],150, label='qtot recon')
    #     ax.legend()

    #     ax = axs[2]
    #     ax.plot(y_pred_inv[:,lev]- y_test[:,lev], label='pred - true')
    #     # ax.hist(X_train_pca[lev,:], 150, label='PCA')
    #     ax.legend()
    #     plt.show()
