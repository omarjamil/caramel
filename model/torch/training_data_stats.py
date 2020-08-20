import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import data_io
from scipy import stats
from sklearn.decomposition import PCA
import numpy as np
from numpy.testing import assert_array_almost_equal
import data_io_batched
import torch

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def truncate_data(dataset: np.array([]), n_sigma: int=3):
    """
    Truncate data to n*sigma values
    """
    idx = (np.abs(stats.zscore(dataset,axis=0)) < n_sigma).all(axis=1)
    d = dataset[idx]
    return d

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

def train_dataloader_batched(args):
    train_dataset_file = "{0}/train_data_{1}.hdf5".format(args.locations["train_test_datadir"],args.region)
    train_dataset = data_io_batched.ConcatDataset("train",args.nlevs, train_dataset_file, args.locations['normaliser_loc'], args.batch_size, xvars=args.xvars,
             yvars=args.yvars, yvars2=args.yvars2, samples_frac=args.samples_fraction, data_frac=args.data_fraction, no_norm=args.no_norm)
    indices = list(range(train_dataset.__len__()))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=None, batch_size=None, sampler=train_sampler)
    return train_loader

def plot_distribution(dataset_file,nn_norm, data_frac, nlevs):
    """
    Plot histograms of the variables
    """
    args = dotdict()

    args.xvars=["qtot", "theta", "p", "rho", "xwind", "ywind", "zwind", "sw_toa", "shf", "lhf"]
    args.xvars2 = ['sw_toa', 'shf', 'lhf']
    args.yvars = ['qtot_next', 'theta_next']
    args.yvars2 = ['qphys', 'theta_phys']
    # args.normaliser = "023001AQT_normalise"
    # args.normaliser = "023001AQT_standardise_mx"
    # args.normaliser = "023001AQT_standardise"
    args.normaliser = "023001AQT_normalise"
    # args.normaliser = "023001AQT_normalise_60_glb"
    args.locations = {"train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain",
                    "normaliser_loc":"/project/spice/radiation/ML/CRM/data/models/normaliser/{0}".format(args.normaliser)}
    args.region = "023001AQTS"
    args.data_fraction = 1.
    args.samples_fraction = 100
    args.batch_size = 150
    args.nlevs = 60
    args.no_norm = True
    train_ldr = train_dataloader_batched(args)

    for batch_idx, batch in enumerate(train_ldr):
        x,y,y2 = batch
        qtot = x[0]
        qtot_next = y[0]
        qdiff = qtot_next - qtot
        theta = x[1]
        theta_next = y[1]
        tdiff = theta_next - theta
        p = x[2][1:] - x[2][:-1]
        rho = x[3][1:] - x[3][:-1]
        xwind = x[4][1:] - x[4][:-1]
        ywind = x[5][1:] - x[5][:-1]
        zwind = x[6][1:] - x[6][:-1]
        sw = x[7][1:] - x[7][:-1]
        shf = x[8][1:] - x[8][:-1]
        lhf = x[9][1:] - x[9][:-1]
        plt.style.use('ggplot')

        for lev in range(0,5,1):
        # lev = 0
            fig, axs = plt.subplots(2,2,figsize=(14,10), sharex=False)
            plt.title('Level {}'.format(lev))
            ax = axs[0,0]
            ax.hist(qtot[:,lev], 100, label='qtot')
            ax.legend()

            ax = axs[0,1]
            ax.hist(qdiff[:,lev]*10000., 100, label='qdiff')
            ax.legend()

            ax = axs[1,0]
            ax.hist(theta[:,lev], 100, label='theta')
            ax.legend()

            ax = axs[1,1]
            ax.hist(tdiff[:,lev]*10., 100, label='theta diff')
            ax.legend()

            fig, axs1 = plt.subplots(2,2,figsize=(14,10), sharex=False)
            ax = axs1[0,0]
            ax.hist(p[:,lev]/10., 100, label='p')
            ax.legend()

            ax = axs1[0,1]
            ax.hist(rho[:,lev]/1.e10, 100, label='rho')
            ax.legend()

            ax = axs1[1,0]
            ax.hist(xwind[:,lev], 100, label='xwind')
            ax.legend()

            ax = axs1[1,1]
            ax.hist(ywind[:,lev], 100, label='ywind')
            ax.legend()

            fig, axs2 = plt.subplots(2,2,figsize=(14,10), sharex=False)
            ax = axs2[0,0]
            ax.hist(zwind[:,lev]*100., 100, label='zwind')
            ax.legend()

            ax = axs2[0,1]
            ax.hist(sw[:,0]/100., 100, label='sw')
            ax.legend()

            ax = axs2[1,0]
            ax.hist(shf[:,0], 100, label='shf')
            ax.legend()

            ax = axs2[1,1]
            ax.hist(lhf[:,0]/10., 100, label='lhf')
            ax.legend()

            plt.show()

def plot_normal(mu,logvar):
    """
    """
    x = np.arange(-5,5,0.01)
    ys = []
    # logvariance to std
    std = np.exp(0.5*logvar)
    print(std)
    for m,s, in zip(mu,std):
        y = stats.norm(m,s)
        # print(y.pdf(x).shape)
        ys.append(y.pdf(x))
    fig, axs = plt.subplots(1,1,figsize=(14,10), sharex=False)
    for i,y in enumerate(ys):
        axs.plot(x,y,label=i)
    axs.legend()
    plt.show()

def PCA_projections(dataset_file,nn_norm, data_frac, nlevs):
    """
    Exploring PCA projections
    """    
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
    # qtot = truncate_data(qtot, n_sigma=2)
    qtot_mean = xdata_and_norm['qtot_train'][1]
    qtot_std = xdata_and_norm['qtot_train'][2]
    qtot_norm = nn_norm.normalise(qtot, qtot_mean, qtot_std)
    qadv = xdata_and_norm['qadv_train'][0]
    qadv_mean = xdata_and_norm['qadv_train'][1]
    qadv_std = xdata_and_norm['qadv_train'][2]
    qadv_norm = nn_norm.normalise(qadv, qadv_mean, qadv_std)
    qphys = ydata_and_norm['qphys_train'][0]
    qphys_mean = ydata_and_norm['qphys_train'][1]
    qphys_std = ydata_and_norm['qphys_train'][2]
    qphys_norm = nn_norm.normalise(qphys, qphys_mean, qphys_std)
    theta = xdata_and_norm['theta_train'][0]
    theta_mean = xdata_and_norm['theta_train'][1]
    theta_std = xdata_and_norm['theta_train'][2]
    theta_norm = nn_norm.normalise(theta, theta_mean, theta_std)
    theta_adv = xdata_and_norm['theta_adv_train'][0]
    theta_adv_mean = xdata_and_norm['theta_adv_train'][1]
    theta_adv_std = xdata_and_norm['theta_adv_train'][2]
    theta_adv_norm = nn_norm.normalise(theta_adv, theta_adv_mean, theta_adv_std)

    # X_train = np.random.randn(100, 50)
    X_train = qtot_norm
    # https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn
    pca = PCA(n_components=50)
    pca.fit(X_train)
    # U, S, VT = np.linalg.svd(X_train - X_train.mean(0))
    # assert_array_almost_equal(VT[:30], pca.components_)

    X_train_pca = pca.transform(X_train)
    print(pca.mean_.shape, pca.components_.shape)
    # X_train_pca2 = (X_train - pca.mean_).dot(pca.components_.T)
    # assert_array_almost_equal(X_train_pca, X_train_pca2)
    print(X_train_pca.shape)
    X_projected = pca.inverse_transform(X_train_pca)
    # X_projected2 = X_train_pca.dot(pca.components_) + pca.mean_
    # assert_array_almost_equal(X_projected, X_projected2)
    print(X_train.shape, X_projected.shape)
    for lev in range(nlevs):
    # lev = 0
        fig, axs = plt.subplots(3,1,figsize=(14,10), sharex=True)
        ax = axs[0]
        # ax.plot(X_train[:,lev], label='qtot')
        ax.hist(X_train[:,lev], 100, label='qtot')
        ax.legend()

        ax = axs[1]
        # ax.plot(X_projected[:,lev], label='qtot recon')
        ax.hist(X_projected[:,lev],100, label='qtot recon')
        ax.legend()

        ax = axs[2]
        # ax.plot(X_projected[:,lev]- X_train[:,lev], label='recon - real')
        ax.hist(X_train_pca[lev,:], 100, label='PCA')
        ax.legend()
        plt.show()

if __name__ == "__main__":
    # training_data_file = "/project/spice/radiation/ML/CRM/data/models/datain/train_data_023001AQ3HT.hdf5"
    training_data_file = "/project/spice/radiation/ML/CRM/data/models/datain/train_data_023001AQT.hdf5"
    # normaliser = "/project/spice/radiation/ML/CRM/data/models/normaliser/021501AQ3H_normalise/"
    # normaliser = "/project/spice/radiation/ML/CRM/data/models/normaliser/023001AQ_standardise_mx/"
    normaliser = "/project/spice/radiation/ML/CRM/data/models/normaliser/023001AQT_normalise/"
    # normaliser = "/project/spice/radiation/ML/CRM/data/models/normaliser/023001AQT_normalise_60_glb/"
    data_norm = data_io.NormalizersData(normaliser)
    data_frac = 0.01
    nlevs = 70
    # plot_variables(training_data_file,data_norm, data_frac, nlevs)
    plot_distribution(training_data_file,data_norm, data_frac, nlevs)
    # mu = np.array([ 0.0021, -0.0005,  0.1959])
    # logvar = np.array([-0.0147, -0.0031, -2.0244])
    # plot_normal(mu,logvar)
    # PCA_projections(training_data_file,data_norm, data_frac, nlevs)