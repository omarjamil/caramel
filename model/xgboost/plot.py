import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py

def visualise_tseries(npfile,level):
    # data = np.load(np_file)
    data = h5py.File(npfile, 'r')
    q_ml = data['qtot_next_ml'][:]
    q_ = data['qtot_next'][:]
    # qphys_ml = data['qphys_ml'][:]
    # qphys = data['qphys'][:]
    # q_sane = data['q_sane'][:]
    t_ml = data['theta_next_ml'][:]
    t_ = data['theta_next'][:]
    # tphys_ml = data['tphys_ml'][:]
    # tphys = data['tphys'][:]
    # t_sane = data['t_sane'][:]
    

    fig, axs = plt.subplots(2,1,figsize=(14, 10),sharex=True)
    ax = axs[0]
    ax.plot(q_ml[:,level],'.-',label='q (ML)')
    ax.plot(q_[:,level],'.-',label='q')
    # ax.plot(q_sane[:,level],'.-',label='q (sane)')
    ax.set_title('Level {0}'.format(level))
    ax.legend()
    
    # ax = axs[1,0]
    # ax.plot(qphys_ml[:,level],'.-', label='qphys (ML)')
    # ax.plot(qphys[:,level],'.-', label='qphys')
    # ax.legend()

    # ax = axs[2,0]
    # ax.plot(qphys_ml[:,level] - qphys[:,level],'.-', label='qphys (ML) - qphys')
    # ax.legend()

    ax = axs[1]
    ax.plot(t_ml[:,level],'.-',label='T (ML)')
    ax.plot(t_[:,level],'.-',label='T')
    # ax.plot(t_sane[:,level],'.-',label='T (sane)')
    ax.set_title('Level {0}'.format(level))
    ax.legend()
    
    # ax = axs[1,1]
    # ax.plot(tphys_ml[:,level],'.-', label='Tphys (ML)')
    # ax.plot(tphys[:,level],'.-', label='Tphys')
    # ax.legend()

    # ax = axs[2,1]
    # ax.plot(tphys_ml[:,level] - tphys[:,level],'.-', label='Tphys (ML) - Tphys')
    # ax.legend()
    plt.show()

def visualise_scm_predictions_qt(np_file, figname):
    # data = np.load(np_file)
    data = h5py.File(np_file, 'r')
    
    q_ml = data['qtot_next_ml'][:].T
    q_ = data['qtot_next'][:].T
    q_ml = np.ma.masked_where(q_ml <= 0.0, q_ml)
    q_ = np.ma.masked_where(q_ == 0.0, q_)
    t_ml = data['theta_next_ml'][:].T
    t_ = data['theta_next'][:].T
    t_ml = np.ma.masked_where(t_ml <= 0.0, t_ml)
    t_ = np.ma.masked_where(t_ == 0.0, t_)
    
    fig, axs = plt.subplots(3,2,figsize=(14, 10))
    ax = axs[0,0]
    vmin,vmax=np.min(q_),np.max(q_)
    c = ax.pcolor(q_ml[:,:])
    ax.set_title('q (ML)')
    fig.colorbar(c,ax=ax)

    ax = axs[1,0]
    c = ax.pcolor(q_[:,:])
    ax.set_title('q ')
    fig.colorbar(c,ax=ax)

    qdiff = q_ml - q_
    ax = axs[2,0]
    c = ax.pcolor(qdiff[:,:])
    ax.set_title('Predict - Test')
    fig.colorbar(c,ax=ax)

    ax = axs[0,1]
    vmin,vmax=np.min(t_),np.max(t_)
    c = ax.pcolor(t_ml[:,:])
    ax.set_title('T (ML)')
    fig.colorbar(c,ax=ax)

    ax = axs[1,1]
    c = ax.pcolor(t_[:,:])
    ax.set_title('T ')
    fig.colorbar(c,ax=ax)

    tdiff = t_ml - t_
    ax = axs[2,1]
    c = ax.pcolor(tdiff[:,:])
    ax.set_title('Predict - Test')
    fig.colorbar(c,ax=ax)
    
    # print("Saving figure {0}".format(figname))
    # plt.savefig(figname)
    # plt.close(fig)
    plt.show()
    data.close()

if __name__ == "__main__":
    model_name = "caramel_lgb_dfrac_020_qnext_lr_p01"
    np_file = model_name+"_scm.hdf5"
    figname = "blah.png"
    visualise_scm_predictions_qt(np_file, figname)
    for l in range(1,45,1):
        level=l
        visualise_tseries(np_file, level)