import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py

def visualise_scm_predictions_q(np_file, figname):
    # data = np.load(np_file)
    data = h5py.File(np_file, 'r')
    
    q_ml = data['q_ml'][:].T
    q_ = data['q'][:].T
    # qphys_test_norm = data['qphys_test_norm'].T
    
    fig, axs = plt.subplots(3,1,figsize=(14, 10))
    ax = axs[0]
    vmin,vmax=np.min(q_),np.max(q_)
    print(vmin,vmax)
    c = ax.pcolor(q_ml[:,:], vmin=vmin, vmax=vmax)
    # c = ax.pcolor(q_ml[:,:])
    ax.set_title('q (ML)')
    fig.colorbar(c,ax=ax)

    ax = axs[1]
    c = ax.pcolor(q_[:,:], vmin=vmin, vmax=vmax)
    # c = ax.pcolor(q_[:,:])
    ax.set_title('q ')
    fig.colorbar(c,ax=ax)

    diff = q_ml - q_
    ax = axs[2]
    # c = ax.pcolor(diff[:,:], vmin=-0.001, vmax=0.001)
    c = ax.pcolor(diff[:,:])
    #c = ax.pcolor(qphys_test_norm[:,0:4000])
    ax.set_title('Predict - Test')
    fig.colorbar(c,ax=ax)

    print("Saving figure {0}".format(figname))
    plt.savefig(figname)
    # plt.close(fig)
    plt.show()
    data.close()

def visualise_tseries(npfile,level):
    # data = np.load(np_file)
    data = h5py.File(npfile, 'r')
    q_ml = data['q_ml'][:]
    q_ = data['q'][:]
    qphys_ml = data['qphys_ml'][:]
    qphys = data['qphys'][:]
    qadv = data['qadv'][:]
    qadv_un = data['qadv_raw'][:]
    qadv_dot = data['qadv_dot'][:]
    qadv_dot_un = data['qadv_dot_raw'][:]*600.
    q_sane = data['q_sane'][:]
    qphys_drift = data['qphys_drift'][:]
    qphys_pred_drift = data['qphys_pred_drift'][:]
    
    fig, axs = plt.subplots(4,1,figsize=(14, 10))
    ax = axs[0]
    ax.plot(q_ml[:,level],'.-',label='q (ML)')
    ax.plot(q_[:,level],'.-',label='q')
    ax.plot(q_sane[:,level],'.-',label='q (sane)')
    ax.set_title('Level {0}'.format(level))
    ax.legend()
    
    ax = axs[1]
    #ax.plot(qadv[:,level],'.-', label='qadv')
    ax.plot(qadv_un[:,level],'.-', label='qadv (un)')
    #ax.plot(qadv_dot[:,level],'.-', label='qadv_dot')
    ax.plot(qadv_dot_un[:,level],'.-', label='qadv_dot (un)')
    ax.legend()

    ax = axs[2]
    ax.plot(qphys_ml[:,level],'.-', label='qphys (ML)')
    ax.plot(qphys[:,level],'.-', label='qphys')
    ax.legend()
    
    ax = axs[3]
    ax.plot(q_ml[:,level] - q_[:,level],'.-', label='q (ML) - q')
    ax.plot(qphys_ml[:,level] - qphys[:,level],'.-', label='qphys (ML) - qphys')
    ax.plot(qphys_drift[:,level], '.-', label='qphys (drift)')
    ax.plot(qphys_pred_drift[:,level], '.-', label='qphys_pred (drift)')
    ax.legend()
    plt.show()
    
def visualise_tseries_q(np_file,level):
    data = np.load(np_file)
    q_ml = data['q_ml']
    q_ = data['q']
    qphys_ml = data['qphys_ml']
    qphys = data['qphys']
    fig, axs = plt.subplots(2,1,figsize=(14, 10))
    ax = axs[0]
    ax.plot(q_ml[:,level],label='q (ML)')
    ax.plot(q_[:,level],label='q')
    ax.set_title('Level {0}'.format(level))
    ax.legend()
    
    ax = axs[1]
    ax.plot(q_ml[:,level] - q_[:,level], label='q (ML) - q')
    ax.plot(qphys_ml[:,level] - qphys[:,level], label='qphys (ML) - qphys')
    ax.legend()
    plt.show()
    
def visualise_tseries_qphys(np_file,level):
    data = np.load(np_file)
    qphys_ml = data['qphys_ml']
    qphys = data['qphys']
    fig, axs = plt.subplots(2,1,figsize=(14, 10))
    ax = axs[0]
    ax.plot(qphys_ml[:,level],'.-',label='qphys (ML)')
    ax.plot(qphys[:,level],'.-',label='qphys')
    ax.set_title('Level {0}'.format(level))
    ax.legend()

    ax = axs[1]
    ax.plot(qphys_ml[:,level]-qphys[:,level],'.-',label='qphys (ML) -  qphys')
    ax.set_title('Level {0}'.format(level))
    ax.legend()
    
    plt.show()

    
if __name__ == "__main__":
    np_file = "scm_predict_qloss.hdf5"
    figname = "scm_predict_qloss.png"
    visualise_scm_predictions_q(np_file,figname)
    for l in range(70):
        level=l
        visualise_tseries(np_file, level)
        # visualise_tseries_q(np_file,level)
        # visualise_tseries_qphys(np_file,level)
   
