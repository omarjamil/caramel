import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import torch

def visualise_scm_predictions_q(np_file, figname):
    # data = np.load(np_file)
    data = h5py.File(np_file, 'r')
    
    q_ml = data['q_ml'][:].T
    q_ = data['q'][:].T
    q_ml = np.ma.masked_where(q_ml <= 0.0, q_ml)
    q_ = np.ma.masked_where(q_ == 0.0, q_)
    # qphys_test_norm = data['qphys_test_norm'].T
    
    fig, axs = plt.subplots(3,1,figsize=(14, 10))
    ax = axs[0]
    vmin,vmax=np.min(q_),np.max(q_)
    # print(vmin,vmax)
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
    qadv_dot = data['qadv_dot'][:]
    qadv_dot_un = data['qadv_dot_raw'][:]*600.
    q_sane = data['q_sane'][:]
    qphys_drift = data['qphys_drift'][:]
    qphys_pred_drift = data['qphys_pred_drift'][:]
    tadv_dot = data['tadv_dot'][:]

    fig, axs = plt.subplots(3,1,figsize=(14, 10),sharex=True)
    ax = axs[0]
    ax.plot(q_ml[:,level],'.-',label='q (ML)')
    ax.plot(q_[:,level],'.-',label='q')
    ax.plot(q_sane[:,level],'.-',label='q (sane)')
    ax.set_title('Level {0}'.format(level))
    ax.legend()
    
    ax = axs[1]
    ax.plot(qphys_ml[:,level],'.-', label='qphys (ML)')
    ax.plot(qphys[:,level],'.-', label='qphys')

    # ax.plot(qadv_dot[:,level],'.-', label='qadv_dot')
    ax.plot(qadv_dot_un[:,level],'.-', label='qadv_dot (un)')
    #ax.plot((qadv_dot_un[:,level] - qadv_un[:,level]),'.-', label='qadv_dot (un) - qadv (un)')
    # ax.plot(tadv_dot[:,level], '.-', label='Tadv')
    
    ax.legend()

    ax = axs[2]
    # ax.plot(((q_ml[:,level] - q_[:,level])/q_[:,level])*100.,'.-', label='q (ML) - q')
    ax.plot(qphys_ml[:,level] - qphys[:,level],'.-', label='qphys (ML) - qphys')
    # ax.plot(qphys_drift[:,level], '.-', label='qphys (drift)')
    # ax.plot(qphys_pred_drift[:,level], '.-', label='qphys_pred (drift)')
    # ax.plot(tadv_dot[:,level], '.-', label='Tadv')
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
    data = h5py.File(np_file, 'r')
    qphys_ml = data['qphys_predict'][:6000,:]
    qphys = data['qphys_test'][:6000,:]
    fig, axs = plt.subplots(2,1,figsize=(14, 10), sharex=False)
    ax = axs[0]
    ax.plot(qphys_ml[:,level],'.-',label='qphys (ML)')
    ax.plot(qphys[:,level],'.',label='qphys')
    ax.set_title('Level {0}'.format(level))
    ax.legend()

    ax = axs[1]
    ax.scatter(qphys[:,level], qphys_ml[:,level])
    ax.plot(qphys[:,level],qphys[:,level],'k-')
    ax.set_xlabel('UM')
    ax.set_ylabel('ML')
    # ax.plot(qphys_ml[:,level]-qphys[:,level],'.-',label='qphys (ML) -  qphys')
    ax.set_title('Level {0}'.format(level))
    # ax.legend()
    
    plt.show()

def compare_qphys_predictions(np_file,np_file_2,level):
    data = h5py.File(np_file, 'r')
    data2 = h5py.File(np_file_2, 'r')
    qphys_ml = data['qphys_ml'][:1000,:]
    qphys_ml2 = data2['qphys_predict'][:1000,:]
    qphys = data['qphys'][:1000,:]
    qphys2 = data2['qphys_test'][:1000,:]

    fig, axs = plt.subplots(2,1,figsize=(14, 10), sharex=True)
    ax = axs[0]
    ax.plot(qphys_ml[:,level],'.-',label='scm run')
    ax.plot(qphys[:,level],'.-',label='qphys')
    ax.set_title('Level {0}'.format(level))
    ax.legend()

    ax = axs[1]
    ax.plot(qphys_ml2[:,level], '.-',label='qphys run')
    ax.plot(qphys2[:,level],'.-',label='qphys')
    ax.set_title('Level {0}'.format(level))
    ax.legend()
    
    plt.show()

def model_loss(model_file):
    """
    Visualise model loss
    """
    checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
    train_loss = checkpoint['training_loss']
    test_loss = checkpoint['validation_loss']
    fig, ax = plt.subplots(1,1,figsize=(14, 10))
    ax.plot(train_loss,'.-',label='Train Loss')
    ax.plot(test_loss,'.-',label='Test Loss')
    ax.set_title('Model loss')
    ax.legend()
    plt.show()

def nooverlap_smooth(arrayin, window=6):
    """
    Moving average with non-overlapping window
    """
    x,y=arrayin.shape
    averaged = np.mean(arrayin.reshape(window,x//window,y),axis=0)
    return averaged

def average_tseries(np_file):
    """
    Average the time series and compare
    """
    data = h5py.File(np_file, 'r')
    qphys = data['qphys'][:996,:]
    qphys_ml = data['qphys_ml'][:996,:]
    print(qphys.shape, qphys_ml.shape)
    qphys_ = nooverlap_smooth(qphys)
    qphys_ml_ = nooverlap_smooth(qphys_ml)
    for l in range(1,70,2):
        fig, ax = plt.subplots(1,1,figsize=(14, 10))
        ax.plot(qphys_ml_[:,l],'.-',label='qphys (ML)')
        ax.plot(qphys_[:,l],'.-',label='qphys')
        ax.set_title('Average qphys prediction vs truth')
        ax.legend()
        plt.show()

if __name__ == "__main__":
    model_name="q_qadv_t_tadv_swtoa_lhf_shf_qphys_006_lyr_123_in_030_out_0256_hdn_100_epch_02000_btch_9999NEWS"
    location = "/project/spice/radiation/ML/CRM/data/models/torch/"
    model_file = location+model_name+".tar"
    model_loss(model_file)
    
    np_file = model_name+".hdf5"
    np_file_2 = model_name+"_qphys.hdf5"
    # average_tseries(np_file)
    figname = np_file.replace("hdf5","png")
    # visualise_scm_predictions_q(np_file,figname)
    for l in range(1,30,5):
        level=l
        # visualise_tseries(np_file, level)
    #     # visualise_tseries_q(np_file,level)
        visualise_tseries_qphys(np_file_2,level)
        # compare_qphys_predictions(np_file, np_file_2, level)