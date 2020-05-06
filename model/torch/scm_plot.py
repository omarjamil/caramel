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
    # c = ax.pcolor(q_ml[:,:], vmin=vmin, vmax=vmax)
    c = ax.pcolor(q_ml[:,:])
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

def visualise_tseries(npfile,level):
    # data = np.load(np_file)
    data = h5py.File(npfile, 'r')
    q_ml = data['qtot_next_ml'][:]
    q_ = data['qtot_next'][:]
    q_y_lim = (np.min(q_[:,level]), np.max(q_[:,level]))
    # qphys_ml = data['qphys_ml'][:]
    # qphys = data['qphys'][:]
    # q_sane = data['q_sane'][:]
    t_ml = data['theta_next_ml'][:]
    t_ = data['theta_next'][:]
    t_y_lim = (np.min(t_[:,level]),np.max(t_[:,level]))
    # tphys_ml = data['tphys_ml'][:]
    # tphys = data['tphys'][:]
    # t_sane = data['t_sane'][:]
    

    fig, axs = plt.subplots(2,1,figsize=(14, 10),sharex=True)
    ax = axs[0]
    ax.plot(q_ml[:,level],'.-',label='q (ML)')
    ax.plot(q_[:,level],'.-',label='q')
    # ax.plot(q_sane[:,level],'.-',label='q (sane)')
    ax.set_title('Level {0}'.format(level))
    ax.set_ylim(q_y_lim[0],q_y_lim[1])
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
    ax.set_ylim(t_y_lim[0],t_y_lim[1])
    ax.legend()
    
    # ax = axs[1,1]
    # ax.plot(tphys_ml[:,level],'.-', label='Tphys (ML)')
    # ax.plot(tphys[:,level],'.-', label='Tphys')
    # ax.legend()

    # ax = axs[2,1]
    # ax.plot(tphys_ml[:,level] - tphys[:,level],'.-', label='Tphys (ML) - Tphys')
    # ax.legend()
    # plt.show()
    
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

def visualise_tseries_qT(npfile,level):
    # data = np.load(np_file)
    data = h5py.File(npfile, 'r')
    qphys_ml = data['qphys_predict'][:]
    qphys = data['qphys_test'][:]
    tphys_ml = data['tphys_predict'][:]
    tphys = data['tphys_test'][:]
    qphys_ml_norm = data['qphys_predict_norm'][:]
    qphys_norm = data['qphys_test_norm'][:]
    tphys_ml_norm = data['tphys_predict_norm'][:]
    tphys_norm = data['tphys_test_norm'][:]
    qtot_test = data['qtot_test_norm'][:]
    qadv_test = data['qadv_test_norm'][:]
    theta_test = data['theta_test_norm'][:]
    theta_adv_test= data['theta_adv_test_norm'][:]
    qphys_mean = np.mean(qphys,axis=0)

    fig, axs = plt.subplots(4,3,figsize=(14, 10),sharex=True)
    ax = axs[0,0]
    ax.plot(qphys_ml[:,level],'.-',label='qphys (ML)')
    ax.plot(qphys[:,level],'.-',label='qphys')
    ax.set_title('Level {0}'.format(level))
    ax.legend()
    
    ax = axs[1,0]
    ax.plot(tphys_ml[:,level],'.-', label='tphys (ML)')
    ax.plot(tphys[:,level],'.-', label='tqphys')
    ax.legend()

    ax = axs[2,0]
    ax.plot(qphys_ml[:,level] - qphys[:,level],'.-', label='qphys (ML) - qphys')
    ax.legend()

    ax = axs[3,0]
    ax.plot(tphys_ml[:,level] - tphys[:,level],'.-', label='Tphys (ML) - Tphys')
    ax.legend()

    ax = axs[0,1]
    ax.plot(qphys_ml_norm[:,level],'.-',label='qphys (ML) norm')
    ax.plot(qphys_norm[:,level],'.-',label='qphys norm')
    # ax.plot(qtot_test[:,level]-qadv_test[:,level],'.-', label='qtot*qadv norm')

    ax.set_title('Level {0}'.format(level))
    ax.legend()
    
    ax = axs[1,1]
    ax.plot(tphys_ml_norm[:,level],'.-', label='Tphys (ML) norm')
    ax.plot(tphys_norm[:,level],'.-', label='Tphys norm')
    ax.legend()

    ax = axs[2,1]
    ax.plot(qphys_ml_norm[:,level] - qphys_norm[:,level],'.-', label='qphys (ML) - qphys')
    ax.legend()

    ax = axs[3,1]
    ax.plot(tphys_ml_norm[:,level] - tphys_norm[:,level],'.-', label='Tphys (ML) - Tphys')
    ax.legend()

    ax = axs[0,2]
    ax.plot(qtot_test[:,level],'.-',label='qtot norm')
    ax.set_title('Level {0}'.format(level))
    ax.legend()
    
    ax = axs[1,2]
    ax.plot(qadv_test[:,level],'.-', label='qadv norm')
    ax.plot(qphys_norm[:,level],'.-',label='qphys norm')
    ax.plot(qtot_test[:,level] + qadv_test[:,level],'.-',label='qtot+qadv')
    ax.plot(qphys_ml_norm[:,level],'.-',label='qphys (ML) norm')


    # ax.plot(qtot_test[:,level]-qadv_test[:,level],'.-', label='qtot*qadv norm')
    ax.legend()

    ax = axs[2,2]
    ax.plot(theta_test[:,level],'.-', label='T norm')
    ax.legend()

    ax = axs[3,2]
    ax.plot(theta_adv_test[:,level],'.-', label='Tadv norm')
    # ax.plot(theta_test[:,level]+theta_adv_test[:,level],'.-', label='T*Tadv norm')
    ax.legend()

    plt.show()

def visualise_tseries_qT_add_adv(npfile,level):
    # data = np.load(np_file)
    data = h5py.File(npfile, 'r')
    qphys_ml = data['qphys_predict'][:]
    qphys = data['qphys_test'][:]
    tphys_ml = data['tphys_predict'][:]
    tphys = data['tphys_test'][:]
    qphys_ml_norm = data['qphys_predict_norm'][:]
    qphys_norm = data['qphys_test_norm'][:]
    tphys_ml_norm = data['tphys_predict_norm'][:]
    tphys_norm = data['tphys_test_norm'][:]
    qphys_mean = np.mean(qphys,axis=0)

    fig, axs = plt.subplots(4,2,figsize=(14, 10),sharex=True)
    ax = axs[0,0]
    ax.plot(qphys_ml[:,level],'.-',label='qphys (ML)')
    ax.plot(qphys[:,level],'.-',label='qphys')
    ax.set_title('Level {0}'.format(level))
    ax.legend()
    
    ax = axs[1,0]
    ax.plot(tphys_ml[:,level],'.-', label='tphys (ML)')
    ax.plot(tphys[:,level],'.-', label='tqphys')
    ax.legend()

    ax = axs[2,0]
    ax.plot(qphys_ml[:,level] - qphys[:,level],'.-', label='qphys (ML) - qphys')
    ax.legend()

    ax = axs[3,0]
    ax.plot(tphys_ml[:,level] - tphys[:,level],'.-', label='Tphys (ML) - Tphys')
    ax.legend()

    ax = axs[0,1]
    ax.plot(qphys_ml_norm[:,level],'.-',label='qphys (ML) norm')
    ax.plot(qphys_norm[:,level],'.-',label='qphys norm')
    # ax.plot(qtot_test[:,level]-qadv_test[:,level],'.-', label='qtot*qadv norm')

    ax.set_title('Level {0}'.format(level))
    ax.legend()
    
    ax = axs[1,1]
    ax.plot(tphys_ml_norm[:,level],'.-', label='Tphys (ML) norm')
    ax.plot(tphys_norm[:,level],'.-', label='Tphys norm')
    ax.legend()

    ax = axs[2,1]
    ax.plot(qphys_ml_norm[:,level] - qphys_norm[:,level],'.-', label='qphys (ML) - qphys')
    ax.legend()

    ax = axs[3,1]
    ax.plot(tphys_ml_norm[:,level] - tphys_norm[:,level],'.-', label='Tphys (ML) - Tphys')
    ax.legend()

    plt.show()

def visualise_tseries_qT_next(npfile,level):
    # data = np.load(np_file)
    data = h5py.File(npfile, 'r')
    qnext_ml = data['qtotn_predict'][:]
    qnext = data['qtotn_test'][:]
    # tnext_ml = data['thetan_predict'][:]
    # tnext = data['thetan_test'][:]
    qnext_ml_norm = data['qtotn_predict_norm'][:]
    qnext_norm = data['qtotn_test_norm'][:]
    # tnext_ml_norm = data['thetan_predict_norm'][:]
    # tnext_norm = data['thetan_test_norm'][:]
    # qphys_ml = data['qphys_ml'][:]
    # tphys_ml = data['theta_phys_ml'][:]
    qphys = data['qphys'][:]
    tphys = data['theta_phys'][:]
    qtot = data['qtot'][:]
    theta = data['theta'][:]

    fig, axs = plt.subplots(4,2,figsize=(14, 10),sharex=True)
    ax = axs[0,0]
    ax.plot(qnext_ml[:,level],'.-',label='qnext (ML)')
    ax.plot(qnext[:,level],'.-',label='qnext')
    ax.plot(qtot[:,level],'.-',label='qin')
    ax.set_title('Level {0}'.format(level))
    ax.legend()
    
    ax = axs[1,0]
    # ax.plot(tnext_ml[:,level],'.-', label='tnext (ML)')
    # ax.plot(tnext[:,level],'.-', label='tnext')
    ax.plot(theta[:,level],'.-', label='tin')
    ax.legend()

    ax = axs[2,0]
    ax.plot(qnext_ml[:,level] - qnext[:,level],'.-', label='qnext (ML) - qnext')
    ax.plot(qnext[:,level] - qtot[:,level],'.-', label='qnext - qin')
    ax.legend()

    ax = axs[3,0]
    # ax.plot(tnext_ml[:,level] - tnext[:,level],'.-', label='Tnext (ML) - Tnext')
    # ax.plot(tnext[:,level] - theta[:,level],'.-', label='Tnext - Tin')
    # ax.legend()

    ax = axs[0,1]
    # ax.plot(qnext_ml_norm[:,level],'.-',label='qnext (ML) norm')
    # ax.plot(qnext_norm[:,level],'.-',label='qnext norm')
    # ax.plot(qphys_ml[:,level], '.-', label='qphys (ML')
    ax.plot(qphys[:,level], '.-', label='qphys')
    # ax.plot(qtot_test[:,level]-qadv_test[:,level],'.-', label='qtot*qadv norm')

    ax.set_title('Level {0}'.format(level))
    ax.legend()
    
    ax = axs[1,1]
    # ax.plot(tphys_ml[:,level], '.-', label='tphys (ML')
    ax.plot(tphys[:,level], '.-', label='tphys')
    # ax.plot(tnext_ml_norm[:,level],'.-', label='Tnext (ML) norm')
    # ax.plot(tnext_norm[:,level],'.-', label='Tnext norm')
    ax.legend()

    ax = axs[2,1]
    # ax.plot(qphys_ml[:,level] - qphys[:,level], '.-', label='qphys (ML) - qphys')
    # ax.plot(qnext_ml_norm[:,level] - qnext_norm[:,level],'.-', label='qnext (ML) - qnext')
    # ax.legend()

    ax = axs[3,1]
    # ax.plot(tphys_ml[:,level] - tphys[:,level], '.-', label='tphys (ML) - tphys')
    # ax.plot(tnext_ml_norm[:,level] - tnext_norm[:,level],'.-', label='Tnext (ML) - Tnext')
    # ax.legend()

    plt.show()

def visualise_all_levels_qT(npfile):
    # data = np.load(np_file)
    data = h5py.File(npfile, 'r')
    qphys_ml = data['qphys_predict'][:]
    qphys = data['qphys_test'][:]
    tphys_ml = data['tphys_predict'][:]
    tphys = data['tphys_test'][:]
    # qphys_ml_norm = data['qphys_predict_norm'][:]
    # qphys_norm = data['qphys_test_norm'][:]
    # tphys_ml_norm = data['tphys_predict_norm'][:]
    # tphys_norm = data['tphys_test_norm'][:]
    # qtot_test = data['qtot_test_norm'][:]
    # qadv_test = data['qadv_test_norm'][:]
    # theta_test = data['theta_test_norm'][:]
    # theta_adv_test= data['theta_adv_test_norm'][:]

    fig, axs = plt.subplots(3,2,figsize=(14, 10), sharex=True)
    ax = axs[0,0]
    vmin,vmax=np.min(qphys_ml),np.max(qphys_ml)
    c = ax.pcolor(qphys_ml[:,:].T)
    ax.set_title('qphys (ML)')
    fig.colorbar(c,ax=ax)

    ax = axs[1,0]
    c = ax.pcolor(qphys[:,:].T)
    ax.set_title('qphys ')
    fig.colorbar(c,ax=ax)

    qdiff = qphys_ml[:].T - qphys[:].T
    ax = axs[2,0]
    c = ax.pcolor(qdiff[:,:])
    ax.set_title('Predict - Test')
    fig.colorbar(c,ax=ax)

    ax = axs[0,1]
    vmin,vmax=np.min(tphys_ml),np.max(tphys_ml)
    c = ax.pcolor(tphys_ml[:,:].T)
    ax.set_title('tphys (ML)')
    fig.colorbar(c,ax=ax)

    ax = axs[1,1]
    c = ax.pcolor(tphys[:,:].T, vmin=vmin, vmax=vmax)
    ax.set_title('tphys ')
    fig.colorbar(c,ax=ax)

    tdiff = tphys_ml[:].T - tphys[:].T
    ax = axs[2,1]
    c = ax.pcolor(tdiff[:,:])
    ax.set_title('Predict - Test')
    fig.colorbar(c,ax=ax)

    plt.show()

def visualise_all_levels_qTnext(npfile):
    # data = np.load(np_file)
    data = h5py.File(npfile, 'r')
    qtotn_ml = data['qtotn_predict'][:]
    qtotn = data['qtotn_test'][:]
    thetan_ml = data['thetan_predict'][:]
    thetan = data['thetan_test'][:]
    # qphys_ml_norm = data['qphys_predict_norm'][:]
    # qphys_norm = data['qphys_test_norm'][:]
    # tphys_ml_norm = data['tphys_predict_norm'][:]
    # tphys_norm = data['tphys_test_norm'][:]
    # qtot_test = data['qtot_test_norm'][:]
    # qadv_test = data['qadv_test_norm'][:]
    # theta_test = data['theta_test_norm'][:]
    # theta_adv_test= data['theta_adv_test_norm'][:]

    fig, axs = plt.subplots(3,2,figsize=(14, 10), sharex=True)
    ax = axs[0,0]
    vmin,vmax=np.min(qtotn_ml),np.max(qtotn_ml)
    c = ax.pcolor(qtotn_ml[:,:].T)
    ax.set_title('qtot_next (ML)')
    fig.colorbar(c,ax=ax)

    ax = axs[1,0]
    c = ax.pcolor(qtotn[:,:].T)
    ax.set_title('qtot_next ')
    fig.colorbar(c,ax=ax)

    qdiff = qtotn_ml[:].T - qtotn[:].T
    ax = axs[2,0]
    c = ax.pcolor(qdiff[:,:])
    ax.set_title('Predict - Test')
    fig.colorbar(c,ax=ax)

    ax = axs[0,1]
    vmin,vmax=np.min(thetan_ml),np.max(thetan_ml)
    c = ax.pcolor(thetan_ml[:,:].T)
    ax.set_title('t_next (ML)')
    fig.colorbar(c,ax=ax)

    ax = axs[1,1]
    c = ax.pcolor(thetan[:,:].T, vmin=vmin, vmax=vmax)
    ax.set_title('t_next ')
    fig.colorbar(c,ax=ax)

    tdiff = thetan_ml[:].T - thetan[:].T
    ax = axs[2,1]
    c = ax.pcolor(tdiff[:,:])
    ax.set_title('Predict - Test')
    fig.colorbar(c,ax=ax)

    plt.show()

def visualise_tseries_qphys(np_file,level):
    data = h5py.File(np_file, 'r')
    qphys_ml = data['qphys_predict'][:,:]
    qphys = data['qphys_test'][:,:]
    qphys_ml_norm = data['qphys_predict_norm'][:,:]
    qphys_norm = data['qphys_test_norm'][:,:]

    fig = plt.figure(figsize=(14,10))
    # fig, axs = plt.subplots(3,1,figsize=(14, 10), sharex=True)
    # ax0 = axs[0]
    ax0 = fig.add_subplot(321)
    ax0.plot(qphys_ml[:,level],'-.',label='qphys (ML)')
    ax0.plot(qphys[:,level],'-.',label='qphys')
    ax0.set_title('Level {0}'.format(level))
    ax0.legend()

    # ax1 = axs[1]
    ax1 = fig.add_subplot(323,sharex=ax0)
    ax1.plot(qphys_ml[:,level]-qphys[:,level],'.-',label='qphys (ML) -  qphys')
    ax1.set_title('Level {0}'.format(level))
    ax1.legend()

    # ax2 = axs[2]
    ax2 = fig.add_subplot(325)
    ax2.scatter(qphys[:,level], qphys_ml[:,level])
    ax2.plot(qphys[:,level],qphys[:,level],'k-')

    ax3 = fig.add_subplot(322)
    ax3.plot(qphys_ml_norm[:,level],'-.',label='qphys (ML) norm')
    ax3.plot(qphys_norm[:,level],'-.',label='qphys norm')
    ax3.set_title('Level {0}'.format(level))
    ax3.legend()

    # ax1 = axs[1]
    ax4 = fig.add_subplot(324,sharex=ax3)
    ax4.plot(qphys_ml_norm[:,level]-qphys_norm[:,level],'.-',label='qphys (ML) -  qphys norm')
    ax4.set_title('Level {0}'.format(level))
    ax4.legend()

    # ax2 = axs[2]
    ax5 = fig.add_subplot(326)
    ax5.scatter(qphys_norm[:,level], qphys_ml_norm[:,level])
    ax5.plot(qphys_norm[:,level],qphys_norm[:,level],'k-')
    
    xmin, xmax = np.min(qphys[:,level]), np.max(qphys[:,level])
    ymin, ymax = np.min(qphys_ml[:,level]), np.max(qphys_ml[:,level])

    ax2.set_xlim([xmin, xmax])
    ax2.set_ylim([ymin, ymax])
    ax2.set_xlabel('UM')
    ax2.set_ylabel('ML')
    ax2.set_title('Level {0}'.format(level))
    
    plt.show()

def visualise_tseries_tphys(np_file,level):
    data = h5py.File(np_file, 'r')
    tphys_ml = data['tphys_predict'][:6000,:]
    tphys = data['tphys_test'][:6000,:]
    fig = plt.figure(figsize=(14,10))
    # fig, axs = plt.subplots(3,1,figsize=(14, 10), sharex=True)
    # ax0 = axs[0]
    ax0 = fig.add_subplot(311)
    ax0.plot(tphys_ml[:,level],'-.',label='Tphys (ML)')
    ax0.plot(tphys[:,level],'-.',label='Tphys')
    ax0.set_title('Level {0}'.format(level))
    ax0.legend()

    # ax1 = axs[1]
    ax1 = fig.add_subplot(312,sharex=ax0)
    ax1.plot(tphys_ml[:,level]-tphys[:,level],'.-',label='Tphys (ML) -  Tphys')
    ax1.set_title('Level {0}'.format(level))
    ax1.legend()

    # ax2 = axs[2]
    ax2 = fig.add_subplot(313)
    ax2.scatter(tphys[:,level], tphys_ml[:,level])
    ax2.plot(tphys[:,level],tphys[:,level],'k-')
    xmin, xmax = np.min(tphys[:,level]), np.max(tphys[:,level])
    ymin, ymax = np.min(tphys_ml[:,level]), np.max(tphys_ml[:,level])
    ax2.set_xlim([xmin, xmax])
    ax2.set_ylim([ymin, ymax])
    ax2.set_xlabel('UM')
    ax2.set_ylabel('ML')
    ax2.set_title('Level {0}'.format(level))
    
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
    plt_title = model_file.split('/')[-1]
    ax.set_title('{0}'.format(plt_title))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
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
    model_name="q_qadv_t_tadv_swtoa_lhf_shf_qtphys_006_lyr_183_in_090_out_0210_hdn_030_epch_00500_btch_023001AQ3H_mse_163001AQ3H_normalise_levs"
    location = "/project/spice/radiation/ML/CRM/data/models/torch/"
    model_file = location+model_name+".tar"
    model_loss(model_file)
    
    np_file = model_name+"_scm.hdf5"
    np_file_2 = model_name+"_qtphys.hdf5"
    # average_tseries(np_file)
    figname = np_file.replace("hdf5","png")
    # visualise_all_levels_qT(np_file_2)
    # visualise_all_levels_qTnext(np_file_2)
    # visualise_scm_predictions_q(np_file,figname)
    # visualise_scm_predictions_qt(np_file,figname)
    for l in range(1,45,1):
        level=l
        # visualise_tseries(np_file, level)
        visualise_tseries_qphys(np_file_2,level)
        visualise_tseries_tphys(np_file_2,level)
        # visualise_tseries_qT(np_file_2,level)
        # visualise_tseries_qT_add_adv(np_file_2, level)
        # visualise_tseries_qT_next(np_file_2, level)
        # compare_qphys_predictions(np_file, np_file_2, level)