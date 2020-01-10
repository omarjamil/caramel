import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py

def visualise_inputs(np_file):
    dataset = np.load(np_file)
    qadv_train = dataset['qadv_train'].T 
    qadv_test = dataset['qadv_test'].T 
    q_train = dataset['q_train'].T
    q_test = dataset['q_test'].T 
    qphys_train = dataset['qphys_train'].T 
    qphys_test = dataset['qphys_test'].T 
    # Temperature variables 
    tadv_train = dataset['tadv_train'].T 
    tadv_test = dataset['tadv_test'].T 
    t_train = dataset['t_train'].T 
    t_test = dataset['t_test'].T 
    tphys_train = dataset['tphys_train'].T 
    tphys_test = dataset['tphys_test'].T 
       
    qcomb_train = np.concatenate((qadv_train,q_train),axis=1) 
    qcomb_test  = np.concatenate((qadv_test,q_test),axis=1) 
    tcomb_train = np.concatenate((tadv_train,t_train),axis=1) 
    tcomb_test  = np.concatenate((tadv_test,t_test),axis=1)

    fig, axs = plt.subplots(3,2,figsize=(14, 10))
    ax = axs[0,0]
    c = ax.pcolor(q_train[:,:5000])
    ax.set_title('Q Train')
    fig.colorbar(c,ax=ax)

    ax = axs[1,0]
    c = ax.pcolor(qadv_train[:,:5000])
    ax.set_title('Qadv Train')
    fig.colorbar(c,ax=ax)

    ax = axs[2,0]
    c = ax.pcolor(qphys_train[:,:5000])
    #c = ax.pcolor(qphys_test_norm[:,0:4000])
    ax.set_title('Qphys Train')
    fig.colorbar(c,ax=ax)

    ax = axs[0,1]
    c = ax.pcolor(t_train[:,:5000])
    ax.set_title('T Train')
    fig.colorbar(c,ax=ax)

    ax = axs[1,1]
    c = ax.pcolor(tadv_train[:,:5000])
    ax.set_title('Tadv Train')
    fig.colorbar(c,ax=ax)

    ax = axs[2,1]
    c = ax.pcolor(tphys_train[:,:5000])
    #c = ax.pcolor(qphys_test_norm[:,0:4000])
    ax.set_title('Tphys Train')
    fig.colorbar(c,ax=ax)
    
    plt.show()
    dataset.close()

def visualise_inputs_raw(np_file):
    dataset = np.load(np_file)
    qadv = dataset['qadv'].T 
    q = dataset['q'].T
    qphys = dataset['qphys'].T 
    # Temperature variables 
    tadv = dataset['tadv'].T 
    t = dataset['t'].T 
    tphys = dataset['tphys'].T

    qadv = np.ma.masked_where(qadv == 0., qadv)
    qphys = np.ma.masked_where(qphys == 0., qphys)
    tadv = np.ma.masked_where(tadv == 0., tadv)
    tphys = np.ma.masked_where(tphys == 0., tphys)
       
    fig, axs = plt.subplots(3,2,figsize=(14, 10))
    ax = axs[0,0]
    c = ax.pcolor(q[:,:5000])
    ax.set_title('Q')
    fig.colorbar(c,ax=ax)

    ax = axs[1,0]
    c = ax.pcolor(qadv[:,:5000])
    ax.set_title('Qadv')
    fig.colorbar(c,ax=ax)

    ax = axs[2,0]
    c = ax.pcolor(qphys[:,:5000])
    #c = ax.pcolor(qphys_test_norm[:,0:4000])
    ax.set_title('Qphys')
    fig.colorbar(c,ax=ax)

    ax = axs[0,1]
    c = ax.pcolor(t[:,:5000])
    ax.set_title('T')
    fig.colorbar(c,ax=ax)

    ax = axs[1,1]
    c = ax.pcolor(tadv[:,:5000])
    ax.set_title('Tadv')
    fig.colorbar(c,ax=ax)

    ax = axs[2,1]
    c = ax.pcolor(tphys[:,:5000])
    #c = ax.pcolor(qphys_test_norm[:,0:4000])
    ax.set_title('Tphys')
    fig.colorbar(c,ax=ax)
    
    plt.show()
    dataset.close()

    
def visualise_history(pklfile):
    data = pickle.load(open(pklfile, 'rb'))
    train_loss_results = data['training_loss']
    train_accuracy_results = data['training_accuracy']
    
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)
    plt.show()

def visualise_predictions_q(pred_file, figname):
    # data = np.load(np_file)
    data = h5py.File(pred_file, 'r')
    qphys_predict = data['qphys_predict'][:].T
    qphys_test = data['qphys_test'][:].T
    # qphys_test_norm = data['qphys_test_norm'].T
    
    fig, axs = plt.subplots(3,1,figsize=(14, 10))
    ax = axs[0]
    # c = ax.pcolor(qphys_predict[:,0:5000], vmin=-0.001, vmax=0.001)
    c = ax.pcolor(qphys_predict[:,0:5000])
    ax.set_title('q predict')
    fig.colorbar(c,ax=ax)

    ax = axs[1]
    # c = ax.pcolor(qphys_test[:,0:5000], vmin=-0.001, vmax=0.001)
    c = ax.pcolor(qphys_test[:,0:5000])
    ax.set_title('q Test')
    fig.colorbar(c,ax=ax)

    diff = qphys_predict - qphys_test
    ax = axs[2]
    # c = ax.pcolor(diff[:,0:5000], vmin=-0.001, vmax=0.001)
    c = ax.pcolor(diff[:,0:5000])
    #c = ax.pcolor(qphys_test_norm[:,0:4000])
    ax.set_title('Predict - Test')
    fig.colorbar(c,ax=ax)

    print("Saving figure {0}".format(figname))
    plt.savefig(figname)
    # plt.close(fig)
    plt.show()
    data.close()

def visualise_predictions_qadv(np_file, figname):
    data = np.load(np_file)
    qadv_predict = data['qadv_predict'].T
    qadv_test = data['qadv_test'].T
    # qphys_test_norm = data['qphys_test_norm'].T
    vmin,vmax=np.min(qadv_test),np.max(qadv_test)
    
    fig, axs = plt.subplots(3,1,figsize=(14, 10))
    ax = axs[0]
    c = ax.pcolor(qadv_predict[:,0:5000], vmin=vmin, vmax=vmax)
    # c = ax.pcolor(qadv_predict[:,0:5000])
    ax.set_title('qadv predict')
    fig.colorbar(c,ax=ax)

    ax = axs[1]
    c = ax.pcolor(qadv_test[:,0:5000], vmin=vmin, vmax=vmax)
    # c = ax.pcolor(qadv_test[:,0:5000])
    ax.set_title('qadv test')
    fig.colorbar(c,ax=ax)

    diff = qadv_predict - qadv_test
    ax = axs[2]
    # c = ax.pcolor(diff[:,0:5000], vmin=-0.001, vmax=0.001)
    c = ax.pcolor(diff[:,0:5000])
    #c = ax.pcolor(qphys_test_norm[:,0:4000])
    ax.set_title('Predict - Test')
    fig.colorbar(c,ax=ax)

    print("Saving figure {0}".format(figname))
    plt.savefig(figname)
    # plt.close(fig)
    plt.show()
    data.close()

def visualise_predictions_t(np_file, figname):
    data = np.load(np_file)
    qphys_predict = data['tphys_predict'].T
    qphys_test = data['tphys_test'].T
    # qphys_test_norm = data['qphys_test_norm'].T
    
    fig, axs = plt.subplots(3,1,figsize=(14, 10))
    ax = axs[0]
    c = ax.pcolor(qphys_predict[:,0:5000], vmin=-1.1, vmax=1.1)
    ax.set_title('T predict')
    fig.colorbar(c,ax=ax)

    ax = axs[1]
    c = ax.pcolor(qphys_test[:,0:5000], vmin=-1.1, vmax=1.1)
    ax.set_title('T Test')
    fig.colorbar(c,ax=ax)

    diff = qphys_predict - qphys_test
    ax = axs[2]
    c = ax.pcolor(diff[:,0:5000], vmin=-1.1, vmax=1.1)
    #c = ax.pcolor(qphys_test_norm[:,0:4000])
    ax.set_title('Predict - Test')
    fig.colorbar(c,ax=ax)

    print("Saving figure {0}".format(figname))
    plt.savefig(figname)
    # plt.close(fig)
    plt.show()
    data.close()
    
def visualise_predictions_qT(np_file, figname):
    data = np.load(np_file)
    qphys_predict = data['qphys_predict'].T
    qphys_test = data['qphys_test'].T
    tphys_predict = data['tphys_predict'].T
    tphys_test = data['tphys_test'].T
    
    fig, axs = plt.subplots(3,2,figsize=(14, 10))
    ax = axs[0,0]
    c = ax.pcolor(qphys_predict[:,0:5000])
    ax.set_title('Q predict')
    fig.colorbar(c,ax=ax)

    ax = axs[1,0]
    c = ax.pcolor(qphys_test[:,0:5000])
    ax.set_title('Q Test')
    fig.colorbar(c,ax=ax)

    diff = qphys_predict - qphys_test
    ax = axs[2,0]
    c = ax.pcolor(diff[:,0:5000])
    #c = ax.pcolor(qphys_test_norm[:,0:4000])
    ax.set_title('Predict - Test')
    fig.colorbar(c,ax=ax)

    ax = axs[0,1]
    c = ax.pcolor(tphys_predict[:,0:5000])
    ax.set_title('T predict')
    fig.colorbar(c,ax=ax)

    ax = axs[1,1]
    c = ax.pcolor(tphys_test[:,0:5000])
    ax.set_title('T Test')
    fig.colorbar(c,ax=ax)

    diff = tphys_predict - tphys_test
    ax = axs[2,1]
    c = ax.pcolor(diff[:,0:5000])
    #c = ax.pcolor(qphys_test_norm[:,0:4000])
    ax.set_title('Predict - Test')
    fig.colorbar(c,ax=ax)

    
    print("Saving figure {0}".format(figname))
    plt.savefig(figname)
    # plt.close(fig)
    plt.show()
    data.close()
    
def prediction_scatter(np_file, figname):
    """
    Create a scatter plot of predictions vs truth
    """
    data = np.load(np_file)
    qphys_predict = data['qphys_predict'].T
    qphys_test = data['qphys_test'].T
    qphys_test_norm = data['qphys_test_norm'].T

    rows, columns = 5,7
    fig, axs = plt.subplots(5,7,figsize=(18, 14))
    plt.rcParams.update({'font.size': 8})
    lev = 0
    for i in range(rows):
        for j in range(columns):
            ax = axs[i,j]
            c = ax.plot(qphys_test[lev,:],qphys_predict[lev,:],'x',)
            ax.set_title('Model lev {0}'.format(lev))
            ax.set_xlabel('Data',fontsize=6)
            ax.set_ylabel('Prediction',fontsize=6)
            lev += 1
    plt.subplots_adjust(top=0.964,
                        bottom=0.056,
                        left=0.054,
                        right=0.986,
                        hspace=0.585,
                        wspace=0.548)
    
    print("Saving figure {0}".format(figname))
    plt.savefig(figname)
    # plt.close(fig)
    plt.show()
    data.close()
    
if __name__ == "__main__":
    #np_file = '/project/spice/radiation/ML/CRM/data/models/train_test_data_50S69W_.npz'
    #visualise_inputs(np_file)

    np_file = '/project/spice/radiation/ML/CRM/data/models/raw_data_50S69W.npz'
    visualise_inputs_raw(np_file)
    
