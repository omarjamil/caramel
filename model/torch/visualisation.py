import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py

def visualise_predictions_q(pred_file, figname):
    # data = np.load(np_file)
    data = h5py.File(pred_file, 'r')
    qphys_predict = data['qphys_predict'][:].T
    qphys_test = data['qphys_test'][:].T
    # qphys_test_norm = data['qphys_test_norm'].T
    vmin,vmax = np.min(qphys_test),np.max(qphys_test)
    fig, axs = plt.subplots(3,1,figsize=(14, 10))
    ax = axs[0]
    c = ax.pcolor(qphys_predict[:,0:5000], vmin=vmin, vmax=vmax)
    # c = ax.pcolor(qphys_predict[:,0:5000])
    ax.set_title('q predict')
    fig.colorbar(c,ax=ax)

    ax = axs[1]
    c = ax.pcolor(qphys_test[:,0:5000], vmin=vmin, vmax=vmax)
    # c = ax.pcolor(qphys_test[:,0:5000])
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

if __name__ == "__main__":
    prediction_file = 'qcomb_add_dot_qloss_predict.hdf5'
    visualise_predictions_q(prediction_file,'q_predict_qloss.png')