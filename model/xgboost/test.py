import xgboost
import numpy as np 
import h5py
import joblib
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

import data_io

def scm(model_name, nlevs):
    """
    SCM type run with qt_next prediction model
    """
    print("loading saved model {0}".format(model_name))
    model = joblib.load(model_name)
    validation_file = "/project/spice/radiation/ML/CRM/data/models/datain/validation_0N100W1H/validation_data_0N100W_015.hdf5"
    norm = data_io.NormalizersData("/project/spice/radiation/ML/CRM/data/models/normaliser/021501AQ1H_standardise_mx", nlevs)
    x_test, y_test = data_io.get_val_data(validation_file, nlevs)
    # x_test, y_test = data_io.get_val_data(validation_file, nlevs, normaliser="/project/spice/radiation/ML/CRM/data/models/normaliser/021501AQ_standardise_mx")
    q_tot_test, q_tot_adv_test, theta_test, theta_adv_test, sw_toa_test, shf_test, lhf_test = x_test
    qnext_test, theta_next_test = y_test
    n_steps = q_tot_test.shape[0]
    qnext_ml = np.zeros((n_steps,nlevs))
    theta_next_ml = np.zeros((n_steps,nlevs))
    # model = set_model(args)
    q_in = q_tot_test[0,:]
    theta_in = theta_test[0,:]
    

    for t in range(n_steps):
        qnext_ml[t,:] = q_in
        theta_next_ml[t,:] = theta_in
        x_test = np.concatenate([q_in, q_tot_adv_test[t,:], theta_in, theta_adv_test[t,:], sw_toa_test[t], shf_test[t], lhf_test[t]]).reshape(1,-1)
        prediction = model.predict(x_test)
        qnext_pred = prediction[...,:nlevs]
        # tnext_pred = prediction[...,nlevs:]
        tnext_pred = theta_next_test[t,:]
        # print(qnext_pred[0,5], qnext_test[t,5])
        q_in = qnext_pred[0,:]
        # theta_in = tnext_pred[0,:]
        theta_in = tnext_pred

    # qnext_test = norm.inverse_transform(qnext_test, norm.q_mean, norm.q_stdscale)
    # theta_next_test = norm.inverse_transform(theta_next_test, norm.t_mean, norm.t_stdscale)

    # qnext_ml = norm.inverse_transform(qnext_ml, norm.q_mean, norm.q_stdscale)
    # theta_next_ml = norm.inverse_transform(theta_next_ml, norm.t_mean, norm.t_stdscale)

    output = {'qtot_next':qnext_test, 'qtot_next_ml':qnext_ml,
            'theta_next':theta_next_test, 'theta_next_ml':theta_next_ml
            }
    hfilename = model_name.replace('.joblib','_scm.hdf5') 
    with h5py.File(hfilename, 'w') as hfile:
        for k, v in output.items():  
            hfile.create_dataset(k,data=v)


def test(model_name):
    validation_file = "/project/spice/radiation/ML/CRM/data/models/datain/validation_0N100W1H/validation_data_0N100W_063.hdf5"
    nlevs = 45
    #[q_tot_test, q_tot_adv_test, theta_test, theta_adv_test, sw_toa_test, shf_test, lhf_test], [qnext_test, theta_next_test]
    norm = data_io.NormalizersData("/project/spice/radiation/ML/CRM/data/models/normaliser/021501AQ1H_standardise_mx", nlevs)
    # x_t, y_t = data_io.get_val_data(validation_file, nlevs, normaliser="/project/spice/radiation/ML/CRM/data/models/normaliser/021501AQ_standardise_mx")
    x_t, y_t = data_io.get_val_data(validation_file, nlevs)
    x_test = np.concatenate(x_t, axis=1)
    y_test = np.concatenate(y_t, axis=1)
    print("loading saved model {0}".format(model_name))
    model = joblib.load(model_name)
    prediction = model.predict(x_test[:,:])
    q_predict = prediction[:,:nlevs]
    # theta_predict = prediction[:,nlevs:]
    # q_predict = norm.inverse_transform(q_predict, norm.q_mean, norm.q_stdscale)
    # theta_predict = norm.inverse_transform(theta_predict, norm.t_mean, norm.t_stdscale)
    # q_test = norm.inverse_transform(x_t[0], norm.q_mean, norm.q_stdscale)
    # theta_test = norm.inverse_transform(x_t[2], norm.t_mean, norm.t_stdscale)
    # q_next = norm.inverse_transform(y_t[0],  norm.q_mean, norm.q_stdscale)
    # theta_next = norm.inverse_transform(y_t[1], norm.t_mean, norm.t_stdscale)
    q_test = x_t[0]
    theta_test = x_t[2]
    q_next = y_t[0]
    theta_next = y_t[1]
    theta_predict = theta_next

    # print(x_test.shape, y_test.shape, prediction.shape)
    # print(y_test[0,:])
    # print(prediction[0,:])
    # for t in range(50):
    #     fig, ax = plt.subplots(2,2, figsize=(14, 10),sharex=True)
    #     ax[0,0].set_title("Time: {0}".format(t))
    #     ax[0,0].plot(q_predict[t,:],'-o', label='q pred')
    #     ax[0,0].plot(q_test[t,:],'-o',label='q test')
    #     ax[0,0].plot(q_next[t,:],'-o',label='q next')
    #     ax[0,0].legend()
    #     ax[1,0].plot(theta_predict[t,:],'-o', label='theta predict')
    #     ax[1,0].plot(theta_test[t,:],'-o', label='theta test')
    #     ax[1,0].plot(theta_next[t,:],'-o', label='theta next')
    #     ax[1,0].legend()
    #     ax[0,1].plot(q_next[t,:] - q_predict[t,:],label='q_next - ml')
    #     ax[0,1].plot(q_next[t,:] - q_test[t,:],label='q_next - in')
    #     ax[0,1].legend()
    #     ax[1,1].plot(theta_next[t,:] - theta_predict[t,:],label='theta_next - ml')
    #     ax[1,1].plot(theta_next[t,:] - theta_test[t,:],label='theta_next - in')
    #     ax[1,1].legend()

    #     plt.show()

    for l in range(nlevs):
        # print("Divisor: {0}".format(divisor))
        fig, ax = plt.subplots(2,2, figsize=(14, 10),sharex=True)
        ax[0,0].set_title("Level: {0}".format(l))
        ax[0,0].plot(q_predict[:,l],'-o', label='q pred')
        ax[0,0].plot(q_test[:,l],'-o',label='q test')
        ax[0,0].plot(q_next[:,l],'-o',label='q next')
        ax[0,0].legend()
        ax[1,0].plot(theta_predict[:,l],'-o', label='theta predict')
        ax[1,0].plot(theta_test[:,l],'-o', label='theta test')
        ax[1,0].plot(theta_next[:,l],'-o', label='theta next')
        ax[1,0].legend()
        ax[0,1].plot(q_next[:,l] - q_predict[:,l],label='q_next - ml')
        ax[0,1].plot(q_next[:,l] - q_test[:,l],label='q_next - in')
        ax[0,1].legend()
        ax[1,1].plot(theta_next[:,l] - theta_predict[:,l],label='theta_next - ml')
        ax[1,1].plot(theta_next[:,l] - theta_test[:,l],label='theta_next - in')
        ax[1,1].legend()

        plt.show()

if __name__ == "__main__":
    model_name = "caramel_lgb_dfrac_020_qnext_lr_p01.joblib"
    # test(model_name)
    nlevs=45
    scm(model_name, nlevs)
    # test_multi_regressor()