import xgboost
import numpy as np 
import h5py
import joblib
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

import data_io




def train(model_name, datafrac):
    datafile = "/project/spice/radiation/ML/CRM/data/models/datain/test_train_data/train_test_data_021501AQ1H_raw.hdf5"
    # datafrac = 0.2
    nlevs = 45
    # x_train, y_train, x_test, y_test = data_io.get_data(datafile, datafrac, nlevs, normaliser="/project/spice/radiation/ML/CRM/data/models/normaliser/021501AQ_standardise_mx")
    x_train, y_train, x_test, y_test = data_io.get_data(datafile, datafrac, nlevs)
    print(x_train.shape, y_train.shape)
    # model = MultiOutputRegressor(xgboost.XGBRegressor(n_estimators=100), n_jobs=4)
    model = MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=100))
    model.fit(x_train, y_train)
    print("saving model {0}".format(model_name))
    joblib.dump(model, model_name)

def test_multi_regressor():
    # get some noised linear data
    X = np.random.random((1000, 10))
    a = np.random.random((10, 3))
    y = np.dot(X, a) + np.random.normal(0, 1e-3, (1000, 3))
    # fitting
    multioutputregressor = MultiOutputRegressor(xgboost.XGBRegressor(learning_rate=0.01), n_jobs=4).fit(X, y)
    # lgb.LGBMRegressor

    # predicting
    print(np.mean((multioutputregressor.predict(X) - y)**2, axis=0))  # 0.004, 0.003, 0.005

if __name__ == "__main__":
    model_name = "caramel_lgb_dfrac_020_qnext_lr_p01.joblib"
    datafrac = 0.3
    train(model_name, datafrac)
    # test(model_name)
    # nlevs=45
    # scm(model_name, nlevs)
    # test_multi_regressor()