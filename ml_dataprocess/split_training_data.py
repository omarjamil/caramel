"""
Split large training data into smaller files
"""
import h5py
import numpy as np

crm_data = "/project/spice/radiation/ML/CRM/data"

def split_file_train_test(infile, region, location):
    """
    Split file into training an testin data
    """
    train_data={}
    test_data={}
    datafile = h5py.File(location+infile,'r')
    for keys in datafile.keys():
        if "train" in keys:
            train_data[keys] = datafile[keys]
        if "test" in keys:
            test_data[keys] = datafile[keys]

    train_test_datadir = "{0}/models/datain/".format(crm_data)
    fname_train = 'train_data_{0}.hdf5'.format(region)
    fname_test = 'test_data_{0}.hdf5'.format(region)

    with h5py.File(train_test_datadir+fname_train, 'w') as hfile:
        for tr in train_data:
            print("Saving normalised training data {0} {1}".format(tr, train_data[tr].shape))
            hfile.create_dataset(tr,data=train_data[tr])

    with h5py.File(train_test_datadir+fname_test, 'w') as hfile:
        for te in test_data:
            print("Saving normalised testing data {0} {1}".format(te, test_data[te].shape))
            hfile.create_dataset(te,data=test_data[te])

def split_training_data(infile, n_splits, location):
    """
    Split single large training file into a n_splits smaller files
    """
    datafile = h5py.File(location+infile,'r')
    data = {}
    print("Reading {0}".format(location+infile))
    for keys in datafile.keys():
        data[keys] = np.array_split(datafile[keys],n_splits)
        for d in data[keys]:
            print(keys, d.shape)

    train_test_datadir = "{0}/models/datain/".format(crm_data)
    for n in range(n_splits):
        outfile=train_test_datadir+"train_test_data_021501AQ_{0}.h5".format(str(n).zfill(3))
        with h5py.File(outfile) as hfile:
            for k in data:
                var_data = data[k][n]
                print("Saving split data {0}: {1} of {2}".format(k,n,n_splits))
                hfile.create_dataset(k, data=var_data)


def main():
    location="/project/spice/radiation/ML/CRM/data/models/datain/"
    # infile="train_test_data_021501AQ_std.hdf5"
    # infile="train_test_data_023001AQ3H_raw.hdf5"
    # infile="train_test_data_9999LEAU_std.hdf5"
    # infile="train_test_data_023001AQ3HT_raw.hdf5""
    # infile="train_test_data_023001AQT_raw.hdf5"
    # infile="cnn_train_test_data_023001AQTT3_t19_raw.hdf5"
    # infile="train_test_data_023001AQTS_stacked.hdf5"
    # infile="train_test_data_023001AQS_stacked_raw.hdf5"
    # infile="train_test_data_023001AQS_stacked_raw_diff.hdf5"
    infile="train_test_data_023001AQS_Qv_stacked_raw_diff.hdf5"
    # infile="train_test_data_163001AQ_raw.hdf5"
    # region="163001AQ1H"
    # region="021501AQ1H"
    # region="023001AQ3H"
    # region="9999LEAU"
    # region = "023001AQ3HT"
    # region = "023001AQTS"
    region = "023001AQS_Qv"
    # region = "021501AQT"
    # region = "023001AQTT3_t19"
    # region = "163001AQT"
    split_file_train_test(infile, region, location)

    # n_splits=10
    # infile="train_test_data_021501AQ_std.hdf5"
    # split_training_data(infile, n_splits, location)

if __name__ == "__main__":
    main()