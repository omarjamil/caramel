import data_io
import data_io_stacked
import torch
import matplotlib.pyplot as plt

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def train_dataloader(args):
    train_dataset_file = "{0}/train_data_{1}.hdf5".format(args.locations["train_test_datadir"],args.region)
    train_loader = torch.utils.data.DataLoader(
             data_io.ConcatDataset("train",args.nlevs,train_dataset_file, args.locations['normaliser_loc'], xvars=args.xvars,
             yvars=args.yvars, yvars2=args.yvars2, data_frac=args.data_fraction, add_adv=False),
             batch_size=args.batch_size, shuffle=True)
    return train_loader

def test_dataloader(args):
    test_dataset_file = "{0}/test_data_{1}.hdf5".format(args.locations["train_test_datadir"],args.region)
    validation_loader = torch.utils.data.DataLoader(
             data_io.ConcatDataset("test",args.nlevs, test_dataset_file, args.locations['normaliser_loc'], xvars=args.xvars,
             yvars=args.yvars, yvars2=args.yvars2, data_frac=args.data_fraction, add_adv=False),
             batch_size=args.batch_size, shuffle=False)
    return validation_loader

def train_dataloader_stacked(args):
    train_dataset_file = "{0}/train_data_{1}.hdf5".format(args.locations["train_test_datadir"],args.region)
    train_loader = torch.utils.data.DataLoader(
             data_io_stacked.ConcatDataset("train",args.nlevs,train_dataset_file, args.locations['normaliser_loc'], xvars=args.xvars,
             yvars=args.yvars, yvars2=args.yvars2, samples_frac=args.samples_fraction, data_frac=args.data_fraction, no_norm=args.no_norm,
             lev_norm=args.lev_norm),
             batch_size=args.batch_size, shuffle=False)
    return train_loader

def test_dataloader_stacked(args):
    test_dataset_file = "{0}/test_data_{1}.hdf5".format(args.locations["train_test_datadir"],args.region)
    validation_loader = torch.utils.data.DataLoader(
             data_io_stacked.ConcatDataset("test",args.nlevs, test_dataset_file, args.locations['normaliser_loc'], xvars=args.xvars,
             yvars=args.yvars, yvars2=args.yvars2, samples_frac=args.samples_fraction, data_frac=args.data_fraction, no_norm=args.no_norm, 
             lev_norm=args.lev_norm),
             batch_size=args.batch_size, shuffle=False)
    return validation_loader

def train_dataloader_CNN(args):
    train_dataset_file = "{0}/cnn_train_data_{1}.hdf5".format(args.locations["train_test_datadir"],args.region)
    train_loader = torch.utils.data.DataLoader(
             data_io.ConcatDatasetCNN2D("train",args.nlevs,train_dataset_file, args.locations['normaliser_loc'], xvars=args.xvars,
             xvars2=args.xvars2, yvars=args.yvars, yvars2=args.yvars2, data_frac=args.data_fraction),
             batch_size=args.batch_size, shuffle=True)
    return train_loader

def test_dataloader_CNN(args):
    test_dataset_file = "{0}/cnn_test_data_{1}.hdf5".format(args.locations["train_test_datadir"],args.region)
    validation_loader = torch.utils.data.DataLoader(
             data_io.ConcatDatasetCNN2D("test",args.nlevs, test_dataset_file, args.locations['normaliser_loc'], xvars=args.xvars,
             xvars2=args.xvars2, yvars=args.yvars, yvars2=args.yvars2, data_frac=args.data_fraction),
             batch_size=args.batch_size, shuffle=False)
    return validation_loader

def loader_loop_CNN():
    args = dotdict()
    args.xvars = ['qtot', 'qadv', 'theta', 'theta_adv']
    args.xvars2 = ['sw_toa', 'shf', 'lhf']
    args.yvars = ['qtot_next', 'theta_next']
    args.yvars2 = ['qphys', 'theta_phys']
    # args.normaliser = "023001AQT_normalise"
    args.normaliser = "023001AQT_standardise_mx"
    args.locations = {"train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain",
                    "normaliser_loc":"/project/spice/radiation/ML/CRM/data/models/normaliser/{0}".format(args.normaliser)}
    args.region = "023001AQTT3_t19"
    args.data_fraction = 0.001
    args.batch_size = 10
    args.nlevs = 5
    train_ldr = train_dataloader_CNN(args)
    test_ldr = test_dataloader_CNN(args)
    cmap='plasma'
    for batch_idx, batch in enumerate(train_ldr):
        # x,y,y2 = batch
        x,x2,y,y2 = batch
        print("X", x.shape)
        fig, axs = plt.subplots(4,1,figsize=(14, 10), sharex=True)
        c = axs[0].pcolor(x[0,0,:,:].T, cmap=cmap, label='q')
        axs[0].legend()
        # c = axs[0].imshow(x[0,0,:,:], cmap='gray')
        fig.colorbar(c,ax=axs[0])
        c = axs[1].pcolor(x[0,1,:,:].T, cmap=cmap, label='qadv')
        axs[1].legend()
        fig.colorbar(c,ax=axs[1])
        c = axs[2].pcolor(x[0,2,:,:].T, cmap=cmap, label='theta')
        axs[2].legend()
        fig.colorbar(c,ax=axs[2])
        c = axs[3].pcolor(x[0,3,:,:].T, cmap=cmap, label='theta_adv')
        axs[3].legend()
        fig.colorbar(c,ax=axs[3])
        plt.show()
        print("x0:", x[:,0,:,:].min(), x[:,0,:,:].max())
        print("x1:", x[:,1,:,:].min(), x[:,1,:,:].max())
        print("x2:", x[:,2,:,:].min(), x[:,2,:,:].max())
        print("x3:", x[:,3,:,:].min(), x[:,3,:,:].max())
        print("X2", x2.shape)
        print("Y", y.shape)
        print("Y2", y2.shape)

def loader_loop_stacked():
    args = dotdict()
    # args.xvars = ['qtot', 'qadv', 'theta', 'theta_adv']
    # args.xvars=["qtot", "qadv", "theta", "theta_adv", "sw_toa", "shf", "lhf", "p", "rho", "xwind", "ywind", "zwind"]
    args.xvars=["qtot", "theta", "p", "rho", "xwind", "ywind", "zwind", "sw_toa", "shf", "lhf"]
    args.xvars2 = ['sw_toa', 'shf', 'lhf']
    args.yvars = ['qtot_next', 'theta_next']
    args.yvars2 = ['qphys', 'theta_phys']
    # args.normaliser = "023001AQT_normalise"
    # args.normaliser = "023001AQT_standardise_mx"
    # args.normaliser = "023001AQT_normalise"
    args.normaliser = "023001AQT_normalise_60_glb"
    args.locations = {"train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain",
                    "normaliser_loc":"/project/spice/radiation/ML/CRM/data/models/normaliser/{0}".format(args.normaliser)}
    args.region = "023001AQTS"
    args.data_fraction = 0.1
    args.samples_fraction = 0.01
    args.batch_size = 10
    args.nlevs = 60
    args.no_norm = False
    args.lev_norm = False
    train_ldr = train_dataloader_stacked(args)
    test_ldr = test_dataloader_stacked(args)
    cmap='plasma'
    for batch_idx, batch in enumerate(train_ldr):
        # x,y,y2 = batch
        x,y,y2 = batch
        print("X", x.shape)
        fig, axs = plt.subplots(4,1,figsize=(14, 10), sharex=True)
        c = axs[0].pcolor(x[:,:60].T, cmap=cmap, label='q')
        fig.colorbar(c,ax=axs[0])
        axs[0].legend()
        c = axs[1].pcolor(x[:,60:120].T, cmap=cmap, label='t')
        fig.colorbar(c,ax=axs[1])
        axs[1].legend()
        c = axs[2].pcolor(x[:,120:180].T, cmap=cmap, label='p')
        fig.colorbar(c,ax=axs[2])
        axs[2].legend()
        c = axs[3].pcolor(x[:,180:240].T, cmap=cmap, label='rho')
        fig.colorbar(c,ax=axs[3])
        axs[3].legend()
        plt.show()
        # print("x0:", x[:,0,:,:].min(), x[:,0,:,:].max())
        # print("x1:", x[:,1,:,:].min(), x[:,1,:,:].max())
        # print("x2:", x[:,2,:,:].min(), x[:,2,:,:].max())
        # print("x3:", x[:,3,:,:].min(), x[:,3,:,:].max())
        # print("X2", x2.shape)
        # print("Y", y.shape)
        # print("Y2", y2.shape)

def test_validation_io():
    args = dotdict()
    args.xvars = ['qtot', 'qadv', 'theta', 'theta_adv']
    args.xvars2 = ['sw_toa', 'shf', 'lhf']
    args.yvars = ['qtot_next', 'theta_next']
    args.yvars2 = ['qphys', 'theta_phys']
    args.normaliser = "023001AQ_normalise"
    args.locations = {"train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain",
                    "normaliser_loc":"/project/spice/radiation/ML/CRM/data/models/normaliser/{0}".format(args.normaliser)}
    args.region = "0N100W"
    args.data_fraction = 0.01
    args.batch_size = 10
    args.nlevs = 45
    datasetfile="/project/spice/radiation/ML/CRM/data/models/datain/validation_0N100W/validation_data_0N100W_000.hdf5"
    nn_data = data_io.Data_IO_validation_CNN(args.region, args.nlevs, datasetfile, args.locations['normaliser_loc'], 
                        xvars=args.xvars,
                        xvars2=args.xvars2,
                        yvars=args.yvars,
                        yvars2=args.yvars2)
    x,x2,y,y2,xmean,xstd,xmean2,xstd2,ymean,ystd,ymean2,ystd2 = nn_data.get_data()
    # print(x.shape, x2.shape, xmean.shape, xmean2.shape)
    qtot_inv = nn_data._inverse_transform(x[:,0,:], xmean[0,:], xstd[0,:])
    qadv_inv = nn_data._inverse_transform(x[:,1,:], xmean[1,:], xstd[1,:])
    theta_inv = nn_data._inverse_transform(x[:,2,:], xmean[2,:], xstd[2,:])
    theta_adv_inv = nn_data._inverse_transform(x[:,3,:], xmean[3,:], xstd[3,:])
    print(qtot_inv[0,:])
    print(qadv_inv[0,:])
    print(theta_inv[0,:])
    print(theta_adv_inv[0,:])


if __name__ == "__main__":
    
    # ones = torch.ones((10,20))
    # twos = torch.ones((10,20))
    # twos[:] = 2
    # threes = torch.ones((10,20))
    # threes[:] = 3
    # stacked = torch.stack([ones,twos,threes],dim=1)
    # print(stacked)
    # print(stacked.shape)
    loader_loop_stacked()
    # test_validation_io()