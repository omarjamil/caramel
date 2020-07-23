import argparse
import torch
import caramel_stacked as caramel

parser = argparse.ArgumentParser(description='Train Q')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--with-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--chkpt-interval', type=int, default=10, metavar='N',
                    help='how many epochs before saving a checkpoint')
parser.add_argument('--isambard', action='store_true', default=False,
                    help='Run on Isambard GPU')
parser.add_argument('--warm-start', type=str, default=None,
                    help='Continue training from this model')
parser.add_argument('--identifier', type=str, 
                    help='Added to model name as a unique identifier;  also needed for warm start from a previous model')                    
parser.add_argument('--data-region', type=str, help='data region')
parser.add_argument('--loss', type=str, help='loss function to use', default='mae')
parser.add_argument('--nb-hidden-layers', type=int, default=6, metavar='N',
                    help='Number of hidden layers (default: 6)')
parser.add_argument('--data-fraction', type=float, default=1.,
                    help='fraction of data points to use for training and testing (default: 1)')  
parser.add_argument('--samples-fraction', type=float, default=1.,
                    help='fraction of samples to use for training and testing (default: 1)')      
parser.add_argument('--normaliser', type=str, 
                    help='Normalisation to use: standardise or normalise')
parser.add_argument('--nlevs', type=int, default=45, metavar='N',
                    help='Number of vertical levels to user')


args_parser = parser.parse_args()

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def set_args():
    if args_parser is not None:
        args = args_parser
    else:
        args = dotdict()
        args.seed = 1
        args.log_interval = 100    
        args.batch_size = 100
        args.epochs = 1
        args.with_cuda = False
        args.chkpt_interval = 10
        args.isambard = False
        args.warm_start = None
        args.identifier = '021501AQ1H'
        args.data_region = '021501AQ1H'
        args.normaliser = '021501AQ1H_standardise_mx'
        args.loss = 'mae'
        args.n_fiters = 20
        args.n_nodes = 10
        args.nlevs = 30
        args.data_fraction = 0.01

    torch.manual_seed(args.seed)
    args.cuda = args.with_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'pin_memory': False} if args.cuda else {}

    # Define the Model
    # n_inputs,n_outputs=140,70
    # args.xvars = ['qtot', 'qadv', 'theta', 'theta_adv', 'sw_toa', 'shf', 'lhf']
    # args.xvars = ['qadv', 'theta_adv', 'sw_toa', 'shf', 'lhf']
    args.xvars = ['qtot', 'theta', 'p', 'rho', 'xwind', 'ywind', 'zwind', 'shf', 'lhf','sw_toa']
    # args.xvars = ['qtot', 'theta', 'sw_toa', 'shf', 'lhf']
    # args.yvars = ['qtot_next', 'theta_next']
    args.yvars = ['qtot_next']
    # args.yvars2 = ['qphys', 'theta_phys']
    args.yvars2 = ['qphys']
    # args.yvars2 = ['theta_phys']
    args.train_on_y2 = False
    args.no_norm = False
    args.lev_norm = True
    args.region=args.data_region
    args.in_features = (args.nlevs*(len(args.xvars)-3)+3)
    if not args.train_on_y2:
        args.nb_classes = (args.nlevs*(len(args.yvars)))
        # args.nb_classes = 1 #(args.nlevs*(len(args.yvars)))
        print("Inputs: {0} Ouputs: {1}".format(args.xvars, args.yvars))
    else:
        args.nb_classes = (args.nlevs*(len(args.yvars2)))
        # args.nb_classes = 1 #(args.nlevs*(len(args.yvars2)))
        print("Inputs: {0} Ouputs: {1}".format(args.xvars, args.yvars2))

    # args.hidden_size = 512 
    args.hidden_size = int(1.0 * args.in_features + args.nb_classes)
    args.model_name = "qnext_{0}_lyr_{1}_in_{2}_out_{3}_hdn_{4}_epch_{5}_btch_{6}_{7}_{8}_stkd_sigm.tar".format(str(args.nb_hidden_layers).zfill(3),
                                                                                        str(args.in_features).zfill(3),
                                                                                        str(args.nb_classes).zfill(3),
                                                                                        str(args.hidden_size).zfill(4),
                                                                                        str(args.epochs).zfill(3),
                                                                                        str(args.batch_size).zfill(5),
                                                                                        args.identifier, 
                                                                                        args.loss,
                                                                                        args.normaliser
                                                                                        )
    print(args.model_name)
    # Get the data
    if args.isambard:
        args.locations={ "train_test_datadir":"/home/mo-ojamil/ML/CRM/data",
                "chkpnt_loc":"/home/mo-ojamil/ML/CRM/data/models/chkpts",
                "hist_loc":"/home/mo-ojamil/ML/CRM/data/models",
                "model_loc":"/home/mo-ojamil/ML/CRM/data/models/torch",
                "normaliser_loc":"/home/mo-ojamil/ML/CRM/data/normaliser/{0}".format(args.normaliser)}
    else:
        args.locations={ "train_test_datadir":"/project/spice/radiation/ML/CRM/data/models/datain",
                "chkpnt_loc":"/project/spice/radiation/ML/CRM/data/models/chkpts/torch",
                "hist_loc":"/project/spice/radiation/ML/CRM/data/models/history",
                "model_loc":"/project/spice/radiation/ML/CRM/data/models/torch",
                "normaliser_loc":"/project/spice/radiation/ML/CRM/data/models/normaliser/{0}".format(args.normaliser)}
    return args

if __name__ == "__main__":
    args = set_args()
    model, loss_function, optimizer, scheduler = caramel.set_model(args)
    training_loss, validation_loss = caramel.train_loop(model, loss_function, optimizer, scheduler, args)