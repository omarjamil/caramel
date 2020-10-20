import torch
from torch.nn import functional as F

class MLP(torch.nn.Module):
    def __init__(self, in_features, nb_classes, nb_hidden_layer, 
        hidden_size, act=torch.nn.LeakyReLU):
        super(MLP, self).__init__()
        self.act = act()
        self.n_hidden_layers = nb_hidden_layer
        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)])
        self.fcs.extend([torch.nn.Linear(hidden_size, hidden_size) for i in range(1,self.n_hidden_layers)] )
        self.out = torch.nn.Linear(hidden_size, nb_classes)
        self.sig = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        for l in self.fcs:
            x = self.act(l(x))
        x = self.out(x)
        # x = self.sig(self.out(x))
        return x

class MLP_RELU(torch.nn.Module):
    def __init__(self, in_features, nb_classes, nb_hidden_layer, 
        hidden_size, act=torch.nn.ReLU):
        super(MLP_RELU, self).__init__()
        self.act = act()
        self.n_hidden_layers = nb_hidden_layer
        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)])
        self.fcs.extend([torch.nn.Linear(hidden_size, hidden_size) for i in range(1,self.n_hidden_layers)] )
        self.out = torch.nn.Linear(hidden_size, nb_classes)
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        for l in self.fcs:
            x = self.act(l(x))
        x = self.act(self.out(x))
        # x = self.sig(self.out(x))
        return x

class MLP_tanh(torch.nn.Module):
    def __init__(self, in_features, nb_classes, nb_hidden_layer, 
        hidden_size, act=torch.nn.LeakyReLU, scale=1.):
        super(MLP_tanh, self).__init__()
        self.act = act()
        self.n_hidden_layers = nb_hidden_layer
        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)])
        self.fcs.extend([torch.nn.Linear(hidden_size, hidden_size) for i in range(1,self.n_hidden_layers)] )
        self.out = torch.nn.Linear(hidden_size, nb_classes)
        self.tanh = torch.nn.Tanh()
        self.scale = scale
    def forward(self, x):
        x = self.act(self.fc1(x))
        for l in self.fcs:
            x = self.act(l(x))
        # x = self.out(x)
        # x = self.tanh(self.out(x))
        x = self.scale*self.tanh(self.out(x))
        # x = self.out(x)
        return x

class MLP_sig(torch.nn.Module):
    def __init__(self, in_features, nb_classes, nb_hidden_layer, 
        hidden_size, act=torch.nn.LeakyReLU):
        super(MLP_sig, self).__init__()
        self.act = act()
        self.n_hidden_layers = nb_hidden_layer
        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)])
        self.fcs.extend([torch.nn.Linear(hidden_size, hidden_size) for i in range(1,self.n_hidden_layers)] )
        self.out = torch.nn.Linear(hidden_size, nb_classes)
        self.sig = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        for l in self.fcs:
            x = self.act(l(x))
        # x = self.out(x)
        x = self.sig(self.out(x))
        # x = self.out(x)
        return x

class MLP_BN_tanh(torch.nn.Module):
    def __init__(self, in_features, nb_classes, nb_hidden_layer, 
        hidden_size, act=torch.nn.LeakyReLU):
        super(MLP_BN_tanh, self).__init__()
        self.act = act()
        self.n_hidden_layers = nb_hidden_layer
        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)])
        self.fcs.extend([torch.nn.Linear(hidden_size, hidden_size) for i in range(1,self.n_hidden_layers)] )
        self.out = torch.nn.Linear(hidden_size, nb_classes)
        self.tanh = torch.nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(in_features)
        
    def forward(self, x):
        x = self.act(self.fc1(self.bn1(x)))
        for l in self.fcs:
            x = self.act(l(x))
        # x = self.out(x)
        x = self.tanh(self.out(x))
        # x = self.out(x)
        return x

class MLP_multiout_tanh(torch.nn.Module):
    def __init__(self, in_features, nb_classes, nb_hidden_layer, 
        hidden_size, act=torch.nn.LeakyReLU):
        super(MLP_multiout_tanh, self).__init__()
        self.act = act()
        self.n_hidden_layers = nb_hidden_layer
        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.fcs1 = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)])
        self.fcs1.extend([torch.nn.Linear(hidden_size, hidden_size) for i in range(1,self.n_hidden_layers-1)] )
        self.fcs2 = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)])
        self.fcs2.extend([torch.nn.Linear(hidden_size, hidden_size) for i in range(1,self.n_hidden_layers)] )
        self.fcs3 = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)])
        self.fcs3.extend([torch.nn.Linear(hidden_size, hidden_size) for i in range(1,self.n_hidden_layers+1)] )
        # self.fcs4 = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)])
        # self.fcs4.extend([torch.nn.Linear(hidden_size, hidden_size) for i in range(1,self.n_hidden_layers+2)] )
        self.out = torch.nn.Linear(hidden_size, nb_classes)
        self.tanh = torch.nn.Tanh()
        
    def forward(self, x):
        x1 = self.act(self.fc1(x))
        x2 = self.act(self.fc1(x))
        x3 = self.act(self.fc1(x))
        # x4 = self.act(self.fc1(x))
        # for l1,l2,l3,l4 in zip(self.fcs1,self.fcs2,self.fcs3,self.fcs4):
        for l1,l2,l3 in zip(self.fcs1,self.fcs2,self.fcs3):
            x1 = self.act(l1(x1))
            x2 = self.act(l2(x2))
            x3 = self.act(l3(x3))
            # x4 = self.act(l4(x4))
        # x = self.out(x)
        # xout_1 = self.tanh(self.out(x1))
        # xout_2 = self.tanh(self.out(x2))
        # xout_3 = self.tanh(self.out(x3))
        xout_1 = self.out(x1)
        xout_2 = self.out(x2)
        xout_3 = self.out(x3)
        # xout_4 = self.tanh(self.out(x4))
        # return xout_1,xout_2,xout_3,xout_4
        return xout_1,xout_2,xout_3

class MLPDrop(torch.nn.Module):
    def __init__(self, in_features, nb_classes, nb_hidden_layer, 
        hidden_size, act=torch.nn.ReLU):
        super(MLPDrop, self).__init__()
        self.act = act()
        self.n_hidden_layers = nb_hidden_layer
        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)])
        self.fcs.extend([torch.nn.Linear(hidden_size, hidden_size) for i in range(1,self.n_hidden_layers)] )
        self.out = torch.nn.Linear(hidden_size, nb_classes)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.act(self.fc1(x))
        for l in self.fcs:
            x = self.act(l(x))
            x = self.dropout(x)
        x = self.out(x)
        return x

class MLPSkip_(torch.nn.Module):
    def __init__(self, in_features, nb_classes, nb_hidden_layer, 
        hidden_size, act=torch.nn.LeakyReLU):
        super(MLPSkip_, self).__init__()
        print("Model with skip connections")
        self.act = act()
        self.sigmoid = torch.nn.Sigmoid()
        self.n_hidden_layers = nb_hidden_layer
        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)])
        self.fcs.extend([torch.nn.Linear(hidden_size, hidden_size) for i in range(1,self.n_hidden_layers)] )
        self.skip = torch.nn.Linear(hidden_size,in_features)
        self.out = torch.nn.Linear(in_features, nb_classes)
        self.sig = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        # self.bn_in = torch.nn.BatchNorm1d(in_features)

        
    def forward(self, x):
        inputs = x
        # x = self.act(self.bn(self.fc1(x)))
        x = self.act(self.fc1(x))
        for l in self.fcs:
            x = self.act(l(x))
        x = self.act(self.skip(x) + inputs)
        # x = self.act(self.skip(x))
        # x = self.act(self.skip(x))
        # x = self.sigmoid(self.out(x))
        # x = self.out(x)
        x = self.sig(self.out(x))
        # x = self.out(x +  inputs)
        return x

class ResidualBlockMLP(torch.nn.Module):
    """
    Simple residual block for hidden layers
    """
    def __init__(self, in_size, out_size, act=torch.nn.LeakyReLU):
        super(ResidualBlockMLP, self).__init__()
        self.act = act()
        self.fc1 = torch.nn.Linear(in_size, out_size)
        self.fc2 = torch.nn.Linear(in_size, out_size)

    def forward(self,x):
        residual = x 
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x += residual
        out = self.act(x)
        return out

class ResMLP(torch.nn.Module):
    """
    MLP ResNet
    """
    def __init__(self,in_features, nb_classes, nb_hidden_layer, hidden_size):
        super(ResMLP, self).__init__()
        print("MLP ResNet")
        self.act = torch.nn.LeakyReLU()
        self.resblock = ResidualBlockMLP
        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.res_layer = self.make_layers(self.resblock, hidden_size, nb_hidden_layer)
        self.out = torch.nn.Linear(hidden_size,nb_classes)
        self.tanh = torch.nn.Tanh()

    def make_layers(self, block, size, nblocks):
        layers = []
        layers.append(block(size,size))
        for _ in range(1,nblocks):
            layers.append(block(size,size))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.res_layer(x)
        out = self.tanh(self.out(x))
        return out

class MLPSkip(torch.nn.Module):
    def __init__(self, in_features, nb_classes, nb_hidden_layer, 
        hidden_size, act=torch.nn.LeakyReLU):
        super(MLPSkip, self).__init__()
        print("Model with skip connections")
        self.act = act()
        self.sigmoid = torch.nn.Sigmoid()
        self.n_hidden_layers = nb_hidden_layer
        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)])
        self.fcs.extend([torch.nn.Linear(hidden_size, hidden_size) for i in range(1,self.n_hidden_layers)] )
        self.skip = torch.nn.Linear(hidden_size,in_features)
        self.out = torch.nn.Linear(in_features, nb_classes)
        # self.sig = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        # self.bn = torch.nn.BatchNorm1d(hidden_size)
        # self.bn_in = torch.nn.BatchNorm1d(in_features)

        
    def forward(self, x):
        inputs = x
        # x = self.act(self.bn(self.fc1(x)))
        x = self.act(self.fc1(x))
        for l in self.fcs:
            x = self.act(l(x))
        # x = self.act(self.skip(x) + self.bn_in(inputs))
        x = self.act(self.skip(x))
        # x = self.act(self.skip(x))
        # x = self.sigmoid(self.out(x))
        # x = self.out(x)
        # x = self.sig(self.out(x + inputs))
        x = self.tanh(self.out(x + inputs))
        # x = self.out(x +  inputs)
        return x

class MLPSubSkip(torch.nn.Module):
    def __init__(self, in_features, nb_classes, nb_hidden_layer, 
        hidden_size, subskip_indx, act=torch.nn.LeakyReLU):
        super(MLPSubSkip, self).__init__()
        print("Model with sub skip connections")
        self.act = act()
        self.sigmoid = torch.nn.Sigmoid()
        self.n_hidden_layers = nb_hidden_layer
        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)])
        self.fcs.extend([torch.nn.Linear(hidden_size, hidden_size) for i in range(1,self.n_hidden_layers)] )
        self.skip = torch.nn.Linear(hidden_size,in_features)
        self.out = torch.nn.Linear(in_features, nb_classes)
        # self.sig = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.subskip_indx = subskip_indx
        
    def forward(self, x):
        inputs = x[...,self.subskip_indx]
        x = self.act(self.fc1(x))
        for l in self.fcs:
            x = self.act(l(x))
        x = self.act(self.skip(x))
        # x[...,self.subskip_indx] = x[...,self.subskip_indx] + inputs
        # print("Model out:", self.tanh(self.out(x))[...,0:5])
        # x = self.sig(self.out(x)) + inputs
        x = self.tanh(self.out(x)) + inputs
        return x


class MLP_BN(torch.nn.Module):
    """
    Neural network with batch normalisation
    """
    def __init__(self, in_features, nb_classes, nb_hidden_layer, 
        hidden_size, act=torch.nn.ReLU):
        super(MLP_BN, self).__init__()
        self.act = act()
        self.n_hidden_layers = nb_hidden_layer
        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)])
        self.fcs.extend([torch.nn.Linear(hidden_size, hidden_size) for i in range(1,self.n_hidden_layers)] )
        # self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.bn1 = torch.nn.BatchNorm1d(in_features)
        self.out = torch.nn.Linear(hidden_size, nb_classes)
        self.sig = torch.nn.Sigmoid()
    def forward(self, x):
        # x = self.act(self.bn(self.fc1(x)))
        x = self.act(self.fc1(self.bn1(x)))
        # x = self.act(self.fc1(x))
        for l in self.fcs:
            # x = self.act(self.bn(l(x)))
            x = self.act(l(x))
        x = self.sig(self.out(x))
        return x

class ConvNN5(torch.nn.Module):
    def __init__(self, in_channels, n_levs, nb_classes,
        act=torch.nn.LeakyReLU):
        super(ConvNN5, self).__init__()
        self.act = act()
        self.conv1 = torch.nn.Conv1d(in_channels, 8, 4, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(8, 8, 4, stride=1, padding=1)
        self.conv3 = torch.nn.Conv1d(8, 16, 4, stride=1, padding=1)
        self.conv4 = torch.nn.Conv1d(16, 32, 4, stride=1, padding=1)
        self.conv5 = torch.nn.Conv1d(32, 4, 4, stride=1, padding=1)
        # If three convolutions with above values, final length = nlevs - 3
        self.final_length = n_levs - 5
        self.fc1 = torch.nn.Linear(4*self.final_length, 2*self.final_length)
        self.fc2 = torch.nn.Linear(2*self.final_length, int(1.5*self.final_length))
        self.out = torch.nn.Linear(int(1.5*self.final_length),nb_classes)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = x.view(-1,4*self.final_length)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.out(x)
        return x

class ConvNN3b(torch.nn.Module):
    def __init__(self, in_channels, n_levs, nb_classes,
        act=torch.nn.LeakyReLU):
        super(ConvNN3b, self).__init__()
        self.act = act()
        self.conv1 = torch.nn.Conv1d(in_channels, 40, 4, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(40, 40, 4, stride=1, padding=1)
        self.conv3 = torch.nn.Conv1d(40, 16, 4, stride=1, padding=1)
        # If three convolutions with above values, final length = nlevs - 3
        self.final_length = n_levs - 3
        self.fc1 = torch.nn.Linear(int(16*self.final_length), int(10*self.final_length))
        self.fc2 = torch.nn.Linear(int(10*self.final_length), int(10*self.final_length))
        self.out = torch.nn.Linear(int(10*self.final_length),nb_classes)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = x.view(-1,16*self.final_length)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.out(x)
        return x



class ConvNN3Skip(torch.nn.Module):
    def __init__(self, in_channels, n_levs, nb_classes, n_filters, n_nodes,
        act=torch.nn.LeakyReLU):
        super(ConvNN3Skip, self).__init__()
        self.act = act()
        self.n_levs = n_levs
        self.in_channels = in_channels
        self.nb_classes = nb_classes
        self.n_filters = n_filters
        self.n_nodes = n_nodes
        self.conv1 = torch.nn.Conv1d(self.in_channels, n_filters, 4, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(n_filters, n_filters, 4, stride=1, padding=1)
        self.conv3 = torch.nn.Conv1d(n_filters, 16, 4, stride=1, padding=1)
        # If three convolutions with above values, final length = nlevs - 3
        self.final_length = n_levs - 3
        self.fc1 = torch.nn.Linear(int(16*self.final_length), int(self.n_nodes*self.final_length))
        self.fc2 = torch.nn.Linear(int(self.n_nodes*self.final_length), int(self.n_nodes*self.final_length))
        self.skiplayer = torch.nn.Linear(int(self.n_nodes*self.final_length),int(self.in_channels*self.n_levs))
        self.out = torch.nn.Linear(int(self.in_channels*n_levs),self.nb_classes)

    def forward(self, x):
        input = x
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = x.view(-1,16*self.final_length)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.skiplayer(x))
        # x = self.act(x + input.view(-1,self.in_channels*self.n_levs))
        x = self.out(x + input.view(-1,self.in_channels*self.n_levs))
        return x

class ConvNN2Pool(torch.nn.Module):
    def __init__(self, in_channels, n_levs, nb_classes, n_filters, n_nodes,
        act=torch.nn.LeakyReLU):
        super(ConvNN2Pool, self).__init__()
        self.act = act()
        self.n_levs = n_levs
        self.in_channels = in_channels
        self.nb_classes = nb_classes
        self.n_filters = n_filters
        self.n_nodes = n_nodes
        self.conv1 = torch.nn.Conv1d(self.in_channels, self.n_filters, 4, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(self.n_filters, 16, 4, stride=1, padding=1)
        self.pool = torch.nn.MaxPool1d(2,2)
        # If three convolutions with above values, final length = nlevs - 3
        self.final_length = n_levs//2 - 1
        self.fc1 = torch.nn.Linear(int(16*self.final_length), int(self.n_nodes*self.final_length))
        self.fc2 = torch.nn.Linear(int(self.n_nodes*self.final_length), int(self.n_nodes*self.final_length))
        # self.skiplayer = torch.nn.Linear(int(self.n_nodes*self.final_length),int(self.in_channels*self.n_levs))
        # self.out = torch.nn.Linear(int(self.in_channels*n_levs),self.nb_classes)
        self.out = torch.nn.Linear(int(self.n_nodes*self.final_length),self.nb_classes)

    def forward(self, x):
        input = x
        x = self.pool(self.act(self.conv1(x)))
        x = self.act(self.conv2(x))
        x = x.view(-1,16*self.final_length)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        # x = self.act(self.skiplayer(x))
        # x = self.act(x + input.view(-1,self.in_channels*self.n_levs))
        x = self.out(x)
        return x

# class ConvNN3Skip(torch.nn.Module):
#     def __init__(self, in_channels, n_levs, nb_classes,
#         act=torch.nn.LeakyReLU):
#         super(ConvNN3Skip, self).__init__()
#         self.act = act()
#         self.n_levs = n_levs
#         self.in_channels = in_channels
#         self.nb_classes = nb_classes
#         self.conv1 = torch.nn.Conv1d(self.in_channels, 16, 4, stride=1, padding=1)
#         self.conv2 = torch.nn.Conv1d(16, 16, 4, stride=1, padding=1)
#         self.conv3 = torch.nn.Conv1d(16, 16, 4, stride=1, padding=1)
#         # If three convolutions with above values, final length = nlevs - 3
#         self.final_length = n_levs - 3
#         self.fc1 = torch.nn.Linear(int(16*self.final_length), int(5*self.final_length))
#         self.fc2 = torch.nn.Linear(int(5*self.final_length), int(5*self.final_length))
#         self.skiplayer = torch.nn.Linear(int(5*self.final_length),int(self.in_channels*self.n_levs))
#         self.out = torch.nn.Linear(int(self.in_channels*n_levs),self.nb_classes)

#     def forward(self, x):
#         input = x
#         x = self.act(self.conv1(x))
#         x = self.act(self.conv2(x))
#         x = self.act(self.conv3(x))
#         x = x.view(-1,16*self.final_length)
#         x = self.act(self.fc1(x))
#         x = self.act(self.fc2(x))
#         x = self.act(self.skiplayer(x))
#         x = self.act(x + input.view(-1,self.in_channels*self.n_levs))
#         x = self.out(x)
#         return x

class ConvNN3Skip2(torch.nn.Module):
    def __init__(self, in_channels, n_levs, nb_classes, n_filters, n_nodes,
        act=torch.nn.LeakyReLU):
        super(ConvNN3Skip2, self).__init__()
        self.act = act()
        self.n_levs = n_levs
        self.in_channels = in_channels
        self.nb_classes = nb_classes
        self.n_filters = n_filters
        self.n_nodes = n_nodes
        self.conv1 = torch.nn.Conv1d(self.in_channels, self.n_filters, 4, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(self.n_filters, self.n_filters, 4, stride=1, padding=1)
        self.conv3 = torch.nn.Conv1d(self.n_filters, 16, 4, stride=1, padding=1)
        # If three convolutions with above values, final length = nlevs - 3
        self.final_length = n_levs - 3
        self.fc1 = torch.nn.Linear(int(16*self.final_length), int(self.n_nodes*self.final_length))
        self.fc2 = torch.nn.Linear(int(self.n_nodes*self.final_length), int(self.n_nodes*self.final_length))
        self.skiplayer = torch.nn.Linear(int(self.n_nodes*self.final_length),int(self.in_channels*self.n_levs))
        self.out = torch.nn.Linear(int(self.in_channels*n_levs),self.nb_classes)

    def forward(self, x):
        input = x
        x = self.act(self.conv1(x))
        conv_skip = x
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        print(x.shape)
        print(16*self.final_length, self.final_length)
        x = torch.nn.MaxPool1d(3,1)(conv_skip).view(-1,self.n_filters*self.final_length)
        print(x.shape)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        print(x.shape)
        x = self.act(self.skiplayer(x))
        # print(x.shape)
        # print(x.shape, self.in_channels*self.n_levs, input.shape)
        x = self.act(x + input.view(-1,self.in_channels*self.n_levs))
        x = self.out(x)
        return x

class ConvNN(torch.nn.Module):
    def __init__(self, in_channels, n_levs, nb_classes,
        act=torch.nn.LeakyReLU):
        super(ConvNN, self).__init__()
        self.act = act()
        self.conv1 = torch.nn.Conv1d(in_channels, 8, 4, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(8, 8, 4, stride=1, padding=1)
        self.conv3 = torch.nn.Conv1d(8, 4, 4, stride=1, padding=1)
        # If three convolutions with above values, final length = nlevs - 3
        self.final_length = n_levs - 3
        self.fc1 = torch.nn.Linear(4*self.final_length, 2*self.final_length)
        self.fc2 = torch.nn.Linear(2*self.final_length, int(1.5*self.final_length))
        self.out = torch.nn.Linear(int(1.5*self.final_length),nb_classes)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = x.view(-1,4*self.final_length)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.out(x)
        return x

class Conv2dwSkip(torch.nn.Module):
    def __init__(self, in_channels, n_levs, nb_classes, n_filters, n_nodes,
        act=torch.nn.LeakyReLU):
        """
        2d CNN
        """
        super(ConvNN2dwSkip, self).__init__()
        self.act = act()
        self.n_levs = n_levs
        self.in_channels = in_channels
        self.nb_classes = nb_classes
        self.n_filters = n_filters
        self.n_nodes = n_nodes
        self.conv1 = torch.nn.Conv2d(self.in_channels, n_filters, (2,2), stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(n_filters, n_filters, (2,2), stride=1, padding=1)
        self.conv3 = torch.nn.Conv1d(n_filters, 16, (2,2), stride=1, padding=1)
        # If three convolutions with above values, final length = nlevs - 3
        self.final_length = n_levs - 3
        self.fc1 = torch.nn.Linear(int(16*self.final_length), int(self.n_nodes*self.final_length))
        self.fc2 = torch.nn.Linear(int(self.n_nodes*self.final_length), int(self.n_nodes*self.final_length))
        self.skiplayer = torch.nn.Linear(int(self.n_nodes*self.final_length),int(self.in_channels*self.n_levs))
        self.out = torch.nn.Linear(int(self.in_channels*n_levs),self.nb_classes)

    def forward(self, x):
        input = x
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = x.view(-1,16*self.final_length)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.skiplayer(x))
        # x = self.act(x + input.view(-1,self.in_channels*self.n_levs))
        x = self.out(x + input.view(-1,self.in_channels*self.n_levs))
        return x


class ResidualBlock(torch.nn.Module):

    def __init__(self, channels):
        
        super(ResidualBlock, self).__init__()
        # self.conv_1 = torch.nn.Conv2d(in_channels=channels[0],
        #                               out_channels=channels[1],
        #                               kernel_size=(3, 3),
        #                               stride=(2, 2),
        #                               padding=1)
        self.conv_1 = torch.nn.Conv2d(in_channels=channels[0],
                                      out_channels=channels[1],
                                      kernel_size=(3, 5),
                                      stride=(2, 2),
                                      padding=1)
        self.conv_1_bn = torch.nn.BatchNorm2d(channels[1])
                                    
        self.conv_2 = torch.nn.Conv2d(in_channels=channels[1],
                                      out_channels=channels[2],
                                      kernel_size=(1, 1),
                                      stride=(1, 1),
                                      padding=0)   
        self.conv_2_bn = torch.nn.BatchNorm2d(channels[2])

        # self.conv_shortcut_1 = torch.nn.Conv2d(in_channels=channels[0],
        #                                        out_channels=channels[2],
        #                                        kernel_size=(1, 1),
        #                                        stride=(2, 2),
        #                                        padding=0)  
        self.conv_shortcut_1 = torch.nn.Conv2d(in_channels=channels[0],
                                               out_channels=channels[2],
                                               kernel_size=(1, 3),
                                               stride=(2, 2),
                                               padding=0)   
        self.conv_shortcut_1_bn = torch.nn.BatchNorm2d(channels[2])

    def forward(self, x):
        shortcut = x
        
        out = self.conv_1(x)
        # out = self.conv_1_bn(out)
        # print("out 1", out.shape)
        out = F.relu(out)

        out = self.conv_2(out)
        # print("out 2", out.shape)

        # out = self.conv_2_bn(out)
        
        # match up dimensions using a linear function (no relu)
        shortcut = self.conv_shortcut_1(shortcut)
        # print("shortcut", shortcut.shape)

        # shortcut = self.conv_shortcut_1_bn(shortcut)
        
        out += shortcut
        # print("out 4", out.shape)

        out = F.relu(out)
        # print("out 5", out.shape)

        return out

class ConvNNet(torch.nn.Module):

    def __init__(self, in_channels, num_classes):
        super(ConvNNet, self).__init__()
        
        self.residual_block_1 = ResidualBlock(channels=[in_channels, 8, 16])
        self.residual_block_2 = ResidualBlock(channels=[16, 32, 64])
        self.residual_block_3 = ResidualBlock(channels=[64, 128, 256])
    
        # self.linear_1 = torch.nn.Linear(5*13*64, num_classes)
        self.linear_1 = torch.nn.Linear(3*5*256, num_classes)

        
    def forward(self, x):

        out = self.residual_block_1.forward(x)
        out = self.residual_block_2.forward(out)
        out = self.residual_block_3.forward(out)
        # logits = self.linear_1(out.view(-1, 5*13*64))
        logits = self.linear_1(out.view(-1, 3*5*256))
        # probas = F.softmax(logits, dim=1)
        return logits


#########################


class ConvNet(torch.nn.Module):

    def __init__(self, in_channels, n_levs, num_classes):
        super(ConvNet, self).__init__()
        
        #########################
        ### 1st residual block
        #########################
        # 18x52x4 => 9x26x8
        self.conv_1 = torch.nn.Conv2d(in_channels=4,
                                      out_channels=8,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)
        self.conv_1_bn = torch.nn.BatchNorm2d(8)
                                    
        # 89x26x8 => 9x26x16
        self.conv_2 = torch.nn.Conv2d(in_channels=8,
                                      out_channels=16,
                                      kernel_size=(1, 1),
                                      stride=(1, 1),
                                      padding=0)   
        self.conv_2_bn = torch.nn.BatchNorm2d(16)
        
        # 18x52x4 => 9x26x16
        self.conv_shortcut_1 = torch.nn.Conv2d(in_channels=4,
                                               out_channels=16,
                                               kernel_size=(1, 1),
                                               stride=(2, 2),
                                               padding=0)   
        self.conv_shortcut_1_bn = torch.nn.BatchNorm2d(16)
        
        #########################
        ### 2nd residual block
        #########################
        # 9x26x16 => 5x13x32 
        self.conv_3 = torch.nn.Conv2d(in_channels=16,
                                      out_channels=32,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)
        self.conv_3_bn = torch.nn.BatchNorm2d(32)
                                    
        # 5x13x32 => 5x13x64
        self.conv_4 = torch.nn.Conv2d(in_channels=32,
                                      out_channels=64,
                                      kernel_size=(1, 1),
                                      stride=(1, 1),
                                      padding=0)   
        self.conv_4_bn = torch.nn.BatchNorm2d(64)
        
        # 9x26x32 => 5x13x64 
        self.conv_shortcut_2 = torch.nn.Conv2d(in_channels=16,
                                               out_channels=64,
                                               kernel_size=(1, 1),
                                               stride=(2, 2),
                                               padding=0)   
        self.conv_shortcut_2_bn = torch.nn.BatchNorm2d(64)

        # #########################
        # ### 3rd residual block
        # #########################
        # # 5x13x64 => 3x7x64 
        # self.conv_5 = torch.nn.Conv2d(in_channels=64,
        #                               out_channels=64,
        #                               kernel_size=(3, 3),
        #                               stride=(2, 2),
        #                               padding=1)
        # self.conv_5_bn = torch.nn.BatchNorm2d(64)
                                    
        # # 3x7x64 => 3x7x128
        # self.conv_6 = torch.nn.Conv2d(in_channels=64,
        #                               out_channels=128,
        #                               kernel_size=(1, 1),
        #                               stride=(1, 1),
        #                               padding=0)   
        # self.conv_6_bn = torch.nn.BatchNorm2d(128)
        
        # # 5x13x64 => 3x7x128 
        # self.conv_shortcut_3 = torch.nn.Conv2d(in_channels=64,
        #                                        out_channels=128,
        #                                        kernel_size=(1, 1),
        #                                        stride=(2, 2),
        #                                        padding=0)   
        # self.conv_shortcut_3_bn = torch.nn.BatchNorm2d(128)

        #########################
        ### Fully connected
        #########################        
        self.linear_1 = torch.nn.Linear(5*13*64, num_classes)

        
    def forward(self, x):
        
        #########################
        ### 1st residual block
        #########################
        shortcut = x
        out = self.conv_1(x)
        out = self.conv_1_bn(out)
        out = F.relu(out)

        out = self.conv_2(out) 
        out = self.conv_2_bn(out)
        
        # match up dimensions using a linear function (no relu)
        shortcut = self.conv_shortcut_1(shortcut)
        shortcut = self.conv_shortcut_1_bn(shortcut)
        
        out += shortcut
        out = F.relu(out)
        
        #########################
        ### 2nd residual block
        #########################
        
        shortcut = out
        
        out = self.conv_3(out) 
        out = self.conv_3_bn(out)
        out = F.relu(out)

        out = self.conv_4(out) 
        out = self.conv_4_bn(out)
        
        # match up dimensions using a linear function (no relu)
        shortcut = self.conv_shortcut_2(shortcut)
        shortcut = self.conv_shortcut_2_bn(shortcut)
        
        out += shortcut
        out = F.relu(out)
        
        #########################
        ### Fully connected
        #########################   
        logits = self.linear_1(out.view(-1, 5*13*64))
        # probas = F.softmax(logits, dim=1)
        return logits

#########################
########   AE    #######
class AE(torch.nn.Module):
    def __init__(self, in_features, latent_size, act=torch.nn.ReLU):
        super(AE, self).__init__()
        print("Model AE")
        self.act = act()
        self.tanh = torch.nn.Tanh()
        self.latent_size = latent_size
        # self.fc1 = torch.nn.Linear(in_features, 50)
        # self.fc2 = torch.nn.Linear(50, 30)
        # self.fc3 = torch.nn.Linear(30, 10)
        # self.fc31 = torch.nn.Linear(10, 5)
        # self.fc32 = torch.nn.Linear(5, 10)
        # self.fc4 = torch.nn.Linear(10, 30)
        # self.fc5 = torch.nn.Linear(30, 50)
        # self.fc6 = torch.nn.Linear(50, in_features)

        # self.fc1 = torch.nn.Linear(in_features, 50)
        # self.fc2 = torch.nn.Linear(50, 45)
        # self.fc3 = torch.nn.Linear(45, 40)
        # self.fc31 = torch.nn.Linear(40, 40)
        # self.fc32 = torch.nn.Linear(40, 40)
        # self.fc4 = torch.nn.Linear(40, 45)
        # self.fc5 = torch.nn.Linear(45, 50)
        # self.fc6 = torch.nn.Linear(50, in_features)

        # self.fc1 = torch.nn.Linear(in_features, in_features)
        # self.fc2 = torch.nn.Linear(in_features, in_features)
        # self.fc3 = torch.nn.Linear(in_features, in_features+5)
        # self.fc31 = torch.nn.Linear(in_features+5, in_features+5)
        # self.fc32 = torch.nn.Linear(in_features+5, in_features+5)
        # self.fc4 = torch.nn.Linear(in_features+5, in_features)
        # self.fc5 = torch.nn.Linear(in_features, in_features)
        # self.fc6 = torch.nn.Linear(in_features, in_features)

        self.fc1 = torch.nn.Linear(in_features, in_features)
        self.fc2 = torch.nn.Linear(in_features, in_features)
        self.fc3 = torch.nn.Linear(in_features, in_features-5)
        self.fc31 = torch.nn.Linear(in_features-5, self.latent_size)
        self.fc32 = torch.nn.Linear(self.latent_size, in_features-5)
        self.fc4 = torch.nn.Linear(in_features-5, in_features)
        self.fc5 = torch.nn.Linear(in_features, in_features)
        self.fc6 = torch.nn.Linear(in_features, in_features)

    def encoder(self,x):
        x = self.act(self.fc1(x))
        # x = self.act(self.fc11(x))
        x = self.act(self.fc2(x))
        # x = self.act(self.fc21(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc31(x))
        return x

    def decoder(self,x):
        x = self.act(self.fc32(x))
        x = self.act(self.fc4(x))
        # x = self.act(self.fc41(x))
        x = self.act(self.fc5(x))
        # x = self.act(self.fc51(x))
        # x = self.tanh(self.fc6(x))
        x = self.fc6(x)
        return x

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
       
        return encoded, decoded

#########################
########   VAE    #######

class VAE(torch.nn.Module):
    def __init__(self, in_features, act=torch.nn.ReLU):
        super(VAE, self).__init__()
        print("Model VAE")
        self.act = act()
        self.fc1 = torch.nn.Linear(in_features, 50)
        self.fc2 = torch.nn.Linear(50, 10)
        self.fc31 = torch.nn.Linear(10, 5)
        self.fc32 = torch.nn.Linear(10, 5)
        self.fc4 = torch.nn.Linear(5, 10)
        self.fc5 = torch.nn.Linear(10, 50)
        self.fc6 = torch.nn.Linear(50, in_features)

    def encode(self, x):
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h4 = self.act(self.fc4(z))
        h5 = self.act(self.fc5(h4))
        return self.fc6(h5)
        # return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

##########################
### ResNet 18
##########################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class ResNet(torch.nn.Module):

    def __init__(self, block, layers, num_classes, in_dim):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = torch.nn.AvgPool2d(7, stride=1)
        self.fc = torch.nn.Linear(1024 * block.expansion, num_classes)
        # self.fc = torch.nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        # print("0", x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print("1", x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print("2", x.shape)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        #x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        # print("N", x.shape)
        logits = self.fc(x)
        # probas = F.softmax(logits, dim=1)
        return logits



def resnet18(num_classes, in_dim):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[2, 2, 2, 2],
                   num_classes=num_classes,
                   in_dim=in_dim)
    return model

