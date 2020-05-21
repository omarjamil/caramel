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
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        for l in self.fcs:
            x = self.act(l(x))
        x = self.out(x)
        return x

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

class MLPSkip(torch.nn.Module):
    def __init__(self, in_features, nb_classes, nb_hidden_layer, 
        hidden_size, act=torch.nn.ReLU):
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
        
    def forward(self, x):
        inputs = x
        x = self.act(self.fc1(x))
        for l in self.fcs:
            x = self.act(l(x))
        x = self.act(self.skip(x) + inputs)
        x = self.sigmoid(self.out(x))
        # x = self.out(x)
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
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.out = torch.nn.Linear(hidden_size, nb_classes)
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        for l in self.fcs:
            x = self.act(self.bn(l(x)))
        x = self.out(x)
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