import torch
from torch.nn import functional as F

class MLP(torch.nn.Module):
    def __init__(self, in_features, nb_classes, nb_hidden_layer, 
        hidden_size, act=torch.nn.ReLU):
        super(MLP, self).__init__()
        self.act = act()
        
        self.fc1 = torch.nn.Linear(in_features, hidden_size)
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)])
        self.out = torch.nn.Linear(hidden_size, nb_classes)
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        for l in self.fcs:
            x = F.relu(l(x))
        x = self.out(x)
        return x

class MLP_06(torch.nn.Module):
    """
    140 inputs and 70 outputs neural networks
    """
    def __init__(self, n_inputs, n_outputs):
        super(MLP_06, self).__init__()
        self.fc1 = torch.nn.Linear(n_inputs,256)
        self.fc2 = torch.nn.Linear(256,512)
        self.fc3 = torch.nn.Linear(512,512)
        self.fc4 = torch.nn.Linear(512,512)
        self.fc5 = torch.nn.Linear(512,512)
        self.fc6 = torch.nn.Linear(512,256)
        self.out = torch.nn.Linear(256,n_outputs)

    def forward(self,x):
        """
        Forward model
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        # predict = torch.tanh(self.out(x))
        predict = self.out(x)
        
        return predict



class MLP_12(torch.nn.Module):
    """
    Deep neural network
    """
    def __init__(self, n_inputs, n_outputs):
        super(MLP_12, self).__init__()
        self.fc1 = torch.nn.Linear(n_inputs,256)
        self.fc2 = torch.nn.Linear(256,512)
        self.fc3 = torch.nn.Linear(512,512)
        self.fc4 = torch.nn.Linear(512,512)
        self.fc5 = torch.nn.Linear(512,512)
        self.fc6 = torch.nn.Linear(512,512)
        self.fc7 = torch.nn.Linear(512,512)
        self.fc8 = torch.nn.Linear(512,512)
        self.fc9 = torch.nn.Linear(512,512)
        self.fc10 = torch.nn.Linear(512,512)
        self.fc11 = torch.nn.Linear(512,512)
        self.fc12 = torch.nn.Linear(512,256)
        self.out = torch.nn.Linear(256,n_outputs)

    def forward(self,x):
        """
        Forward model
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))
        # predict = torch.tanh(self.out(x))
        predict = self.out(x)
        
        return predict
