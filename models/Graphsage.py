import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models.BasicModel import BasicModule  # Importing a custom base class (assuming it handles basic functionalities)

class PyGraphsage(BasicModule):
    def __init__(self, nfeat, nhid, nclass):
        super(PyGraphsage, self).__init__()
        self.model_name = 'PyGraphsage'
        
        # Dropout layer for regularization
        self.droput = nn.Dropout()
        
        # Two GraphSAGE layers
        self.sage1 = Graphsage(nfeat, nhid)
        self.sage2 = Graphsage(nhid, nhid)
        
        # Linear layer for final classification
        self.att = nn.Linear(nhid, nclass)

    def forward(self, input, adj):
        # Forward pass through GraphSAGE layers
        hid1 = self.sage1(input, adj)
        hid1 = self.droput(hid1)
        hid2 = self.sage2(hid1, adj)
        out = self.att(hid2)
        
        # Apply log_softmax activation to the final output
        return F.log_softmax(out, dim=1)

class Graphsage(nn.Module):
    def __init__(self, infeat, outfeat):
        super(Graphsage, self).__init__()
        self.infeat = infeat
        self.model_name = 'Graphsage'
        
        # Weight parameter for linear transformation
        self.W = nn.Parameter(torch.zeros(size=(2 * infeat, outfeat)))
        
        # Bias term for linear transformation
        self.bias = nn.Parameter(torch.zeros(outfeat))
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight parameters with a uniform distribution
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        
        # Initialize bias terms with a uniform distribution
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # GraphSAGE layer forward pass
        
        # Aggregate neighborhood features by multiplying with the adjacency matrix
        h1 = torch.mm(adj, input)
        
        # Normalize aggregated features by dividing by the sum of degrees
        degree = adj.sum(axis=1).repeat(self.infeat, 1).T
        h1 = h1 / degree
        
        # Concatenate input features with normalized neighborhood features
        h1 = torch.cat([input, h1], dim=1)
        
        # Linear transformation using weight parameters and bias term
        h1 = torch.mm(h1, self.W)
        
        return h1
