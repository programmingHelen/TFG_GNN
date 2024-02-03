import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
import math
from models.BasicModel import BasicModule

class PyGCN(BasicModule):
    def __init__(self, nfeat, nhid, nclass):
        super(PyGCN, self).__init__()
        self.model_name = 'PyGCN'
        
        # First graph convolutional layer
        self.gc1 = GraphConvolution(nfeat, nhid)
        
        # Second graph convolutional layer
        self.gc2 = GraphConvolution(nhid, nclass)
        
        # Dropout layer for regularization
        self.droput = nn.Dropout()

    def forward(self, x, adj):
        # Apply ReLU activation to the output of the first layer
        x = F.relu(self.gc1(x, adj))
        
        # Apply dropout for regularization
        x = self.droput(x)
        
        # Pass through the second graph convolutional layer
        x = self.gc2(x, adj)
        
        # Apply log softmax activation to the final output
        return F.log_softmax(x, dim=1)

class GraphConvolution(nn.Module):
    def __init__(self, in_feature, out_feature, bias=True):
        super(GraphConvolution, self).__init__()
        
        # Define the input and output feature dimensions
        self.in_features = in_feature
        self.out_features = out_feature
        
        # Weight parameter for the linear transformation
        self.weight = Parameter(torch.FloatTensor(in_feature, out_feature))
        
        # Bias parameter if bias is set to True
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_feature))
        else:
            self.register_parameter('bias', None)
        
        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the weight matrix with Xavier initialization
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
        # Initialize the bias (if exists) with Xavier initialization
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # Linear transformation: input * weight
        support = torch.mm(input, self.weight)
        
        # Graph convolution: adj * support
        output = torch.spmm(adj, support)
        
        # Add bias if exists
        if self.bias is not None:
            return output + self.bias
        else:
            return output
