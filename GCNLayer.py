import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch_geometric.nn import GCNConv

class GCNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, in_features))
        self.conv = GCNConv(in_features, out_features)
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x, edge_index, active=True):
        x0 = torch.mm(x, self.weight)
        x = self.conv(x0, edge_index)

        if active:
            x = F.leaky_relu(x, negative_slope=0.2)
        return x
