import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv


def sim(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x, edge_weight=None):

        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](graph, x, edge_weight=edge_weight))
        x = self.convs[-1](graph, x)

        return x

class GRAPE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, num_nodes, num_proj_hidden, tau: float=0.5):
        super(GRAPE, self).__init__()
        self.encoder = GCN(in_dim, hid_dim, out_dim, n_layers)
        self.tau: float = tau
        self.fc1 = torch.nn.Linear(out_dim, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, out_dim)
        self.num_nodes = num_nodes
        self.num_proj_hidden = num_proj_hidden

        self.neighboraggr = GraphConv(num_nodes, num_nodes, norm='both', weight=False, bias=False)
    
    def posaug(self, graph, x, edge_weight):
        return self.neighboraggr(graph, x, edge_weight=edge_weight)

    def forward(self, graph1, feat1, graph2, feat2, graph, feat):
        z1 = self.encoder(graph1, feat1)
        z2 = self.encoder(graph2, feat2)
        z = self.encoder(graph, feat)
        return z1, z2, z

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def semi_loss(self, z1, adj1, z2, adj2, S, para):
        def f(x): return torch.exp(x / self.tau)
        refl_sim = f(sim(z1, z1))
        between_sim = f(sim(z1, z2))
        if para.scheme == 'weight':
            if para.mean:
                positive = between_sim.diag() + (refl_sim * S).sum(1) / (adj1.sum(1)+0.01)
            else:
                positive = between_sim.diag() + (refl_sim * S).sum(1)
            negative = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        elif para.scheme == 'mask':
            positive = between_sim.diag()
            negative = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag() - (
                (between_sim*S).sum(1) + (refl_sim*S).sum(1))

        loss = -torch.log(positive / (positive + negative))

        return loss

    def loss(self, z1, graph1, z2, graph2, S, para):
        if self.num_proj_hidden > 0:
            h1 = self.projection(z1)
            h2 = self.projection(z2)
        else:
            h1 = z1
            h2 = z2

        l1 = self.semi_loss(h1, graph1, h2, graph2, S, para)
        l2 = self.semi_loss(h2, graph2, h1, graph1, S, para)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret

    def get_embedding(self, graph, feat):
        with torch.no_grad():
            out = self.encoder(graph, feat)
            return out.detach()


class One_Layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(One_Layer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret
