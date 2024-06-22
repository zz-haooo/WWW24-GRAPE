import numpy as np
import dgl
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset
from dgl.data import WikiCSDataset

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

from model import One_Layer

def load(name):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'ama_photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'ama_computer':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'co_physics':
        dataset = CoauthorPhysicsDataset()
    elif name == 'co_cs':
        dataset = CoauthorCSDataset()
    elif name == 'wiki':
        dataset = WikiCSDataset()
    else:
        raise NotImplementedError("Unexpected Dataset")

    graph = dataset[0]
    citegraph = ['cora', 'citeseer', 'pubmed']
    cograph = ['ama_photo', 'ama_computer', 'co_physics', 'co_cs', 'wiki']

    if name in citegraph:
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')

        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
        val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze()
        test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()

    if name in cograph:
        train_ratio = 0.1
        val_ratio = 0.1
        test_ratio = 0.8

        N = graph.number_of_nodes()
        train_num = int(N * train_ratio)
        val_num = int(N * (train_ratio + val_ratio))

        idx = np.arange(N)
        np.random.shuffle(idx)

        train_idx = idx[:train_num]
        val_idx = idx[train_num:val_num]
        test_idx = idx[val_num:]

        train_idx = torch.tensor(train_idx)
        val_idx = torch.tensor(val_idx)
        test_idx = torch.tensor(test_idx)

    num_class = dataset.num_classes
    feat = graph.ndata.pop('feat')
    labels = graph.ndata.pop('label')

    return graph, feat, labels, num_class, train_idx, val_idx, test_idx


def random_aug(graph, x, feat_drop_rate, edge_mask_rate):
    new_feat = drop_feature(x, feat_drop_rate) 
    new_graph = mask_edge(graph, edge_mask_rate) 

    return new_graph, new_feat

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def mask_edge(graph, mask_prob):
    N = graph.number_of_nodes() 
    E = graph.number_of_edges() 
    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob) 

    reserve = torch.bernoulli(1 - mask_rates)  
    reserve_idx = reserve.nonzero().squeeze(1)

    new_graph = dgl.graph([]) 
    new_graph.add_nodes(N)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    new_graph.add_edges(src[reserve_idx], dst[reserve_idx])

    return new_graph


def Generate_C(C, para, Phi, in_dim, z):
    optimizer_C = torch.optim.Adam([C], lr=para.lr_C)
    for epoch_C in range(para.num_epochs_C):
        X = z.detach().clone()
        self_exp_loss = torch.norm(X - torch.matmul(torch.relu(C), X), 'fro')
        l1_norms = torch.norm(C, p=1, dim=0)
        l2_norms = torch.norm(C, p=2, dim=0)
        regularization_loss = torch.sum(
            para.mu * l1_norms + ((1-para.mu)/2) * l2_norms)
        loss_C = (1/in_dim)*self_exp_loss + \
            para.lam*regularization_loss
        optimizer_C.zero_grad()
        loss_C.backward()

        C.grad[Phi == 0] = 0.0
        optimizer_C.step()

    return C

def Generate_Phi(C, para, graph_cuda_k_hop, z):
    X = z.detach().clone() 
    P = X - torch.matmul(torch.relu(C), X)
    P = F.normalize(P, p=2, dim=1)
    Q = F.normalize(X.t(), p=2, dim=0)
    Phi = torch.zeros_like(C)
    Phi[torch.abs( torch.matmul(P, Q) ) >= para.lam*para.mu] = 1
    Phi[graph_cuda_k_hop == 0] = 0
    return Phi

def Generate_S(C, args):
    S = torch.relu(C.detach())
    S = (S + S.t())/2
    if args.scheme == 'weight':
        return S
    elif args.scheme == 'mask':
        nonzero_indices = torch.nonzero(S)
        random_indices = random.sample(
            range(nonzero_indices.shape[0]), args.prop_eta)
        selected_values = [S[idx[0], idx[1]]
                        for idx in nonzero_indices[random_indices]]
        S = args.theta * S / max(selected_values)
        S[S > 1] = 1
        return S
    else:
        raise NotImplementedError("Unexpected Scheme")


class MLP_parameterization(nn.Module):
    def __init__(self, nlayers, feature_dim, hid_dim, activation='relu'):
        super(MLP_parameterization, self).__init__()

        self.mlp1 = self._create_mlp(nlayers, feature_dim, hid_dim, activation)
        self.mlp2 = self._create_mlp(nlayers, feature_dim, hid_dim, activation)
        self.similarity_metric = nn.CosineSimilarity(dim=1)

    def _create_mlp(self, nlayers, feature_dim, hid_dim, activation):
        layers = [nn.Linear(feature_dim, hid_dim)]
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hid_dim, hid_dim))
        layers.append(nn.Linear(hid_dim, hid_dim))

        activations = [nn.ReLU() if activation == 'relu' else nn.Tanh() for _ in range(nlayers - 1)]
        
        mlp = nn.Sequential()
        for i in range(nlayers - 1):
            mlp.add_module(f'linear_{i}', layers[i])
            mlp.add_module(f'activation_{i}', activations[i])
        mlp.add_module(f'linear_{nlayers-1}', layers[-1])
        
        return mlp

    def forward(self, x1, x2):
        h1 = self.mlp1(x1)
        h2 = self.mlp2(x2)
        C = self.similarity_metric(h1, h2)
        return C


class Attentive(nn.Module):
    def __init__(self, feature_dim):
        super(Attentive, self).__init__()
        self.att_layer = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, x):
        return F.relu(self.att_layer(x))

class Attentive_parameterization(nn.Module):
    def __init__(self, nlayers, feature_dim, activation='relu'):
        super(Attentive_parameterization, self).__init__()

        self.att1 = self._create_att(nlayers, feature_dim, activation)
        self.att2 = self._create_att(nlayers, feature_dim, activation)
        self.similarity_metric = nn.CosineSimilarity(dim=1)

    def _create_att(self, nlayers, feature_dim, activation):
        layers = [Attentive(feature_dim)]
        for _ in range(nlayers - 1):
            layers.append(Attentive(feature_dim))

        activations = [nn.ReLU() if activation == 'relu' else nn.Tanh() for _ in range(nlayers)]
        
        att = nn.Sequential()
        for i in range(nlayers):
            att.add_module(f'attentive_{i}', layers[i])
            att.add_module(f'activation_{i}', activations[i])
        
        return att

    def forward(self, x1, x2):
        h1 = self.att1(x1)
        h2 = self.att2(x2)
        C = self.similarity_metric(h1, h2)
        return C


def evaluate_classification(emb, num_class, true_y, args, train_idx, val_idx, test_idx):
    train_embs = emb[train_idx]
    val_embs = emb[val_idx]
    test_embs = emb[test_idx]


    train_labels = true_y[train_idx]
    val_labels = true_y[val_idx]
    test_labels = true_y[test_idx]

    logreg = One_Layer(input_dim=train_embs.shape[1], output_dim=num_class)
    opt = torch.optim.Adam(logreg.parameters(),
                           lr=args.lr2, weight_decay=args.wd2)

    logreg = logreg.to(args.device)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epoch2+1):
        logreg.train()
        opt.zero_grad()
        logits = logreg(train_embs)
        preds = torch.argmax(logits, dim=1)
        train_acc = torch.sum(preds == train_labels).float() / train_labels.shape[0]
        loss = loss_fn(logits, train_labels)
        loss.backward()
        opt.step()

        logreg.eval()
        with torch.no_grad():
            val_logits = logreg(val_embs)
            test_logits = logreg(test_embs)
            val_preds = torch.argmax(val_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)
            val_acc = torch.sum(val_preds == val_labels).float() / val_labels.shape[0]
            test_acc = torch.sum(test_preds == test_labels).float() / test_labels.shape[0]
            if epoch % 10 == 0:
                print('Epoch: {}, train_acc: {:.4f}, val_acc: {:4f}, test_acc: {:4f}'.format(epoch, train_acc, val_acc, test_acc))

    return train_acc.cpu().numpy(), val_acc.cpu().numpy(), test_acc.cpu().numpy()

def evaluate_clustering(emb, num_class, true_y, repetition_cluster):
    true_y = true_y.detach().cpu().numpy()
    embeddings = F.normalize(emb, dim=-1, p=2).detach().cpu().numpy()

    estimator = KMeans(n_clusters=num_class)

    NMI_list = []
    ARI_list = []

    for _ in range(repetition_cluster):
        estimator.fit(embeddings)
        y_pred = estimator.predict(embeddings)

        nmi_score = normalized_mutual_info_score(true_y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(true_y, y_pred)
        NMI_list.append(nmi_score)
        ARI_list.append(ari_score)

    return np.mean(NMI_list), np.std(NMI_list), np.mean(ARI_list), np.std(ARI_list)
