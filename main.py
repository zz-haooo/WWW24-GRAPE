import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import dgl
import random
import argparse
import csv
import os
import warnings
warnings.filterwarnings('ignore')

from model import GRAPE
from utils import *
from hyper_para import parse_para, set_para



if __name__ == '__main__':

    para = parse_para() 
    para = set_para(para) # custom hyperparameter on dataset

    if para.gpu != -1 and torch.cuda.is_available():
        para.gpu = 1
        para.device = 'cuda:{}'.format(para.gpu)
    else:
        para.device = 'cpu'

    np.random.seed(para.seed)
    torch.manual_seed(para.seed)
    random.seed(para.seed)
    torch.cuda.manual_seed(para.seed)
    print(f"hyperparameter setting: {para}")

    graph, feat, labels, num_class, train_idx, val_idx, test_idx = load(para.dataset) # load data
    print("feature_shape: {}, label_shape: {}, train_shape: {}, validate_shape: {}, test_shape: {}".format(feat.shape, labels.shape, train_idx.shape, val_idx.shape, test_idx.shape))
 
    in_dim = feat.shape[1]
    N = graph.number_of_nodes()

    true_Y = labels.to(para.device).view(-1, 1)
    true_Y = (true_Y == true_Y.t()).float()

    graph_cuda = graph.to(para.device)
    graph_cuda = graph_cuda.remove_self_loop().add_self_loop()
    feat_cuda = feat.to(para.device)

    graph_cuda_k_hop = dgl.khop_graph(
        graph_cuda, para.k_hop).adjacency_matrix().to_dense()
    Phi = dgl.khop_graph(graph_cuda, 1).adjacency_matrix().to_dense()

    model = GRAPE(in_dim, para.hid_dim, para.out_dim, para.n_layers, N, num_proj_hidden=para.proj_dim, tau=para.tau)
    model = model.to(para.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=para.lr1, weight_decay=para.wd1)

    C = nn.Parameter(torch.randn(N, N), requires_grad=True)
    C = C.to(para.device)
    C = C * graph_cuda_k_hop
    C = C.clone().detach().requires_grad_(True)


    for epoch in range(0, para.epoch1+1):
        model.train()
        optimizer.zero_grad()

        graph1, feat1 = random_aug(graph, feat, para.dfr, para.der)
        graph2, feat2 = random_aug(graph, feat, para.dfr, para.der)

        graph1 = graph1.remove_self_loop().add_self_loop()
        graph2 = graph2.remove_self_loop().add_self_loop()
        graph1, graph2, feat1, feat2 = graph1.to(
            para.device), graph2.to(para.device), feat1.to(para.device), feat2.to(para.device)
  
        z1, z2, z = model(graph1, feat1, graph2, feat2, graph_cuda, feat_cuda)

        adj1 = torch.zeros(N, N, dtype=torch.int).to(para.device)
        adj1[graph1.remove_self_loop().edges()] = 1
        adj2 = torch.zeros(N, N, dtype=torch.int).to(para.device)
        adj2[graph2.remove_self_loop().edges()] = 1

        if epoch % para.interval == 0:
            C = Generate_C(C, para, Phi, in_dim, z)
            Phi = Generate_Phi(C, para, graph_cuda_k_hop, z)
            C = Generate_C(C, para, Phi, in_dim, z)
            S = Generate_S(C, para)

        loss = model.loss(z1, adj1, z2, adj2, S, para)

        loss.backward()
        optimizer.step()

        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))

    # Task: Node classification evaluation
    embeds = model.get_embedding(graph_cuda, feat_cuda)
    labels = labels.to(para.device)

    train_acc, val_acc, test_acc = evaluate_classification(embeds, num_class, labels, para,
                            train_idx, val_idx, test_idx)
    print('Final result: train_acc: {:.4f}, val_acc: {:4f}, test_acc: {:4f}'.format(
        train_acc, val_acc, test_acc))
    print(f"train_acc_type: {type(train_acc)}")

    # save results
    if not os.path.exists(os.path.join(os.getcwd(), "results")):
        os.makedirs("results")

    results = {'train_acc': [train_acc], 'val_acc': [val_acc], 'test_acc': [test_acc]}

    filename = f'results/{para.dataset}_result.csv'

    df = pd.DataFrame(results)
    df.to_csv(filename,  sep='\t', index=False)

    print(f"Results are saved to {filename}")