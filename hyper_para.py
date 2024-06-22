import argparse

def parse_para():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='cora', help='dataset name')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--interval', type=int, default=5, help='interval of update C')
    parser.add_argument("--lam", type=float, default=1, help='trade off parameter')
    parser.add_argument('--mu', type=float, default=0.5, help='trade off l1 and l2')
    parser.add_argument("--tau", type=float, default=0.5, help='temperature')
    parser.add_argument("--prop_eta", type=int, default=1000, help='samples selected for probability transformation')
    parser.add_argument("--theta", type=float, default=1, help='scaled parameter')
    parser.add_argument('--scheme', type=str, default='mask', help='scheme of implementation')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--lr_C', type=float, default=1e-1, help='learning rate of C')
    parser.add_argument('--lr1', type=float, default=1e-3, help='learning rate of Grape')
    parser.add_argument('--lr2', type=float, default=1e-2, help='learning rate of linear evaluator.')
    parser.add_argument('--num_epochs_C', type=int, default=10, help='update C epochs')
    parser.add_argument('--wd1', type=float, default=0, help='weight decay of Grape.')
    parser.add_argument('--wd2', type=float, default=1e-4, help='Weight decay of linear evaluator.')
    parser.add_argument('--epoch1', type=int, default=100, help='Training epochs.')
    parser.add_argument('--epoch2', type=int, default=1000, help='Evaluation epochs.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--der', type=float, default=0.4, help='Drop edge ratio.')
    parser.add_argument('--dfr', type=float, default=0.1, help='Drop feature ratio.')
    parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')
    parser.add_argument("--out_dim", type=int, default=256, help='Output layer dim.')
    parser.add_argument("--proj_dim", type=int, default=0, help='Project dim.')
    parser.add_argument('--mean', type=bool, default=False, help='Calculate mean for neighbor pos')
    parser.add_argument("--k_hop", type=int, default=2, help='positive hop')
    
    return parser.parse_args()

def set_para(para):
    if para.dataset == 'cora':
        para.k_hop = 5
    elif para.dataset == 'pubmed':
        para.epoch2 = 100 
        para.k_hop = 4 
        para.interval = 10
        para.tau = 1.0 
    elif para.dataset == 'wiki':
        para.epoch2 = 2000 
    elif para.dataset == 'ama_photo':
        para.mu = 0.1
        para.seed = 5000
    elif para.dataset == 'citeseer':
        para.epoch1 = 20
        para.k_hop = 3 
        para.interval = 2 
        para.tau = 1.0 
        para.seed = 5000
    elif para.dataset == 'ama_computer':
        para.interval = 10 
        para.seed = 5000
    elif para.dataset == 'co_cs':
        para.mu = 0.1  
        para.epoch2 = 200
        para.k_hop = 3 
        para.der = 0.3 
    elif para.dataset == 'co_physics':
        para.lam = 100 
    else:
        raise NotImplementedError("Unexpected Dataset")
    return para
