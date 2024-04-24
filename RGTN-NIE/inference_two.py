import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
import dgl
import os
import sys
import pickle as pk
import tqdm

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)

from utils.EarlyStopping import EarlyStopping_simple
from utils.utils import set_random_seed, load_data, get_rank_metrics, rank_evaluate, convert_to_gpu, get_centrality
from utils.metric import overlap
from two_branch.model import rgtn


def main(args):

    set_random_seed(0)

    ndcg_scores = []
    spearmans = []
    rmses = []
    overlaps = []


    g, edge_types, _, rel_num, struct_feat, content_feat, labels, train_idx, val_idx, test_idx = \
        load_data(args.data_path, args.dataset, args.cross_id)
    torch.cuda.set_device(args.gpu)
    struct_feat = struct_feat.cuda()
    content_feat = content_feat.cuda()
    labels = labels.cuda()

    # g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    num_struct_feat = struct_feat.shape[1]
    num_content_feat = content_feat.shape[1]
    n_edges = g.number_of_edges()

    print("""----Data statistics------'
        #Edges %d
        #Train samples %d
        #Val samples %d
        #Test samples %d""" %
            (n_edges,
            len(train_idx),
            len(val_idx),
            len(test_idx)))

    # add self loop
    g = dgl.add_self_loop(g)
    new_edge_types = torch.tensor([rel_num for _ in range(g.number_of_nodes())])
    edge_types = torch.cat([edge_types, new_edge_types], 0)
    rel_num += 1
    n_edges = g.number_of_edges()

    # create model
    loss_fcn = torch.nn.MSELoss()
    model = rgtn(args, g, rel_num, num_struct_feat, num_content_feat, get_centrality(g), loss_fcn)
    
    if cuda:
        model.cuda()
        edge_types = edge_types.cuda()
    
    model.load_state_dict(torch.load(args.model_path))
    
    # evaluation
    model.eval()
    with torch.no_grad():
        logits = model(struct_feat, content_feat, edge_types)
    
    output_path = os.path.join(args.output_dir, f'logits_tot_geni_{args.cross_id}.pk')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, 'wb') as f:
        pk.dump(logits.cpu().numpy(), f)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT-Two')
    parser.add_argument("--dataset", type=str, default='FB15k_rel_two',
                        help="The input dataset. Can be FB15k_two")
    parser.add_argument("--data_path", type=str, default='datasets/fb15k_rel.pk',
                        help="path of dataset")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--cross-num", type=int, default=5,
                        help="number of cross validation")
    parser.add_argument("--epochs", type=int, default=10000,
                        help="number of training epochs")
    parser.add_argument('--min-epoch', type=int, default=-1,
                        help='the least epoch for training, avoiding stopping at the start time')
    parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=4,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--feat-drop", type=float, default=0.)
    parser.add_argument("--in-drop", type=float, default=.3,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.3,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--patience', type=int, default=1000,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--scale', action="store_true", default=False,
                        help="utilize centrality to scale scores")
    parser.add_argument('--pred-dim', type=int, default=10,
                        help="the size of predicate embedding vector")
    parser.add_argument('--save-path', type=str, default='gat-two_checkpoint.pt',
                        help='the path to save the best model')

    parser.add_argument('--loss-lambda', type=float, default=0.5,
                        help='the weight to add unsupervised loss')
    parser.add_argument('--norm', action="store_true", default=False)
    parser.add_argument('--edge-mode', type=str, default='MUL')
    parser.add_argument('--spm', action="store_true", default=False)
    parser.add_argument('--loss-alpha', type=float, default=0.3)
    parser.add_argument('--list-num', type=int, default=100)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--cross_id', type=int)
    parser.add_argument('--model_path', type=str)
    

    args = parser.parse_args()
    print(args)

    main(args)