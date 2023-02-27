import os
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from cogdl.oag import oagbert
from cogdl.data import Graph
from cogdl.datasets import NodeDataset, build_dataset
from cogdl import experiment
from cogdl.models import build_model
from cogdl.options import get_default_args
from cogdl.trainer.trainer_utils import load_model
from sklearn.metrics import average_precision_score
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="graphsage")
parser.add_argument("--num-features", type=int, default=768)
parser.add_argument("--num-classes", type=int, default=2)
parser.add_argument("--hidden-size", type=int, default=256)
args = parser.parse_args()


def gen_node_emb_all():
    cs_paper_list = utils.load_json("data/", "dpv12_last.json")
    n_nodes = len(cs_paper_list)
    features = np.zeros((n_nodes, 768))

    tokenizer, model = oagbert("oagbert-v2")
    model.cuda()

    for item in tqdm(cs_paper_list):
        index = item['index']
        title=item['title']
        # abstract=item['abstract']
        abstract = ""
        authors=[]
        concepts=[]
        author=item['authors']
        fos_name=item['fos']
        for i in author:
            authors.append(i.get('name'))
        if fos_name==None:
            concepts=[]
        else:
            for i in fos_name:
                concepts.append(i.get('name'))

        input_ids, input_masks, token_type_ids, masked_lm_labels, position_ids, position_ids_second, masked_positions, num_spans = model.build_inputs(title=title, abstract=abstract, authors=authors, concepts=concepts)
        # 使用模型进行前向传播
        sequence_output, pooled_output = model.bert.forward(
            input_ids=torch.LongTensor(input_ids).unsqueeze(0).cuda(),
            token_type_ids=torch.LongTensor(token_type_ids).unsqueeze(0).cuda(),
            attention_mask=torch.LongTensor(input_masks).unsqueeze(0).cuda(),
            output_all_encoded_layers=False,
            checkpoint_activations=False,
            position_ids=torch.LongTensor(position_ids).unsqueeze(0).cuda(),
            position_ids_second=torch.LongTensor(position_ids).unsqueeze(0).cuda()
        )
        pooled_output=torch.squeeze(pooled_output)
        features[index] = pooled_output.cpu().detach().numpy()
    
    np.save("data/graph/node_emb.npy", features.astype(np.float16))


def split_graph_data():
    train=[]
    valid=[]
    test=[]

    venue_year_to_candidates_train = utils.load_json("data/", "venue_year_to_candidates_train.json")
    venue_year_to_candidates_valid = utils.load_json("data/", "venue_year_to_candidates_valid.json")
    venue_year_to_candidates_test = utils.load_json("data/", "venue_year_to_candidates_test.json")

    y_train=[]
    y_test=[]
    y_valid=[]

    mid_to_label = {}

    mids_train = set()
    for cur_key in venue_year_to_candidates_train:
        for cur_mid in venue_year_to_candidates_train[cur_key]:
            mids_train.add(cur_mid)
            mid_to_label[cur_mid] = venue_year_to_candidates_train[cur_key][cur_mid]
    mids_valid = set()
    for cur_key in venue_year_to_candidates_valid:
        for cur_mid in venue_year_to_candidates_valid[cur_key]:
            mids_valid.add(cur_mid)
            mid_to_label[cur_mid] = venue_year_to_candidates_valid[cur_key][cur_mid]
    mids_test = set()
    for cur_key in venue_year_to_candidates_test:
        for cur_mid in venue_year_to_candidates_test[cur_key]:
            mids_test.add(cur_mid)
            mid_to_label[cur_mid] = venue_year_to_candidates_test[cur_key][cur_mid]

    data = utils.load_json("data/", "dpv12_last.json")

    for index,da in enumerate(tqdm(data)):
        cur_mid = str(da["id"])
        if cur_mid in mids_train:
            train.append(da)
            y_train.append(mid_to_label[cur_mid])
        elif cur_mid in mids_valid:
            valid.append(da)
            y_valid.append(mid_to_label[cur_mid])
        elif cur_mid in mids_test:
            test.append(da)
            y_test.append(mid_to_label[cur_mid])

    print("train长度",len(train),"test",len(test),"valid",len(valid))
    os.makedirs("data/graph/", exist_ok=True)
    with open('data/graph/dpv12_train.json','w')as fs2:
        json.dump(train,fs2,ensure_ascii=False)
    with open('data/graph/dpv12_valid.json','w')as fs2:
        json.dump(valid,fs2,ensure_ascii=False)
    with open('data/graph/dpv12_test.json','w')as fs2:
        json.dump(test,fs2,ensure_ascii=False)
    
    with open('data/graph/y.txt','w') as fs2:
        for i in y_train:
            fs2.write(str(i)+'\n')
    
    with open("data/graph/ty.txt",'w') as fs2:
        for i in y_test:
            fs2.write(str(i)+'\n')

    with open("data/graph/vy.txt",'w') as fs2:
        for i in y_valid:
            fs2.write(str(i)+'\n')


class MyNodeDataset2(NodeDataset):
    def __init__(self, path="data.pt"):
        super(MyNodeDataset2, self).__init__(path)

    def remove_self_loops(self,indices, values=None):
        row, col = indices
        mask = indices[0] != indices[1]
        row = row[mask]
        col = col[mask]
        if values is not None:
            values = values[mask]
        return (row, col), values

    def coalesce(self,row, col, value=None):
        device = row.device
        if torch.is_tensor(row):
            row = row.cpu().numpy()
        if torch.is_tensor(col):
            col = col.cpu().numpy()
        indices = np.lexsort((col, row))
        row = torch.from_numpy(row[indices]).long().to(device)
        col = torch.from_numpy(col[indices]).long().to(device)

        num = col.shape[0] + 1
        idx = torch.full((num,), -1, dtype=torch.long).to(device)
        max_num = max(row.max(), col.max()) + 100
        idx[1:] = (row + 1) * max_num + col
        mask = idx[1:] > idx[:-1]

        if mask.all():
            return row, col, value
        row = row[mask]
        if value is not None:
            _value = torch.zeros(row.shape[0], dtype=torch.float).to(device)
            value = _value.scatter_add_(dim=0, src=value, index=col)
        col = col[mask]
        return row, col, value

    def edge_index_from_dict(self,graph, num_nodes=None):
        row, col = [], []
        for item in graph:
            items=item.strip().split()
            key=[int(items[1])]
            value=[int(items[0])]
            row.append(np.array(key))
            col.append(np.array(value))
        _row = np.concatenate(row)
        _col = np.concatenate(col)
        edge_index = np.stack([_row, _col], axis=0)

        row_dom = edge_index[:, _row > _col]
        col_dom = edge_index[:, _col > _row][[1, 0]]
        edge_index = np.concatenate([row_dom, col_dom], axis=1)
        _row, _col = edge_index

        edge_index = np.stack([_row, _col], axis=0)

        order = np.lexsort((_col, _row))
        edge_index = edge_index[:, order]

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        # There may be duplicated edges and self loops in the datasets.
        edge_index, _ = self.remove_self_loops(edge_index)
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1], edge_index[0]])

        row, col, _ = self.coalesce(row, col)
        edge_index = torch.stack([row, col])
        return edge_index 

    def index_to_mask(self,index, size):
        mask = torch.full((size,), False, dtype=torch.bool)
        mask[index] = True
        return mask
    
    def process(self):
        x = np.load("data/graph/node_emb.npy")
        x = torch.from_numpy(x).float()
        y = np.zeros((x.shape[0]))
        y = torch.from_numpy(y).long()

        tot_papers = utils.load_json("data/", "tot_paper_dict_with_mag_id_copy.json")
        tot_mids = set()
        for pid in tqdm(tot_papers):
            paper = tot_papers[pid]
            if "mag_id" in paper:
                tot_mids.add(str(paper["mag_id"]))
        print("tot_mids", len(tot_mids))

        roles = ["train", "valid", "test"]
        ind_roles = []
        y_hit = 0
        for role in roles:
            data = utils.load_json("data/graph", "dpv12_{}.json".format(role))
            cur_ind_list = []
            for item in tqdm(data):
                cur_mid = str(item["id"])
                if cur_mid in tot_mids:
                    y[item["index"]] = 1
                    y_hit += 1
                cur_idx = item["index"]
                cur_ind_list.append(cur_idx)
            cur_ind_list = self.index_to_mask(cur_ind_list, x.shape[0])
            ind_roles.append(cur_ind_list)
        print("y_hit", y_hit)
        
        with open("data/graph/graph.txt",'r') as f2:
            graph = self.edge_index_from_dict(f2)

        data = Graph(x=x, edge_index=graph, y=y, train_mask=ind_roles[0], val_mask=ind_roles[1], test_mask=ind_roles[2])
        return data


def gnn_predict(args=args):
    dataset = MyNodeDataset2()
    args = get_default_args(model=args.model, dataset=["data.pt"])

    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    
    model = build_model(args)
    
    x = torch.load("{}.pt".format(args.model[0]))
    print(x)
    if args.model[0] == "sgc":
        x["nn.W.weight"] = x["model.nn.W.weight"]
        x["nn.W.bias"] = x["model.nn.W.bias"]
        x.pop("model.nn.W.weight")
        x.pop("model.nn.W.bias")
        print("after", x)
    elif args.model[0] == "sign":
        x["mlp.nn.mlp.0.weight"] = x["model.mlp.nn.mlp.0.weight"]
        x.pop("model.mlp.nn.mlp.0.weight")
        x["mlp.nn.mlp.0.bias"] = x["model.mlp.nn.mlp.0.bias"]
        x.pop("model.mlp.nn.mlp.0.bias")
        x["mlp.nn.mlp.1.weight"] = x["model.mlp.nn.mlp.1.weight"]
        x.pop("model.mlp.nn.mlp.1.weight")
        x["mlp.nn.mlp.1.bias"] = x["model.mlp.nn.mlp.1.bias"]
        x.pop("model.mlp.nn.mlp.1.bias")
    elif args.model[0] == "graphsage":
        x["convs.0.fc.weight"] = x["model.convs.0.fc.weight"]
        x.pop("model.convs.0.fc.weight")
        x["convs.0.fc.bias"] = x["model.convs.0.fc.bias"]
        x.pop("model.convs.0.fc.bias")
        x["convs.1.fc.weight"] = x["model.convs.1.fc.weight"]
        x.pop("model.convs.1.fc.weight")
        x["convs.1.fc.bias"] = x["model.convs.1.fc.bias"]
        x.pop("model.convs.1.fc.bias")
    model.load_state_dict(x)
    
    model.eval()
    output = model.predict(dataset.data).detach().numpy()
    print(output.shape)

    data = utils.load_json("data/", "dpv12_last.json")

    mid_to_score = {}
    for item in tqdm(data):
        cur_mid = str(item["id"])
        cur_idx = item["index"]
        mid_to_score[cur_mid] = output[cur_idx][1]
    print("mid_to_score", len(mid_to_score))

    venue_year_to_candidates_test = utils.load_json("data/", "venue_year_to_candidates_test.json")
    metrics = []
    for cur_key in venue_year_to_candidates_test:
        cur_mid_to_label = venue_year_to_candidates_test[cur_key]
        cur_labels = []
        cur_scores = []
        for cur_mid in cur_mid_to_label:
            cur_labels.append(cur_mid_to_label[cur_mid])
            cur_scores.append(mid_to_score[cur_mid])
        cur_metric = average_precision_score(cur_labels, cur_scores)
        metrics.append(cur_metric)
    
    print("model", args.model[0])
    print("mean", np.mean(metrics))


if __name__ == "__main__":
    # gen_node_emb_all()
    # split_graph_data()
    dataset = MyNodeDataset2()
    experiment(dataset=dataset,model="graphsage",device="cuda:1", seed=[1, 2, 3], epochs=1000, checkpoint_path="graphsage.pt")
    gnn_predict()
