from tqdm import tqdm
import networkx as nx
from sklearn.metrics import average_precision_score
import utils
import json
import pickle
import numpy as np


def cal_paper_pagerank():
    g = nx.DiGraph()
    cs_paper_list = utils.load_json("data/", "dpv12_last.json")
    n_nodes = len(cs_paper_list)
    g.add_nodes_from(range(0, n_nodes))
    edges = []
    with open("data/graph/graph.txt") as rf:
        for line in tqdm(rf):
            dst, src = line.strip().split()
            src = int(src)
            dst = int(dst)
            edges.append((src, dst))
    g.add_edges_from(edges)
    print("graph loaded")
    pagerank = nx.pagerank(g, alpha=0.85)
    print("pagerank calculated")
    utils.joblib_dump_obj(pagerank, "data/graph/", "pagerank.pkl")


def eval_pagerank():
    # pagerank = utils.joblib_load_obj("data/graph/", "pagerank.pkl")
    # cs_paper_list = utils.load_json("data/", "dpv12_last.json")

    venue_year_to_candidates_test = utils.load_json("data/", "venue_year_to_candidates_test.json")
    # mid_to_pr = {}

    # for paper in tqdm(cs_paper_list):
    #     mid = str(paper["id"])
    #     index = paper["index"]
    #     pr = pagerank[index]
    #     mid_to_pr[mid] = pr
    
    # 读取文件，获取论文的index列表
    with open('/home/zhangfanjin/ssj/tot-prediction/data/test_cand_list.json', 'r') as f:
        paper_indices = json.load(f)

    # 读取文件，获取每个论文的index和对应的pids
    with open('/home/zhangfanjin/ssj/tot-prediction/data/dpv12_last.json', 'r') as f:
        papers = json.load(f)

    # 读取文件，获取node_to_index字典
    with open('/home/zhangfanjin/ssj/tot-prediction/data/graph/node_to_idx.json', 'r') as f:
        node_to_index = json.load(f)

    # 读取文件，获取每个文章的logits数据
    with open('/home/zhangfanjin/ssj/RGTN-NIE/results/FB15k_rel_two_inf/logits_tot_geni_1.pk', 'rb') as f:
        paper_logits = pickle.load(f)

    paper_index_to_pid = {str(p['index']) : p['id'] for p in papers}
    
    # 创建结果字典
    result_dict = {}

    # 遍历论文的index列表
    for paper_index in paper_indices:
        if str(paper_index) in node_to_index:
            # 获取论文的有序index
            ordered_index = node_to_index[str(paper_index)]
        
            # 获取论文的pids
            pids = paper_index_to_pid[str(paper_index)]
            
            # 获取论文的logits
            logits = paper_logits[ordered_index]
            
            # 将pids和logits存放在结果字典中
            result_dict[pids] = logits
    
    metrics = []
    for cur_key in tqdm(venue_year_to_candidates_test):
        cur_paper_to_label = venue_year_to_candidates_test[cur_key]
        labels = []
        pr_scores = []
        for pid in cur_paper_to_label:
            if int(pid) in result_dict:
                labels.append(cur_paper_to_label[pid])
                pr_scores.append(result_dict[int(pid)])
        metrics.append(average_precision_score(labels, pr_scores))
    print("MAP: ", sum(metrics) / len(metrics))


if __name__ == "__main__":
    # cal_paper_pagerank()
    eval_pagerank()
