from tqdm import tqdm
import networkx as nx
from sklearn.metrics import average_precision_score
import utils


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
    pagerank = utils.joblib_load_obj("data/graph/", "pagerank.pkl")
    cs_paper_list = utils.load_json("data/", "dpv12_last.json")

    venue_year_to_candidates_test = utils.load_json("data/", "venue_year_to_candidates_test.json")
    mid_to_pr = {}

    for paper in tqdm(cs_paper_list):
        mid = str(paper["id"])
        index = paper["index"]
        pr = pagerank[index]
        mid_to_pr[mid] = pr
    
    metrics = []
    for cur_key in tqdm(venue_year_to_candidates_test):
        cur_paper_to_label = venue_year_to_candidates_test[cur_key]
        labels = []
        pr_scores = []
        for pid in cur_paper_to_label:
            labels.append(cur_paper_to_label[pid])
            pr_scores.append(mid_to_pr[pid])
        metrics.append(average_precision_score(labels, pr_scores))
    print("MAP: ", sum(metrics) / len(metrics))


if __name__ == "__main__":
    cal_paper_pagerank()
    eval_pagerank()
