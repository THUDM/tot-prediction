import os
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict as dd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import average_precision_score

import utils


def predict_with_only_citation():
    venue_year_to_candidates = utils.load_json("data/", "venue_year_to_candidates_test.json")
    tot_papers_test = utils.load_json("data/", "tot_papers_test.json")

    cs_paper_list = utils.load_json("data/", "dpv12_last.json")
    cs_paper_dict = {str(p["id"]): p for p in cs_paper_list}

    venue_year_to_delta_year = {}

    for paper in tqdm(tot_papers_test):
        mid = paper["mag_id"]
        p_hit = cs_paper_dict[mid]
        vid = p_hit.get("venue", {}).get("id")
        year = p_hit.get("year")
        if vid and year:
            cur_key = "{}_{}".format(vid, year)
            year_award = paper["award_year"]
            assert year_award > year
            venue_year_to_delta_year[cur_key] = year_award - year

    paper_per_year_citations = dd(lambda: dd(int))
    for pid in tqdm(cs_paper_dict):
        paper = cs_paper_dict[pid]
        refs = paper.get("references", [])
        year = paper["year"]
        if refs is None:
            continue
        for ref_id in refs:
            paper_per_year_citations[str(ref_id)][year] += 1
    
    venue_year_to_scores = dd(dict)
    for cur_key in tqdm(venue_year_to_candidates):
        delta_year = min(10, venue_year_to_delta_year[cur_key])
        year = int(cur_key.split("_")[-1])
        cands = venue_year_to_candidates[cur_key]
        for cid in cands:
            cid = str(cid)
            n_citation = 0
            for yr in range(year, year + delta_year):
                n_citation += paper_per_year_citations.get(cid, {}).get(yr, 0)
            venue_year_to_scores[cur_key][cid] = n_citation
    
    out_dir = "out/"
    os.makedirs(out_dir, exist_ok=True)
    utils.dump_json(venue_year_to_scores, out_dir, "venue_year_to_num_citations_test.json")


def eval_only_citation():
    venue_year_to_scores = utils.load_json("out/", "venue_year_to_num_citations_test.json")
    venue_year_to_candidates = utils.load_json("data/", "venue_year_to_candidates_test.json")

    metrics = []
    for cur_key in venue_year_to_candidates:
        cur_pred = []
        cur_labels = []
        for cand in venue_year_to_candidates[cur_key]:
            cur_labels.append(venue_year_to_candidates[cur_key][cand])
            cur_pred.append(venue_year_to_scores[cur_key][cand])
        cur_map = average_precision_score(cur_labels, cur_pred)
        metrics.append(cur_map)
    print(metrics)
    print("mean map", np.mean(metrics))


if __name__ == "__main__":
    predict_with_only_citation()
    eval_only_citation()
    