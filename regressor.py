import os
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict as dd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import average_precision_score

import utils


def gen_classifier_features_train():
    venue_year_to_candidates_train = utils.load_json("data/", "venue_year_to_candidates_train.json")
    venue_year_to_candidates_valid = utils.load_json("data/", "venue_year_to_candidates_valid.json")
    venue_year_to_candidates = deepcopy(venue_year_to_candidates_train)
    for cur_key in venue_year_to_candidates_valid:
        cur_pid_to_label = venue_year_to_candidates_valid[cur_key]
        for pid in cur_pid_to_label:
            if pid not in venue_year_to_candidates.get(cur_key, {}):
                if cur_key not in venue_year_to_candidates:
                    venue_year_to_candidates[cur_key] = {}
                venue_year_to_candidates[cur_key][pid] = cur_pid_to_label[pid]

    tot_papers_train = utils.load_json("data/", "tot_papers_train.json")
    tot_papers_valid = utils.load_json("data/", "tot_papers_valid.json")
    tot_papers = tot_papers_train + tot_papers_valid

    cs_paper_list = utils.load_json("data/", "dpv12_last.json")
    cs_paper_dict = {str(p["id"]): p for p in cs_paper_list}

    mids_tot_set = set()
    for paper in tqdm(tot_papers):
        mids_tot_set.add(paper["mag_id"])

    paper_per_year_citations = dd(lambda: dd(int))
    for pid in tqdm(cs_paper_dict):
        paper = cs_paper_dict[pid]
        refs = paper.get("references", [])
        year = paper["year"]
        if refs is None:
            continue
        for ref_id in refs:
            paper_per_year_citations[str(ref_id)][year] += 1

    venue_year_to_delta_year = {}
    for paper in tqdm(tot_papers):
        mid = paper["mag_id"]
        p_hit = cs_paper_dict[mid]
        vid = p_hit.get("venue", {}).get("id")
        year = p_hit.get("year")
        if vid and year:
            cur_key = "{}_{}".format(vid, year)
            year_award = paper["award_year"]
            try:
                assert year_award > year
                venue_year_to_delta_year[cur_key] = year_award - year
            except:
                print("delta year", paper)
                venue_year_to_delta_year[cur_key] = 5
    
    pid_to_venue_year = {}
    for cur_key in tqdm(venue_year_to_candidates):
        cur_cands = venue_year_to_candidates[cur_key]
        for pid in cur_cands:
            pid_to_venue_year[pid] = cur_key
    
    np.random.seed(415)

    pos_papers = list(mids_tot_set)
    neg_papers = []
    for paper in tqdm(pos_papers):
        mid = paper
        p_hit = cs_paper_dict[mid]
        vid = p_hit.get("venue", {}).get("id")
        year = p_hit.get("year")
        if vid and year:
            cur_key = "{}_{}".format(vid, year)
            cands = venue_year_to_candidates[cur_key].keys()
            cands_neg = [p for p in cands if p not in mids_tot_set]
            if len(cands_neg) == 0:
                continue
            cands_neg_sample = np.random.choice(cands_neg, 10, replace=True)
            neg_papers += cands_neg_sample.tolist()
    print(len(pos_papers), len(neg_papers))
    labels = [1] * len(pos_papers) + [0] * len(neg_papers)
    papers_train = pos_papers + neg_papers
    features = []
    for pid in tqdm(papers_train):
        year = cs_paper_dict[pid]["year"]
        citation_total = 0
        citation_list = []
        last_citation = 0
        cur_key = pid_to_venue_year[str(pid)]
        cur_delta = venue_year_to_delta_year[cur_key]
        if cur_delta >= 10:
            for yr in range(year, year + 10):
                cur_yr_citation = paper_per_year_citations.get(pid, {}).get(yr, 0)
                citation_list.append(cur_yr_citation)
                citation_total += cur_yr_citation
        else:
            for yr in range(year, year + cur_delta):
                cur_yr_citation = paper_per_year_citations.get(pid, {}).get(yr, 0)
                citation_list.append(cur_yr_citation)
                citation_total += cur_yr_citation
                if cur_yr_citation != 0:
                    last_citation = cur_yr_citation
            
            for yr in range(cur_delta, 10):
                citation_list.append(last_citation)
        
        citation_list.append(citation_total)
        features.append(citation_list)
    
    np.save("out/train_features.npy", np.array(features))
    np.save("out/train_labels.npy", labels)


def train_and_predict_classifier():
    venue_year_to_candidates = utils.load_json("data/", "venue_year_to_candidates_test.json")
    tot_papers = utils.load_json("data/", "tot_papers_test.json")

    cs_paper_list = utils.load_json("data/", "dpv12_last.json")
    cs_paper_dict = {str(p["id"]): p for p in cs_paper_list}

    paper_per_year_citations = dd(lambda: dd(int))
    for pid in tqdm(cs_paper_dict):
        paper = cs_paper_dict[pid]
        refs = paper.get("references", [])
        year = paper["year"]
        if refs is None:
            continue
        for ref_id in refs:
            paper_per_year_citations[str(ref_id)][year] += 1

    venue_year_to_delta_year = {}
    for paper in tqdm(tot_papers):
        mid = paper["mag_id"]
        p_hit = cs_paper_dict[mid]
        vid = p_hit.get("venue", {}).get("id")
        year = p_hit.get("year")
        if vid and year:
            cur_key = "{}_{}".format(vid, year)
            year_award = paper["award_year"]
            try:
                assert year_award > year
                venue_year_to_delta_year[cur_key] = year_award - year
            except:
                print("delta year", paper)
                venue_year_to_delta_year[cur_key] = 5
    
    venue_year_to_candidate_features_labels = {}
    for cur_key in tqdm(venue_year_to_candidates):
        cur_pid_to_labels = venue_year_to_candidates[cur_key]
        pids = cur_pid_to_labels.keys()
        cur_labels = [cur_pid_to_labels[x] for x in pids]
        features = []
        for pid in pids:
            year = cs_paper_dict[pid]["year"]
            citation_total = 0
            citation_list = []
            last_citation = 0
            cur_delta = venue_year_to_delta_year[cur_key]   

            if cur_delta >= 10:
                for yr in range(year, year + 10):
                    cur_yr_citation = paper_per_year_citations.get(pid, {}).get(yr, 0)
                    citation_list.append(cur_yr_citation)
                    citation_total += cur_yr_citation
            else:
                for yr in range(year, year + cur_delta):
                    cur_yr_citation = paper_per_year_citations.get(pid, {}).get(yr, 0)
                    citation_list.append(cur_yr_citation)
                    citation_total += cur_yr_citation
                    if cur_yr_citation != 0:
                        last_citation = cur_yr_citation
                
                for yr in range(cur_delta, 10):
                    citation_list.append(last_citation)  

            citation_list.append(citation_total)
            features.append(citation_list)   
        venue_year_to_candidate_features_labels[cur_key] = {}
        venue_year_to_candidate_features_labels[cur_key]["features"] = np.array(features)
        venue_year_to_candidate_features_labels[cur_key]["labels"] = cur_labels
    
    x_train = np.load("out/train_features.npy")
    y_train = np.load("out/train_labels.npy")
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scale = scaler.transform(x_train)

    models = ["RandomForestClassifier", "GradientBoostingClassifier"]
    for model in models:
        print("=======================================")
        print(model)
        classifier = eval(model)()
        classifier.fit(x_train_scale, y_train)
        metrics = []
        for cur_key in venue_year_to_candidate_features_labels:
            cur_data = venue_year_to_candidate_features_labels[cur_key]
            cur_feature = cur_data["features"]
            cur_labels = cur_data["labels"]
            cur_feature = scaler.transform(cur_feature)
            cur_pred = classifier.predict_proba(cur_feature)[:, 1]
            cur_map = average_precision_score(cur_labels, cur_pred)
            metrics.append(cur_map)
        print("mean map", np.mean(metrics))
        print()


if __name__ == "__main__":
    # gen_classifier_features_train()
    train_and_predict_classifier()
