import os
import pandas as pd
import json
from tqdm import tqdm
from bson import ObjectId
from collections import defaultdict as dd

import utils


def gen_candidate_papers(role="test"):
    tot_papers = utils.load_json("data/", "tot_papers_{}.json".format(role))
    all_tot_papers = utils.load_json("data/", "tot_paper_dict_with_mag_id_copy.json")
    tot_mids = set()
    for pid in tqdm(all_tot_papers):
        paper = all_tot_papers[pid]
        if "mag_id" in paper:
            tot_mids.add(str(paper["mag_id"]))

    cs_paper_list = utils.load_json("data/", "dpv12_last.json")
    cs_paper_dict = {str(p["id"]): p for p in cs_paper_list}
    venue_year_to_papers = dd(set)

    for pid in tqdm(cs_paper_dict):
        paper = cs_paper_dict[pid]
        vid = paper.get("venue", {}).get("id")
        year = paper.get("year")
        if vid and year:
            cur_key = "{}_{}".format(vid, year)
            venue_year_to_papers[cur_key].add(pid)
    
    venue_year_to_candidates = dd(dict)
    n_pos = 0
    n_neg = 0
    for paper in tqdm(tot_papers):
        mid = paper["mag_id"]
        p_hit = cs_paper_dict[mid]
        vid = p_hit.get("venue", {}).get("id")
        year = p_hit.get("year")
        if vid and year:
            cur_key = "{}_{}".format(vid, year)
            cur_pids_cand = venue_year_to_papers[cur_key]
            for cand in cur_pids_cand:
                if cand == mid:
                    venue_year_to_candidates[cur_key][cand] = 1
                    n_pos += 1
                elif cand not in tot_mids and cand not in venue_year_to_candidates.get(cur_key, {}):
                    venue_year_to_candidates[cur_key][cand] = 0
                    n_neg += 1
    
    print(n_pos, n_neg)

    utils.dump_json(venue_year_to_candidates, "data/", "venue_year_to_candidates_{}.json".format(role))


if __name__ == "__main__":
    gen_candidate_papers(role="train")
    gen_candidate_papers(role="valid")
    gen_candidate_papers(role="test")
