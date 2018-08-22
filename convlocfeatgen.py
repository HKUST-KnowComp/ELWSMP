import json
import logging
import os
from os.path import isfile
import numpy as np
import pandas as pd
from scipy import sparse

from utils import utils
import config


def __get_reviews_by_id(reviews_file, review_ids):
    reviews = dict()
    f = open(reviews_file, encoding='utf-8')
    for line in f:
        review = json.loads(line)
        review_id = review['review_id']
        if review_id in review_ids:
            reviews[review_id] = review
    f.close()
    return reviews


def __location_dist(lat0, long0, lat1, long1):
    return abs(lat0 - lat1) + abs(long0 - long1)


def __is_acronym(s0, s1):
    if ' ' in s0:
        return 0
    words = s1.split(' ')
    pos = 0
    for w in words:
        if not w or (not w[0].isalpha()):
            continue
        if pos >= len(s0):
            return 0
        if s0[pos].lower() != w[0].lower():
            return 0
        pos += 1
    return 1 if pos == len(s0) else 0


def __ordinary_features_for_candidate(reviewed_biz, cand_biz, name_str, feats_include):
    rev_biz_id = reviewed_biz['business_id']
    cand_biz_id = cand_biz['business_id']
    feats = list()

    if 'NAME-SIM' in feats_include:
        name_sim = utils.words_jaccard_sim(name_str, cand_biz['name'])
        feats.append(name_sim)

    if 'IS-ACRONYM' in feats_include:
        is_acronym = __is_acronym(name_str, cand_biz['name'])
        # print(name_str, cand_biz['name'], is_acronym)
        feats.append(is_acronym)

    if 'SELF-REF' in feats_include:
        self_ref = 1 if cand_biz_id == rev_biz_id else 0
        feats.append(self_ref)

    # reviewed_biz_name = reviewed_biz['name']
    # self_ref0, self_ref1 = 0, 0
    # if reviewed_biz_name.lower() == name_str.lower() and cand_biz_id == rev_biz_id:
    #     self_ref0 = 1
    # if name_str.lower() in reviewed_biz_name.lower() and cand_biz_id == rev_biz_id:
    #     self_ref1 = 1

    if 'CITY-MATCH' in feats_include:
        city_match = 1 if cand_biz['city'] == reviewed_biz['city'] else 0
        feats.append(city_match)

    if 'GEO-DIST' in feats_include:
        loc_dist = __location_dist(reviewed_biz['latitude'], reviewed_biz['longitude'],
                                   cand_biz['latitude'], cand_biz['longitude'])
        # print(loc_dist)
        loc_dist = min(1.0, loc_dist / 0.01)
        feats.append(loc_dist)

    return feats


def __get_businesses_by_id(biz_file, biz_ids):
    businesses = dict()
    f = open(biz_file, encoding='utf-8')
    for line in f:
        biz = json.loads(line)
        biz_id = biz['business_id']
        if biz_id in biz_ids:
            businesses[biz_id] = biz
    f.close()
    return businesses


def __load_tfidf_vecs(filename, id_name):
    ids = list()
    f = open(filename, 'r', encoding='utf-8')
    data, indices, indptr = list(), list(), [0]
    for i, line in enumerate(f):
        v = json.loads(line)
        ids.append(v[id_name])
        idxs, vals = v['indices'], v['vals']
        data += vals
        indices += idxs
        indptr.append(len(data))
        if i % 100000 == 0:
            print(i)
    f.close()

    X = sparse.csr_matrix((data, indices, indptr), (len(ids), config.YELP_N_TFIDF_VOCAB_LEN), dtype=np.float32)
    return ids, X


def get_ordinary_features(mentions, mention_candidates, reviews_file, biz_file, feats_include):
    print('extracting features ...')

    if 'TEXT-SIM' in feats_include:
        biz_ids_tfidf, biz_tfidf_vecs = __load_tfidf_vecs(config.YELP_BIZ_TF_FILE, 'business_id')
        review_ids_tfidf, review_tfidf_vecs = __load_tfidf_vecs(config.YELP_REVIEW_TF_FILE, 'review_id')
        biz_id_tfidf_to_idx = {biz_id: i for i, biz_id in enumerate(biz_ids_tfidf)}
        review_id_tfidf_to_idx = {review_id: i for i, review_id in enumerate(review_ids_tfidf)}

    df = pd.read_csv(config.YELP_BIZ_REVIEW_NUM_FILE, header=None)
    biz_rev_num_dict = {biz_id: n for biz_id, n in df.itertuples(False, None)}

    mention_review_ids = set([m['doc_id'] for m in mentions])
    reviews = __get_reviews_by_id(reviews_file, mention_review_ids)
    related_biz_ids = {r['business_id'] for r in reviews.values()}
    for candidates in mention_candidates.values():
        for cand_biz in candidates:
            related_biz_ids.add(cand_biz)
    businesses = __get_businesses_by_id(biz_file, related_biz_ids)

    mention_cand_feats = dict()
    for m in mentions:
        mention_id = m['mention_id']
        name_str = m['name_str']
        review = reviews[m['doc_id']]
        reviewed_biz_id = review['business_id']
        reviewed_biz = businesses[reviewed_biz_id]

        if 'TEXT-SIM' in feats_include:
            tfidf_vec_rev = review_tfidf_vecs[review_id_tfidf_to_idx[m['doc_id']]]
            tfidf_vec_rev_norm = np.sqrt(np.sum(tfidf_vec_rev.data * tfidf_vec_rev.data))

        cand_feats = list()
        candidates = mention_candidates[mention_id]
        for biz_id in candidates:
            cand_biz = businesses[biz_id]
            feat = __ordinary_features_for_candidate(reviewed_biz, cand_biz, name_str, feats_include)

            if 'POPULARITY' in feats_include:
                rev_num = biz_rev_num_dict[biz_id]
                feat.append(min(rev_num / 100.0, 1.0))

            if 'TEXT-SIM' in feats_include:
                tfidf_vec_biz = biz_tfidf_vecs[biz_id_tfidf_to_idx[biz_id]]
                tfidf_vec_biz_norm = np.sqrt(np.sum(tfidf_vec_rev.data * tfidf_vec_rev.data))
                tfidf_sim = tfidf_vec_rev.dot(tfidf_vec_biz.T) / tfidf_vec_rev_norm / tfidf_vec_biz_norm
                tfidf_sim = tfidf_sim.data[0] if tfidf_sim.data else 0
                feat.append(tfidf_sim)

            cand_feats.append(feat)
        mention_cand_feats[mention_id] = np.asarray(cand_feats, np.float32)
    return mention_cand_feats


def gen_ordinary_features_file(mentions_file, candidates_file, review_file, biz_file, dst_file):
    print('generation ordinary features ...')
    mentions = utils.load_json_objs(mentions_file)
    mention_candidates = utils.load_candidates(candidates_file)
    mention_cand_feats = get_ordinary_features(
        mentions, mention_candidates, review_file, biz_file, feats_include)
    fout = open(dst_file, 'w', encoding='utf-8', newline='\n')
    for m in mentions:
        mention_id = m['mention_id']
        candidates = mention_candidates[mention_id]
        feats = mention_cand_feats[mention_id]
        utils.write_candidate_features(mention_id, candidates, feats, fout)
    fout.close()


if __name__ == '__main__':
    candidates_file = os.path.join(config.YELP_DATA_DIR, 'dataset/candidates.json')

    # dst_feat_file = os.path.join(config.LOCAL_DATA_DIR, 'convloc-features.txt')
    # dst_feat_file = os.path.join(config.LOCAL_DATA_DIR, 'conv-features.txt')
    dst_feat_file = os.path.join(config.LOCAL_DATA_DIR, 'loc-features.txt')

    # feats_include = {'NAME-SIM', 'IS-ACRONYM', 'SELF-REF', 'TEXT-SIM', 'POPULARITY', 'CITY-MATCH', 'GEO-DIST'}
    # feats_include = {'NAME-SIM', 'IS-ACRONYM', 'SELF-REF', 'TEXT-SIM', 'POPULARITY'}
    feats_include = {'CITY-MATCH', 'GEO-DIST'}
    # if not isfile(yelp_ordinary_cand_feat_file):
    gen_ordinary_features_file(
        config.YELP_ALL_MENTIONS_FILE, candidates_file, config.YELP_REVIEW_FILE, config.YELP_BIZ_FILE, dst_feat_file)
