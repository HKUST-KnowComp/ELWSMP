import json
import numpy as np


def load_json_objs(json_file):
    objs = list()
    f = open(json_file, 'r', encoding='utf-8')
    for line in f:
        obj = json.loads(line)
        objs.append(obj)
    f.close()
    return objs


def load_candidates_for_mentions(candidates_file, mention_ids):
    candidates_objs = load_json_objs(candidates_file)
    return {c['mention_id']: c['target_ids'] for c in candidates_objs if c['mention_id'] in mention_ids}


def get_mention_cadidate_idxs_dict(mention_candidates, mention_id_idx_dict, biz_id_idx_dict):
    mention_candidates_idxs = dict()
    for mention_id, candidates in mention_candidates.items():
        mention_idx = mention_id_idx_dict[mention_id]
        mention_candidates_idxs[mention_idx] = cur_candidates = list()
        for c in candidates:
            cur_candidates.append(biz_id_idx_dict[c])
    return mention_candidates_idxs


def write_candidate_features(mention_id, candidates, features, fout):
    fout.write('%s %d\n' % (mention_id, len(candidates)))
    for cand_id, feat in zip(candidates, features):
        fout.write('%s' % cand_id)
        for v in feat:
            fout.write(' {}'.format(v))
        fout.write('\n')


def load_candidates(candidates_file, candidate_id_name='target_ids'):
    candidates_objs = load_json_objs(candidates_file)
    cands_dict = dict()
    for c in candidates_objs:
        cands = c[candidate_id_name]
        # random.shuffle(cands)
        cands_dict[c['mention_id']] = cands
    return cands_dict


def load_id_to_idx(id_idx_file, skip_rows=0, implicit_idx=False, sep='\t'):
    f = open(id_idx_file, encoding='utf-8')
    for _ in range(skip_rows):
        next(f)

    if implicit_idx:
        id_to_idx = {line.strip(): i for i, line in enumerate(f)}
    else:
        id_to_idx = dict()
        for line in f:
            obj_id, idx = line.strip().split(sep)
            idx = int(idx)
            id_to_idx[obj_id] = idx

    f.close()
    return id_to_idx


def load_idx_to_id(id_idx_file):
    ids = list()
    f = open(id_idx_file, encoding='utf-8')
    for line in f:
        mid, idx = line.strip().split('\t')
        ids.append(mid)
    f.close()
    return ids


def get_y_true(mentions, mention_candidates_dict, target_id_name='target_id'):
    y_true = -np.ones(len(mentions), np.int32)
    for i, m in enumerate(mentions):
        candidates = mention_candidates_dict[m['mention_id']]
        if target_id_name not in m:
            continue
        try:
            y_true[i] = candidates.index(m[target_id_name])
        except ValueError:
            pass
    return y_true


def jaccard_sim(set0, set1):
    cnt = 0
    for e in set0:
        if e in set1:
            cnt += 1
    return float(cnt) / (len(set0) + len(set1) - cnt)


def words_jaccard_sim(s0, s1):
    words0 = set(s0.lower().split(' '))
    words1 = set(s1.lower().split(' '))
    return jaccard_sim(words0, words1)


def ssd(v):
    return np.sqrt(sum((v - v.mean()) ** 2) / (len(v) - 1))
