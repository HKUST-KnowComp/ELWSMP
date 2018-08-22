import numpy as np
from collections import namedtuple


DataTrain = namedtuple('DataTrain', ['x_c', 'x_w'])
DataTest = namedtuple('DataTest', ['x', 'y'])


def get_bnd_feat_train(training_instances_df, mention_cand_feats):
    feats_dict = next(iter(mention_cand_feats.values()))
    cdim = next(iter(feats_dict.values())).shape[0]

    n = len(training_instances_df)
    x_c = np.zeros((n, cdim), np.float32)
    x_w = np.zeros((n, cdim), np.float32)
    for i, (mention_id, gold_biz_id, crpt_biz_id) in enumerate(training_instances_df.itertuples(False, None)):
        cur_mention_cand_feats = mention_cand_feats[mention_id]
        x_c[i] = cur_mention_cand_feats[gold_biz_id]
        x_w[i] = cur_mention_cand_feats[crpt_biz_id]
    return DataTrain(x_c, x_w)


def get_bnd_feat_test(mentions, mention_candidates_dict, mention_cand_feats):
    feats_dict = next(iter(mention_cand_feats.values()))
    cdim = next(iter(feats_dict.values())).shape[0]

    X = list()
    for i, m in enumerate(mentions):
        mention_id = m['mention_id']
        cur_mention_cand_feats = mention_cand_feats[mention_id]

        biz_ids = mention_candidates_dict[mention_id]
        num_candidates = len(biz_ids)
        vecs_cand_cs = np.zeros((num_candidates, cdim), dtype=np.float32)
        for j, biz_id in enumerate(biz_ids):
            vecs_cand_cs[j][:] = cur_mention_cand_feats[biz_id]
        X.append(vecs_cand_cs)

    return X


def get_ind_biz_feat_train(training_instance_df, biz_id_to_idx, biz_features):
    dim = biz_features.shape[1]
    n = training_instance_df.shape[0]
    xc = np.zeros((n, dim), np.float32)
    xw = np.zeros((n, dim), np.float32)
    for i, (mention_id, gold_biz_id, crpt_biz_id) in enumerate(training_instance_df.itertuples(False, None)):
        xc[i] = biz_features[biz_id_to_idx[gold_biz_id]]
        xw[i] = biz_features[biz_id_to_idx[crpt_biz_id]]
    return DataTrain(xc, xw)


def get_ind_mention_feat_train(training_instance_df, mention_id_to_idx, mention_features):
    dim = mention_features.shape[1]
    n = training_instance_df.shape[0]
    xc = np.zeros((n, dim), np.float32)
    xw = np.zeros((n, dim), np.float32)
    for i, (mention_id, gold_biz_id, crpt_biz_id) in enumerate(training_instance_df.itertuples(False, None)):
        xc[i] = mention_features[mention_id_to_idx[mention_id]]
        xw[i] = mention_features[mention_id_to_idx[mention_id]]
    return DataTrain(xc, xw)


def get_ind_flat_feat_train(training_instances_df, mention_id_to_idx, biz_id_to_idx,
                            mention_features, biz_features):
    Xc_mention, Xw_mention = get_ind_mention_feat_train(training_instances_df, mention_id_to_idx, mention_features)
    Xc_biz, Xw_biz = get_ind_biz_feat_train(training_instances_df, biz_id_to_idx, biz_features)
    Xc = np.concatenate([Xc_mention, Xc_biz], axis=1)
    Xw = np.concatenate([Xw_mention, Xw_biz], axis=1)
    return DataTrain(Xc, Xw)


def get_ind_flat_feat_test(mentions, mention_candidates_dict, mention_id_to_idx, biz_id_to_idx,
                           mention_features, biz_features):
    x = list()
    for i, m in enumerate(mentions):
        mention_id = m['mention_id']
        mention_idx = mention_id_to_idx[mention_id]
        mention_feat = mention_features[mention_idx]

        biz_ids = mention_candidates_dict[mention_id]
        feat_vecs = list()
        for j, biz_id in enumerate(biz_ids):
            biz_idx = biz_id_to_idx[biz_id]
            feat_vecs.append(np.concatenate((mention_feat, biz_features[biz_idx])))
        x.append(np.asarray(feat_vecs, np.float32))
    return x


def load_candidate_feats(mention_candidates_dict, cand_feat_file):
    print('loading cand feats from {}'.format(cand_feat_file))
    mention_cand_feat = dict()
    f = open(cand_feat_file, encoding='utf-8')
    for line in f:
        mention_id, num_candidates = line.strip().split(' ')
        biz_ids = mention_candidates_dict[mention_id]

        num_candidates = int(num_candidates)
        if num_candidates != len(biz_ids):
            print(mention_id, num_candidates, len(biz_ids))
        assert num_candidates == len(biz_ids)
        feats = dict()
        for i in range(num_candidates):
            line = next(f)
            vals = line.strip().split(' ')
            biz_id = vals[0]
            assert biz_id == biz_ids[i]
            feat = np.asarray([float(v) for v in vals[1:]], np.float32)
            feats[biz_id] = feat
        mention_cand_feat[mention_id] = feats
    f.close()

    return mention_cand_feat


def concatenate_feats(feats_list):
    n = len(feats_list[0])
    x = list()
    for i in range(n):
        x.append(np.concatenate([x_tmp[i] for x_tmp in feats_list], axis=1))
    return x
