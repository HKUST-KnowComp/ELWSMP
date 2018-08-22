import os

import pandas
from scipy.sparse import csr_matrix

from utils import utils
from utils.loggingutils import init_logging
import config


def __get_max_val_in_network(network_mat):
    flg = False
    vmax = 0
    for v in network_mat.data:
        if not flg:
            vmax = v
            flg = True
        elif v > vmax:
            vmax = v
    return vmax


def __load_network_file(network_file, num_rows, num_cols):
    x = pandas.read_csv(network_file, header=None, na_filter=False)
    return csr_matrix((x[2], (x[0], x[1])), shape=(num_rows, num_cols))


def __get_path_count_features(commuting_matrix_file, mention_candidates_idxs, num_mentions, num_bizs, cnt_thres):
    print('loading {} ...'.format(commuting_matrix_file))
    mat = __load_network_file(commuting_matrix_file, num_mentions, num_bizs)

    max_val = float(__get_max_val_in_network(mat))
    print('done. max value', max_val)
    if cnt_thres < 0 or max_val < cnt_thres:
        cnt_thres = max_val

    cnt_thres = float(cnt_thres)
    mention_candidate_features = dict()
    for mention_idx, biz_idxs in mention_candidates_idxs.items():
        r = mat.getrow(mention_idx)
        biz_idx_vals = dict()
        for idx, v in zip(r.indices, r.data):
            # print(idx, v)
            biz_idx_vals[idx] = v

        mention_candidate_features[mention_idx] = cur_features = list()
        for biz_idx in biz_idxs:
            if cnt_thres > 0:
                cur_features.append(min(biz_idx_vals.get(biz_idx, 0), cnt_thres) / cnt_thres)
            else:
                cur_features.append(biz_idx_vals.get(biz_idx, 0))
    return mention_candidate_features


def __write_mention_candidate_features(features_list, mention_candidates, mention_id_idx_dict, dst_file):
    fout = open(dst_file, 'w', encoding='utf-8', newline='\n')
    for mention_id, candidates in mention_candidates.items():
        mention_idx = mention_id_idx_dict[mention_id]

        tmp = [features[mention_idx] for features in features_list]
        features_candidates = [v for v in zip(*tmp)]

        fout.write('%s %d\n' % (mention_id, len(candidates)))
        for c, features in zip(candidates, features_candidates):
            fout.write('%s' % c)
            for f in features:
                fout.write(' %f' % f)
            fout.write('\n')
    fout.close()


def gen_path_count_feats_file(data_info_file, mention_candidates, mention_id_to_idx, biz_id_to_idx,
                              commuting_matrix_files, for_pra, dst_file):
    mention_candidates_idxs = utils.get_mention_cadidate_idxs_dict(
        mention_candidates, mention_id_to_idx, biz_id_to_idx)
    data_info = utils.load_json_objs(data_info_file)[0]
    if for_pra:
        # __get_path_count_features(commuting_matrix_files, mention_candidates_idxs, data_info['mentions'],
        #                           data_info['bizs'], -1)
        features_list = [__get_path_count_features(
            f, mention_candidates_idxs, data_info['mentions'], data_info['bizs'], -1
        ) for f in commuting_matrix_files]
        __write_mention_candidate_features(features_list, mention_candidates, mention_id_to_idx, dst_file)
    else:
        features_list = [__get_path_count_features(
            f, mention_candidates_idxs, data_info['mentions'], data_info['bizs'], NORM_THRES
        ) for f in commuting_matrix_files]
        __write_mention_candidate_features(features_list, mention_candidates, mention_id_to_idx, dst_file)


def __gen_yelp_path_count_feat(mentions_file, mention_id_idx_file, path_strs, for_pra, dst_file):
    mentions = utils.load_json_objs(mentions_file)
    mention_ids = {m['mention_id'] for m in mentions}
    mention_candidates = utils.load_candidates_for_mentions(config.YELP_CANDIDATES_FILE, mention_ids)
    mention_id_to_idx = utils.load_id_to_idx(mention_id_idx_file)
    biz_id_to_idx = utils.load_id_to_idx(config.YELP_BIZ_ID_TO_IDX_FILE)
    if for_pra:
        commuting_matrix_files = [os.path.join(
            config.YELP_DATA_DIR, 'network/{}_norm.txt'.format(s)) for s in path_strs]
    else:
        commuting_matrix_files = [os.path.join(config.YELP_DATA_DIR, 'network/{}.txt'.format(s)) for s in path_strs]
    gen_path_count_feats_file(config.YELP_DATA_INFO_FILE, mention_candidates, mention_id_to_idx, biz_id_to_idx,
                              commuting_matrix_files, for_pra, dst_file)


if __name__ == '__main__':
    init_logging('log/pc_feature_gen.log', mode='a', to_stdout=True)

    yelp_data_info_file = os.path.join(config.YELP_DATA_DIR, 'dataset-info.json')
    yelp_candidates_file = os.path.join(config.YELP_DATA_DIR, 'dataset/candidates.json')
    yelp_cs_candidates_file = os.path.join(config.YELP_DATA_DIR, 'casestudy/cs-mention-candidates.txt')

    # path_strs = ['MRURB']
    # path_strs = ['MRURBRURB']
    path_strs = ['MRUURB']
    tag = 'pc'
    # tag = 'rw'
    NORM_THRES = 100
    yelp_path_count_feat_file = os.path.join(
        config.LOCAL_DATA_DIR, '{}_features_{}.txt'.format(tag, path_strs[0]))
    __gen_yelp_path_count_feat(config.YELP_ALL_LINKED_MENTIONS_FILE, config.YELP_MENTION_ID_IDX_FILE, path_strs,
                               False, yelp_path_count_feat_file)
    # __gen_yelp_path_count_feat(YELP_ALL_LINKED_MENTIONS_FILE, YELP_MENTION_ID_IDX_FILE, path_strs,
    #                            tag == 'rw', yelp_path_count_feat_file)
