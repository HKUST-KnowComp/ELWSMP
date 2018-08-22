import datetime
import os
import numpy as np
import logging
from utils import utils
from utils.loggingutils import init_logging
from models.linearrank import LinearRank
import config


str_today = datetime.date.today().strftime('%y-%m-%d')
init_logging('log/flat-yelp-{}.log'.format(str_today), mode='a', to_stdout=True)


def __run_linearrank(training_instances_file, val_linked_mentions_file, test_linked_mentions_file):
    flat_model = LinearRank(
        training_instances_file, val_linked_mentions_file, test_linked_mentions_file,
        config.YELP_CANDIDATES_FILE, cand_feat_files, config.YELP_MENTION_ID_IDX_FILE,
        config.YELP_BIZ_ID_TO_IDX_FILE, mention_feat_files, biz_feat_files,
        learning_rate, n_epochs, l2_reg, batch_size)
    acc_list = list()
    for i in range(n_rounds):
        print('Round {}'.format(i))
        acc = flat_model.train()
        acc_list.append(acc)

    avg_best_acc = sum(acc_list) / len(acc_list)
    # ssd_val = 0 if len(acc_list) == 1 else utils.ssd(np.asarray(acc_list))
    logging.info('rounds={}, acc={}'.format(n_rounds, avg_best_acc))
    return avg_best_acc


meta_paths = ['MRURB', 'MRURBRURB', 'MRUURB']
l2_reg = 0.001
learning_rate = 0.001
batch_size = 3
use_non_social_feat = True
non_social_feat_file = ''
n_epochs = 20
n_rounds = 1

ns_cand_feat_file = os.path.join(config.LOCAL_DATA_DIR, 'convloc-features.txt')
# ns_cand_feat_file = os.path.join(config.LOCAL_DATA_DIR, 'conv-features.txt')

pc_cand_feat_files = [os.path.join(
    config.LOCAL_DATA_DIR, 'pc_features_{}.txt'.format(s)) for s in meta_paths]
cand_feat_files = pc_cand_feat_files
if use_non_social_feat:
    cand_feat_files += [ns_cand_feat_file]

mention_feat_files, biz_feat_files = list(), list()
acc_list = list()
for fold_idx in range(5):
    training_instances_file = os.path.join(config.LOCAL_DATA_DIR, '5fold/train_instances-{}.txt'.format(fold_idx))
    val_linked_mentions_file = os.path.join(config.LOCAL_DATA_DIR, '5fold/tvt_val_mentions-{}.json'.format(fold_idx))
    test_linked_mentions_file = os.path.join(config.LOCAL_DATA_DIR, '5fold/test_mentions-{}.json'.format(fold_idx))
    acc = __run_linearrank(training_instances_file, val_linked_mentions_file, test_linked_mentions_file)
    acc_list.append(acc)

ssd_val = 0 if len(acc_list) == 1 else utils.ssd(np.asarray(acc_list))
logging.info('avg_acc={:.4f}, ssd={:.4f}'.format(sum(acc_list) / len(acc_list), ssd_val))
