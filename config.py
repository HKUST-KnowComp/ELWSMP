from os import path

YELP_DATA_DIR = 'd:/data/yelp/'
LOCAL_DATA_DIR = 'data/'

# YELP_CANDIDATES_FILE = path.join(YELP_DATA_DIR, 'dataset/candidates.json')
# YELP_MENTION_ID_IDX_FILE = path.join(YELP_DATA_DIR, 'misc/mention_id_to_idx.txt')
# YELP_BIZ_ID_TO_IDX_FILE = path.join(YELP_DATA_DIR, 'misc/biz_id_to_idx.txt')

YELP_CANDIDATES_FILE = path.join(LOCAL_DATA_DIR, 'candidates.json')
YELP_MENTION_ID_IDX_FILE = path.join(LOCAL_DATA_DIR, 'mention_id_to_idx.txt')
YELP_BIZ_ID_TO_IDX_FILE = path.join(LOCAL_DATA_DIR, 'biz_id_to_idx.txt')

YELP_ALL_MENTIONS_FILE = path.join(YELP_DATA_DIR, 'dataset/mentions-all.json')

YELP_BIZ_FILE = path.join(YELP_DATA_DIR, 'srcdata/yelp_academic_dataset_business.json')
YELP_REVIEW_FILE = path.join(YELP_DATA_DIR, 'srcdata/yelp_academic_dataset_review.json')
YELP_USER_FILE = path.join(YELP_DATA_DIR, 'srcdata/yelp_academic_dataset_user.json')
YELP_DATA_INFO_FILE = path.join(YELP_DATA_DIR, 'dataset-info.json')

YELP_N_TOTAL_REVIEWS = 4153150
YELP_N_TFIDF_VOCAB_LEN = 16462

YELP_REVIEW_TF_FILE = path.join(YELP_DATA_DIR, 'misc/review_tf.txt')
YELP_BIZ_TF_FILE = path.join(YELP_DATA_DIR, 'misc/biz_tf.txt')
YELP_BIZ_REVIEW_NUM_FILE = path.join(YELP_DATA_DIR, 'misc/biz_review_num.txt')
YELP_ALL_LINKED_MENTIONS_FILE = path.join(YELP_DATA_DIR, 'dataset/mentions-linked.json')
