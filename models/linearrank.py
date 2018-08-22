import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

from utils import utils, modelutils


class LinearRank:
    def __init__(self, training_instance_file, val_mentions_file, test_mentions_file, candidates_file,
                 cand_feat_files, mention_id_idx_file, biz_id_to_idx_file, mention_feat_files, biz_feat_files,
                 learning_rate, n_epochs, l2_reg, batch_size):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.l2_reg = l2_reg
        self.batch_size = batch_size

        logging.info('Test mentions file: %s', test_mentions_file)

        mention_candidates_dict = utils.load_candidates(candidates_file, 'target_ids')
        biz_id_idx_dict = utils.load_id_to_idx(biz_id_to_idx_file)

        mention_feat_list = [pd.read_csv(
            f, header=None, na_filter=False).as_matrix() for f in mention_feat_files]
        biz_feat_list = [pd.read_csv(
            f, header=None, na_filter=False).as_matrix() for f in biz_feat_files]
        mention_id_to_idx = utils.load_id_to_idx(mention_id_idx_file)
        training_instances_df = pd.read_csv(training_instance_file, header=None, na_filter=False)
        train_feat_list = [modelutils.get_ind_flat_feat_train(
            training_instances_df, mention_id_to_idx, biz_id_idx_dict, mf, bf
        ) for mf, bf in zip(mention_feat_list, biz_feat_list)]

        mention_cand_feats_list = None
        if cand_feat_files:
            mention_cand_feats_list = [modelutils.load_candidate_feats(
                mention_candidates_dict, f) for f in cand_feat_files]
            bnd_feat_list = [modelutils.get_bnd_feat_train(
                training_instances_df, feats) for feats in mention_cand_feats_list]
            train_feat_list += bnd_feat_list

        self.Xc_all = np.concatenate([Xc for Xc, _ in train_feat_list], axis=1)
        self.Xw_all = np.concatenate([Xw for _, Xw in train_feat_list], axis=1)

        def get_flat_feats_test(mentions_file):
            mentions = utils.load_json_objs(mentions_file)
            feats_list = [modelutils.get_ind_flat_feat_test(
                mentions, mention_candidates_dict, mention_id_to_idx, biz_id_idx_dict, mf, bf
            ) for mf, bf in zip(mention_feat_list, biz_feat_list)]

            if mention_cand_feats_list is not None:
                X_bnd_list = [modelutils.get_bnd_feat_test(
                    mentions, mention_candidates_dict, feats) for feats in mention_cand_feats_list]
                feats_list += X_bnd_list

            X = modelutils.concatenate_feats(feats_list)
            y_true = utils.get_y_true(mentions, mention_candidates_dict, 'target_id')
            return modelutils.DataTest(X, y_true)

        self.x_cand_cs_val, self.y_gold_val = get_flat_feats_test(val_mentions_file)
        self.feats_list_test, self.y_gold_test = get_flat_feats_test(test_mentions_file)
        self.w_val = None

    def train(self):
        n_train, cdim = self.Xc_all.shape
        n_train_batches = int(n_train / self.batch_size)

        logging.info(
            'Learning rate=%f, dim=%d, n_train=%d, n_epochs=%d, batch_size=%d, l2_reg=%f',
            self.learning_rate, cdim, n_train, self.n_epochs, self.batch_size, self.l2_reg
        )

        x_cand_cs_gold = tf.placeholder("float", [None, cdim])
        x_cand_cs_crpt = tf.placeholder("float", [None, cdim])
        # x_test = tf.constant(test_vecs)
        ones = tf.ones([self.batch_size])
        zeros = tf.zeros([self.batch_size])
        # w = tf.Variable(tf.random_normal([cdim, 1]))
        w = tf.Variable(tf.random_uniform([cdim, 1], -0.1, 0.1))

        scores_gold = LinearRank.__rank_score(x_cand_cs_gold, w)
        scores_crpt = LinearRank.__rank_score(x_cand_cs_crpt, w)

        loss = tf.reduce_mean(tf.maximum(zeros, ones - scores_gold + scores_crpt)) + self.l2_reg * tf.nn.l2_loss(w)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)

        w_value = sess.run([w])
        accuracy_val = LinearRank.__test(self.x_cand_cs_val, self.y_gold_val, w_value)
        accuracy_test = LinearRank.__test(self.feats_list_test, self.y_gold_test, w_value)
        logging.info('Val Acc: %f, Test Acc: %f', accuracy_val, accuracy_test)

        max_val_accuracy = 0
        self.w_val = w_value
        for epoch in range(self.n_epochs):
            losses = list()
            for i in range(n_train_batches):
                batch_x_cand_cs_gold = self.Xc_all[i * self.batch_size: (i + 1) * self.batch_size]
                batch_x_cand_cs_crpt = self.Xw_all[i * self.batch_size: (i + 1) * self.batch_size]
                _, c = sess.run([train, loss], feed_dict={x_cand_cs_gold: batch_x_cand_cs_gold,
                                                          x_cand_cs_crpt: batch_x_cand_cs_crpt})
                losses.append(c)

            loss_val = np.mean(losses)

            w_value = sess.run([w])[0]
            accuracy_val = LinearRank.__test(self.x_cand_cs_val, self.y_gold_val, w_value)
            if max_val_accuracy < accuracy_val:
                accuracy_test = LinearRank.__test(self.feats_list_test, self.y_gold_test, w_value)
                logging.info('Iter %d, Loss: %f, Val Acc: %f, Test Acc: %f', epoch, loss_val, accuracy_val,
                             accuracy_test)
                max_val_accuracy = accuracy_val
                self.w_val = w_value
            else:
                logging.info('Iter %d, Loss: %f, Val Acc: %f', epoch, loss_val, accuracy_val)
        return accuracy_test

    @staticmethod
    def __rank_score(x_cand, w):
        return tf.reshape(tf.matmul(x_cand, w), [-1])

    @staticmethod
    def __test(x_cand_cs, y_gold, w_val):
        y_pred, _ = LinearRank.pred_with_param(x_cand_cs, w_val)
        return accuracy_score(y_gold, y_pred)

    def pred(self, cand_feats_list):
        return LinearRank.pred_with_param(cand_feats_list, self.w_val)

    def pred_test(self):
        return self.pred(self.feats_list_test)[0]

    @staticmethod
    def pred_with_param(cand_feats_list, w_val):
        y_pred, max_scores = list(), list()
        for i, cand_vecs in enumerate(cand_feats_list):
            if len(cand_vecs) == 0:
                y_pred.append(0)
                max_scores.append(0)
                continue
            scores = np.dot(cand_vecs, w_val).flatten()
            rank = np.argmax(scores, axis=0)
            max_scores.append(scores[rank])
            y_pred.append(rank)
        return y_pred, max_scores


def feats_for_model(mentions, mention_candidates_dict, mention_id_to_idx, biz_id_idx_dict,
                    mention_cand_feats_list, mention_feat_list, biz_feat_list):
    feats_list = [modelutils.get_ind_flat_feat_test(
        mentions, mention_candidates_dict, mention_id_to_idx, biz_id_idx_dict, mf, bf
    ) for mf, bf in zip(mention_feat_list, biz_feat_list)]

    if mention_cand_feats_list is not None:
        X_bnd_list = [modelutils.get_bnd_feat_test(
            mentions, mention_candidates_dict, feats) for feats in mention_cand_feats_list]
        feats_list += X_bnd_list

    return modelutils.concatenate_feats(feats_list)