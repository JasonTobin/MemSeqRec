from collections import defaultdict
import numpy as np
import scipy.special as ss
import tensorflow as tf

from au2actr.constants import SESSION_LEN
from au2actr.models.model import Model
from au2actr.models.core.net import embedding


class ACTR_BPR(Model):
    def __init__(self, sess, params, n_users, n_items, pretrained_embs):
        super().__init__(sess, params, n_users, n_items, pretrained_embs)
        model_params = params['model']['params']
        self.use_reg = model_params.get('use_reg', True)
        self.l2_emb = model_params.get('l2_emb', 0.0)
        self.user_reg = model_params.get('user_regularization', 0.0025)
        self.user_bias_reg = model_params.get('user_bias_regularization', 0.001)
        self.pos_item_reg = model_params.get('pos_item_regularization', 0.0025)
        self.neg_item_reg = model_params.get('neg_item_regularization', 0.00025)
        self.item_bias_reg = model_params.get('item_bias_regularization', 0.001)
        self.activate_actr = model_params.get('activate_actr', False)
        self.bpr_scores = None

    def build_feedict(self, batch, is_training=True):
        feedict = {
            self.is_training: is_training,
            self.user_ids: batch[0]
        }
        if is_training is True:
            feedict[self.pos_ids] = batch[1]
            feedict[self.neg_ids] = batch[2]
        elif self.activate_actr is True:
            feedict[self.ctx_ids] = batch[2]
        return feedict

    def predict(self, feed_dict, top_n=50):
        item_ids = feed_dict['item_ids']
        reco_items = defaultdict(list)
        if self.activate_actr is False:
            scores = -tf.matmul(
                self.users, self.item_embeddings, transpose_b=True)
            scores = self.sess.run(scores, feed_dict['model_feed'])
            for i, (uid, u_scores) in enumerate(zip(feed_dict['user_ids'],
                                                    scores)):
                topn_indices = np.argsort(scores[i])[:top_n]
                reco_items[uid].append([item_ids[idx] for idx in topn_indices])
        else:
            last_embeddings = tf.nn.embedding_lookup(self.item_embeddings,
                                                     self.ctx_ids,
                                                     name='last_embs')
            ctx_emb = tf.reduce_mean(last_embeddings, axis=1)
            scores = -tf.matmul(ctx_emb, self.item_embeddings,
                                transpose_b=True)
            scores = self.sess.run(scores, feed_dict['model_feed'])
            bpr_scores = np.array(scores, dtype=np.float32)
            # actr_scores = []
            # for uid, nxt_idx in zip(feed_dict['user_ids'],
            #                         feed_dict['nxt_indices']):
            #     actr_scores.append(self.actr_scores[uid][nxt_idx].toarray()[0])
            # actr_scores = np.array(actr_scores)
            actr_scores = feed_dict['actr_scores']
            scores = ss.softmax(bpr_scores, axis=-1) - ss.softmax(actr_scores, axis=-1)
            # scores = bpr_scores - actr_scores
            for i, (uid, u_scores) in enumerate(
                    zip(feed_dict['user_ids'], scores)):
                topn_indices = np.argsort(scores[i])[:top_n]
                reco_items[uid].append([item_ids[idx] for idx in topn_indices])
        return reco_items

    def get_scores(self, feed_dict):
        scores = -tf.matmul(
            self.users, self.item_embeddings, transpose_b=True)
        scores = self.sess.run(scores, feed_dict['model_feed'])
        return scores

    def _create_placeholders(self):
        """
        Build input graph
        :return:
        """
        self.logger.debug('--> Create input placeholders')
        with tf.name_scope('input_data'):
            # boolean to check if training, used for dropout
            self.is_training = tf.compat.v1.placeholder(
                name='is_training',
                dtype=tf.bool,
                shape=())
            self.user_ids = tf.compat.v1.placeholder(name='user_ids',
                                                     dtype=tf.int32,
                                                     shape=[None])
            self.pos_ids = tf.compat.v1.placeholder(name='pos_ids',
                                                    dtype=tf.int32,
                                                    shape=[None])
            # batch of negative sessions, with negative items
            self.neg_ids = tf.compat.v1.placeholder(name='neg_ids',
                                                    dtype=tf.int32,
                                                    shape=[None])
            self.ctx_ids = tf.compat.v1.placeholder(name='last_sess_ids',
                                                    dtype=tf.int32,
                                                    shape=[None, SESSION_LEN])

    def _create_variables(self, reuse=None):
        """
        Build variables
        :return:
        """
        self.logger.debug('--> Create variables')
        self.user_embedding_table = embedding(vocab_size=self.n_users,
                                              embedding_dim=self.embedding_dim,
                                              zero_pad=False,
                                              use_reg=self.use_reg,
                                              l2_reg=self.l2_emb,
                                              scope='user_embedding_table',
                                              initializer='random_normal',
                                              reuse=reuse)
        self.item_embeddings = embedding(vocab_size=self.n_items,
                                         embedding_dim=self.embedding_dim,
                                         zero_pad=False,
                                         use_reg=self.use_reg,
                                         l2_reg=self.l2_emb,
                                         scope='item_embedding_table',
                                         initializer=self.initializer,
                                         reuse=reuse)
        # # user bias
        # self.user_bias_table = tf.compat.v1.get_variable(
        #     name='user_bias_table',
        #     shape=[self.n_users, 1],
        #     initializer=tf.constant_initializer(0.0))
        # # item bias
        # self.item_bias_table = tf.compat.v1.get_variable(
        #     name='item_bias_table',
        #     shape=[self.n_items, 1],
        #     initializer=tf.constant_initializer(0.0))

    def _create_inference(self, name, reuse=None):
        """
        Build inference graph
        :return:
        """
        self.logger.debug('--> Create inference')
        self.users = tf.nn.embedding_lookup(self.user_embedding_table,
                                            self.user_ids,
                                            name='batch_user_embeddings')
        self.pos_items = tf.nn.embedding_lookup(self.item_embeddings,
                                                self.pos_ids,
                                                name='batch_positive_embeddings')
        self.neg_items = tf.nn.embedding_lookup(self.item_embeddings,
                                                self.neg_ids,
                                                name='batch_negative_embeddings')
        # self.user_bias = tf.nn.embedding_lookup(self.user_bias_table,
        #                                         self.user_ids)
        # self.pos_item_bias = tf.nn.embedding_lookup(self.item_bias_table,
        #                                             self.pos_ids)
        # self.neg_item_bias = tf.nn.embedding_lookup(self.item_bias_table,
        #                                             self.neg_ids)

    def _pos_distances(self):
        self.logger.debug('--> Define BPR positive distances')
        # distances = self.pos_item_bias + tf.reduce_sum(self.users * self.pos_items,
        #                                                axis=-1,
        #                                                name='pos_distances')
        distances = tf.reduce_sum(self.users * self.pos_items,
                                  axis=-1, name='pos_distances')
        return distances

    def _neg_distances(self):
        self.logger.debug('--> Define BPR negative distances')
        # distances = self.neg_item_bias + tf.reduce_sum(self.users * self.neg_items,
        #                                                axis=-1,
        #                                                name='neg_distances')
        distances = tf.reduce_sum(self.users * self.neg_items,
                                  axis=-1, name='neg_distances')
        return distances

    def _create_loss(self):
        self.logger.debug('--> Define BPR loss')
        # positive distances
        pos_distances = self._pos_distances()
        # negative distances
        neg_distances = self._neg_distances()
        posneg_score = -tf.math.log(tf.nn.sigmoid(pos_distances - neg_distances))
        # self.loss = tf.reduce_mean(posneg_score) + self._l2_reg()
        self.loss = tf.reduce_mean(posneg_score)

    def _l2_reg(self):
        self.logger.debug('Define BPR L2 Regularization')
        norm = tf.add_n([
            self.user_reg * tf.reduce_sum(tf.multiply(self.users, self.users)),
            # self.user_bias_reg * tf.reduce_sum(tf.multiply(self.user_bias, self.user_bias)),
            self.pos_item_reg * tf.reduce_sum(tf.multiply(self.pos_items, self.pos_items)),
            self.neg_item_reg * tf.reduce_sum(tf.multiply(self.neg_items, self.neg_items)),
            # self.item_bias_reg * tf.reduce_sum(tf.multiply(self.pos_item_bias, self.pos_item_bias)),
            # self.item_bias_reg * tf.reduce_sum(tf.multiply(self.neg_item_bias, self.neg_item_bias))
        ])
        return norm
