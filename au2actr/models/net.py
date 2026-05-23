from collections import defaultdict
import numpy as np
import tensorflow as tf

from au2actr.constants import SESSION_LEN
from au2actr.models.baselines.pisa import PISA
from au2actr.models.core import embedding


class AU2ACTR(PISA):
    # The original PISA do not work with tracks that user never listened before.
    # This model leverages audio to overcome this drawback with the assumption
    # that, users seem to have the same behaviors with items that are close
    # together in acoustic space.
    # f([sess_emb; audio_emb_curr_item_in_sess]) -> [BLL; spread] seq_pos item
    # f([sess_emb; audio_emb_curr_item_in_sess]) -> [BLL; spread; svd] seq_pos item
    # sess_emb could be frozen or learnable
    def __init__(self, sess, params, n_users, n_items, pretrained_embs):
        super().__init__(sess, params, n_users, n_items,
                                   pretrained_embs)
        model_params = params['model']['params']
        self.lbda_actrpred = model_params.get('lbda_actrpred', 0.)
        self.lbda_auenc = model_params.get('lbda_auenc', 0.)
        self.audio_dim = model_params.get('audio_dim', 1024)
        self.hidden_dim = model_params.get('hidden_dim', 512)
        self.actr_dropout = model_params.get('actr_dropout', 0.)
        self.au_enc_dropout = model_params.get('au_enc_dropout', 0.)
        self.audio_initializer = pretrained_embs['audio_embeddings']

    def predict(self, feed_dict, top_n=50):
        item_ids = feed_dict['item_ids']
        pred_embeddings = self.st_rep[:, -1, :]
        scores = -tf.matmul(
            pred_embeddings, self.item_embeddings, transpose_b=True)
        if self.lbda_ls > 0:
            long_embeddings = self.lt_rep[:, -1, :]
            long_scores = -tf.matmul(
                long_embeddings, self.item_embeddings, transpose_b=True)
            scores += self.lbda_ls * long_scores
        # audio encoder
        au_emb = self._audio_encoder(inputs=self.audio_embeddings,
                                     reuse=tf.compat.v1.AUTO_REUSE)
        # normalize audio encoder output
        au_emb = au_emb / (tf.expand_dims(tf.norm(au_emb, ord=2, axis=-1), -1))
        def fn(t):
            in_rep = tf.concat([
                tf.tile(tf.expand_dims(t, axis=0), [self.n_items, 1]),
                au_emb], axis=-1)
            out = self._actr_predict_net(in_rep, reuse=tf.compat.v1.AUTO_REUSE)
            return out

        pred_actr = tf.compat.v1.map_fn(fn, elems=pred_embeddings)
        actr_scores = -tf.reduce_sum(pred_actr, axis=-1)
        actr_scores = tf.reshape(actr_scores, [-1, self.n_items])
        scores = scores + self.lbda_actrpred * actr_scores
        # scores = scores + self.lbda_actrpred * tf.nn.softmax(actr_scores,
        #                                                      axis=-1)
        scores = self.sess.run(scores, feed_dict['model_feed'])
        reco_items = defaultdict(list)
        for i, (uid, u_scores) in enumerate(zip(feed_dict['user_ids'], scores)):
            topn_indices = np.argsort(scores[i])[:top_n]
            reco_items[uid].append([item_ids[idx] for idx in topn_indices])
        return reco_items

    def _create_variables(self, reuse=None):
        super()._create_variables(reuse=reuse)
        # item embedding
        self.audio_embeddings = embedding(vocab_size=self.n_items,
                                          embedding_dim=self.audio_dim,
                                          zero_pad=False,
                                          use_reg=False,
                                          scope='audio_embedding_table',
                                          initializer=self.audio_initializer,
                                          trainable=False,
                                          reuse=reuse)

    def _create_inference(self, name, reuse=None):
        """
        Build inference graph
        :return:
        """
        super()._create_inference(name=name, reuse=reuse)
        # Audio Encoder
        self.au_enc_hidden = tf.compat.v1.layers.Dense(
            self.hidden_dim, activation=tf.nn.relu,
            kernel_initializer='glorot_uniform')
        self.au_enc_hidden_2 = tf.compat.v1.layers.Dense(
            self.hidden_dim/2, activation=tf.nn.relu,
            kernel_initializer='glorot_uniform')
        self.au_enc_output = tf.compat.v1.layers.Dense(
            self.embedding_dim, activation=tf.nn.relu,
            kernel_initializer='glorot_uniform')
        # ACTR prediction layers
        self.pred_actr_hidden = tf.compat.v1.layers.Dense(
            self.hidden_dim, activation=tf.nn.relu,
            kernel_initializer='glorot_uniform')
        self.pred_actr_hidden_2 = tf.compat.v1.layers.Dense(
            self.hidden_dim/4, activation=tf.nn.relu,
            kernel_initializer='glorot_uniform')
        self.pred_actr_output = tf.compat.v1.layers.Dense(
            2, activation=tf.nn.relu,
            kernel_initializer='glorot_uniform')
        with tf.compat.v1.variable_scope(name, reuse=reuse):
            st_rep = tf.expand_dims(self.st_rep, axis=2)
            st_rep = tf.tile(st_rep, [1, 1, SESSION_LEN, 1])
            audio_pos_ids = self.pos_ids - 1
            au_pos_seq = tf.nn.embedding_lookup(self.audio_embeddings,
                                                     audio_pos_ids)
            # audio encoder
            au_pos_emb = self._audio_encoder(
                tf.reshape(au_pos_seq, shape=[-1, self.audio_dim]), reuse=reuse)
            # normalize
            self.au_pos_emb = au_pos_emb / (
                tf.expand_dims(tf.norm(au_pos_emb, ord=2, axis=-1), -1))
            concat_rep = tf.concat(
                [tf.reshape(st_rep, shape=[-1, self.embedding_dim]),
                 self.au_pos_emb], axis=-1)
            # ACTR prediction
            self.pred_actr = self._actr_predict_net(concat_rep, reuse=reuse)
            output_actr = tf.concat([
                tf.expand_dims(self.pos_actr_bla, axis=-1),
                tf.expand_dims(self.pos_actr_spread, axis=-1),
            ], axis=-1)
            self.output_actr = tf.reshape(output_actr, shape=[-1, 2])

    def _create_loss(self):
        super()._create_loss()
        enc_loss = self._create_audio_loss()
        actr_loss = tf.reduce_mean(tf.reduce_sum(
            tf.math.squared_difference(self.pred_actr, self.output_actr),
            axis=1))
        self.loss = self.loss + self.lbda_actrpred * actr_loss + \
                    self.lbda_auenc * enc_loss

    def _create_audio_loss(self):
        pos_emb = tf.reshape(self.pos_seq, shape=[-1, self.embedding_dim])
        neg_emb = tf.reshape(self.neg_seq, shape=[-1, self.embedding_dim])
        pos_distances = tf.reduce_sum(tf.multiply(pos_emb, self.au_pos_emb),
                                      axis=1)
        neg_distances = tf.reduce_sum(tf.multiply(neg_emb, self.au_pos_emb),
                                      axis=1)
        x_hat = pos_distances - neg_distances
        x_hat = -tf.math.log(tf.nn.sigmoid(x_hat))
        # avoid NaN loss in the case only BLL
        x_hat = tf.where(tf.math.is_nan(x_hat), 0., x_hat)
        loss = tf.reduce_mean(x_hat)
        return loss


    def _audio_encoder(self, inputs, reuse=None, name='audio_encoder'):
        with tf.compat.v1.variable_scope(name, reuse=reuse):
            hiddens = self.au_enc_hidden(inputs)
            if self.au_enc_dropout > 0:
                hiddens = tf.compat.v1.layers.dropout(
                    hiddens,
                    rate=self.au_enc_dropout,
                    training=tf.convert_to_tensor(self.is_training))
            hiddens = self.au_enc_hidden_2(hiddens)
            if self.au_enc_dropout > 0:
                hiddens = tf.compat.v1.layers.dropout(
                    hiddens,
                    rate=self.au_enc_dropout,
                    training=tf.convert_to_tensor(self.is_training))
            outputs = self.au_enc_output(hiddens)
            return outputs

    def _actr_predict_net(self, inputs, reuse=None, name='actr_predictor'):
        with tf.compat.v1.variable_scope(name, reuse=reuse):
            hiddens = self.pred_actr_hidden(inputs)
            if self.actr_dropout > 0:
                hiddens = tf.compat.v1.layers.dropout(
                    hiddens,
                    rate=self.actr_dropout,
                    training=tf.convert_to_tensor(self.is_training))
            hiddens = self.pred_actr_hidden_2(hiddens)
            if self.actr_dropout > 0:
                hiddens = tf.compat.v1.layers.dropout(
                    hiddens,
                    rate=self.actr_dropout,
                    training=tf.convert_to_tensor(self.is_training))
            outputs = self.pred_actr_output(hiddens)
        return outputs

    def _get_sess_representation(self, seq_ids, prev_seq_ids,
                                 emb_table='svd', **kwargs):
        if emb_table == 'svd':
            item_embedding_table = self.item_embedding_table
        else:
            seq_ids = seq_ids - 1
            item_embedding_table = self.audio_embedding_table
        seq_emb = tf.nn.embedding_lookup(item_embedding_table, seq_ids)
        if prev_seq_ids is None:
            first_seq_ids = seq_ids[:, 0:1, :]
            prev_seq_ids = seq_ids[:, 0:self.seqlen - 1, :]
            prev_seq_ids = tf.concat([first_seq_ids, prev_seq_ids], axis=1)
        prev_seq_emb = tf.nn.embedding_lookup(item_embedding_table, prev_seq_ids)
        sim_seq_emb = seq_emb
        seq_ids = tf.reshape(
            seq_ids,
            shape=[tf.shape(seq_emb)[0] * self.seqlen * SESSION_LEN])
        seq_n_elems = tf.compat.v1.to_float(tf.not_equal(seq_ids, 0))
        seq_n_elems = tf.reshape(
            seq_n_elems,
            shape=[tf.shape(seq_emb)[0], self.seqlen, SESSION_LEN])
        seq_n_elems = tf.reduce_sum(seq_n_elems, axis=-1)
        if 'neg_seq' not in kwargs:
            seq_actr_weights = kwargs['seq_actr_weights']
            # partial matching
            if self.pm_activate:
                self.logger.info('----> ACTR-PM Activate')
                sim = tf.matmul(sim_seq_emb,
                                tf.transpose(prev_seq_emb, perm=[0, 1, 3, 2]))
                sim = tf.reduce_sum(sim, axis=-1)
                seq_actr_weights = tf.concat(
                    [seq_actr_weights, tf.expand_dims(sim, axis=-1)], axis=-1)
            else:
                self.logger.info('----> ACTR-PM Off')
            # positive constraints
            seq_actr_weights = tf.nn.relu(seq_actr_weights) + 1e-8
            # sum
            seq_actr_weights = tf.reduce_sum(seq_actr_weights, axis=-1)
            if self.num_active_comp > 1:
                weighted_seq_emb = tf.reduce_sum(seq_emb * tf.expand_dims(
                    seq_actr_weights, axis=-1), axis=2)
            else:
                seq_actr_weights = tf.nn.relu(seq_actr_weights) + 1e-8
                weighted_seq_emb = tf.reduce_sum(seq_emb * seq_actr_weights,
                                                 axis=2)
        else:
            weighted_seq_emb = tf.reduce_mean(seq_emb, axis=2)
        output = weighted_seq_emb, seq_n_elems
        if 'output_item_emb' in kwargs and kwargs['output_item_emb'] is True:
            output = output + (seq_emb,)
        return output
