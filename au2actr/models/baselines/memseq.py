from collections import defaultdict

import math
import numpy as np
import tensorflow as tf

from au2actr.constants import SESSION_LEN
from au2actr.models.core import embedding, multi_head_attention_blocks
from au2actr.models.model import Model


class MemSeqRec(Model):
    """
    MemSeqRec: Memory-augmented Sequential Recommender with ANN Re-ranking.

    Architecture (per paper "Paper Justification"):
      1. Repeated-padding cold-start: session history padded to seqlen.
      2. Track embeddings: SVD pre-trained, frozen.
      3. Memory strength approximation of ACT-R:
             Memory(i) = sigmoid(alpha * BLL(i) + beta)
         where alpha, beta are learnable scalars that weight recency vs.
         frequency from the pre-computed BLL activation values.
      4. SASRec-style Transformer encodes memory-weighted session history.
      5. Gated fusion of Transformer output with a global BLL-weighted
         memory representation.
      6. Two-stage inference:
         a. Coarse ANN retrieval: inner-product top-K over all items.
         b. MLP re-ranker: corrects ANN approximation error on top-K.
    """

    # ------------------------------------------------------------------
    # learning_rate property — returns current (scheduled) LR for logging
    # ------------------------------------------------------------------
    @property
    def learning_rate(self):
        if hasattr(self, '_lr_tensor'):
            try:
                return float(self.sess.run(self._lr_tensor))
            except Exception:
                pass
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, v):
        self._learning_rate = v

    def __init__(self, sess, params, n_users, n_items, pretrained_embs):
        super().__init__(sess, params, n_users, n_items, pretrained_embs)
        model_params = params['model']['params']
        # ANN retrieval candidate set size
        self.ann_k = model_params.get('ann_k', 500)
        # Re-ranker MLP hidden dimension
        self.hidden_dim = model_params.get('hidden_dim', 256)
        # Weight of re-rank BPR loss vs coarse BPR loss
        self.lambda_rerank = model_params.get('lambda_rerank', 0.5)
        # Coarse BPR vs re-rank BPR trade-off (task weight)
        self.lbda_task = model_params.get('lbda_task', 0.9)
        # num_favs drives the PISA sampler batch format; MemSeqRec does not use
        # the long-term fav items themselves, but the sampler always computes them
        self.num_favs = model_params.get('num_favs', 20)
        # LR schedule: linear warmup then cosine decay
        self._warmup_steps = model_params.get('warmup_steps', 0)
        self._total_steps = model_params.get('total_steps', 22100)
        self._min_lr = params.get('min_lr', 1e-8)

    # ------------------------------------------------------------------
    # Feed dict
    # ------------------------------------------------------------------

    def build_feedict(self, batch, is_training=True):
        """
        Map a raw batch tuple to a TF feed_dict.

        Training batch indices (from sampler_pisa train_sample):
          batch[0]  seq_in       [B, seqlen, SESSION_LEN]
          batch[1]  seq_actr_bll [B, seqlen, SESSION_LEN]
          batch[3]  pos_actr_bll [B, seqlen, SESSION_LEN]
          batch[-2] seq_pos      [B, seqlen, SESSION_LEN]  (pos_ids)
          batch[-1] seq_neg      [B, seqlen, SESSION_LEN]  (neg_ids)

        Inference batch indices (from sampler_pisa test_sample):
          batch[0]  seq_in       [B, seqlen, SESSION_LEN]
          batch[1]  seq_actr_bll [B, seqlen, SESSION_LEN]
        """
        feedict = {
            self.is_training: is_training,
            self.seqin_ids: batch[0],
            self.seqin_actr_bla: batch[1],
        }
        if is_training:
            feedict[self.pos_ids] = batch[-2]
            feedict[self.neg_ids] = batch[-1]
            feedict[self.pos_actr_bla] = batch[3]
        return feedict

    # ------------------------------------------------------------------
    # Inference (predict)
    # ------------------------------------------------------------------

    def predict(self, feed_dict, top_n=50):
        """
        Two-stage prediction:
          1. Run coarse inner-product scoring to get top ann_k candidates.
          2. Re-rank candidates with the trained MLP re-ranker.
          3. Return top_n items per user, sorted by re-rank score.
        """
        item_ids = feed_dict['item_ids']

        topk_indices_np, rerank_scores_np = self.sess.run(
            [self.topk_indices_op, self.rerank_scores_op],
            feed_dict['model_feed'])

        reco_items = defaultdict(list)
        for i, uid in enumerate(feed_dict['user_ids']):
            # argsort descending (higher re-rank score = better)
            sorted_pos = np.argsort(-rerank_scores_np[i])[:top_n]
            final_indices = topk_indices_np[i][sorted_pos]
            reco_items[uid].append([item_ids[idx] for idx in final_indices])
        return reco_items

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _create_placeholders(self):
        super()._create_placeholders()
        # BLL weights for the input history sequence
        self.seqin_actr_bla = tf.compat.v1.placeholder(
            name='seq_in_bll', dtype=tf.float32,
            shape=[None, self.seqlen, SESSION_LEN])
        # BLL weights for the positive (target) session
        self.pos_actr_bla = tf.compat.v1.placeholder(
            name='pos_bll', dtype=tf.float32,
            shape=[None, self.seqlen, SESSION_LEN])

    def _create_variables(self, reuse=None):
        super()._create_variables(reuse=reuse)
        # Learnable memory-strength scaling parameters for ACT-R approximation:
        #   Memory(i) = sigmoid(mem_alpha * BLL(i) + mem_beta)
        # mem_alpha controls recency/frequency sensitivity; mem_beta is a bias.
        self.mem_alpha = tf.compat.v1.get_variable(
            'mem_alpha', shape=(), dtype=tf.float32,
            initializer=tf.compat.v1.constant_initializer(1.0))
        self.mem_beta = tf.compat.v1.get_variable(
            'mem_beta', shape=(), dtype=tf.float32,
            initializer=tf.compat.v1.constant_initializer(0.0))

    def _reranker(self, inputs):
        """
        Shared 2-layer MLP re-ranker.
        inputs : [..., 3 * embedding_dim]
        returns: [..., 1]  (scalar relevance score)
        """
        with tf.compat.v1.variable_scope('reranker',
                                         reuse=tf.compat.v1.AUTO_REUSE):
            h = tf.compat.v1.layers.dense(inputs, self.hidden_dim,
                                          activation=tf.nn.relu, name='fc1')
            h = tf.compat.v1.layers.dropout(
                h, rate=self.dropout_rate,
                training=tf.convert_to_tensor(self.is_training))
            score = tf.compat.v1.layers.dense(h, 1, name='fc2')
        return score

    def _create_train_ops(self):
        """Override to add linear warmup + cosine decay LR schedule."""
        self.logger.debug('--> Define training operators')
        self.step = tf.compat.v1.Variable(0, trainable=False)

        step_f = tf.cast(self.step, tf.float32)
        warmup_f = float(max(self._warmup_steps, 1))
        total_f = float(max(self._total_steps, self._warmup_steps + 1))
        max_lr = float(self._learning_rate)
        min_lr = float(self._min_lr)

        # Linear warmup phase
        warmup_lr = max_lr * step_f / warmup_f
        # Cosine decay phase
        cosine_progress = tf.maximum(step_f - warmup_f, 0.0) / (total_f - warmup_f)
        cosine_lr = min_lr + 0.5 * (max_lr - min_lr) * (
            1.0 + tf.cos(math.pi * cosine_progress))

        self._lr_tensor = tf.where(step_f < warmup_f, warmup_lr, cosine_lr)

        optimizer = tf.compat.v1.train.AdamOptimizer(
            self._lr_tensor, beta2=0.98)
        self.train_ops = [optimizer.minimize(self.loss, global_step=self.step)]

    def _create_inference(self, name, reuse=None):
        self.logger.debug('--> Create inference')
        eps = 1e-8

        with tf.compat.v1.variable_scope(name, reuse=reuse):

            # ----------------------------------------------------------
            # 1. ACT-R memory-strength weights
            #    Memory(i) = sigmoid(alpha * bll_i + beta)
            # ----------------------------------------------------------
            mem_w = tf.nn.sigmoid(
                self.mem_alpha * self.seqin_actr_bla + self.mem_beta)
            # mem_w: [B, seqlen, SESSION_LEN]

            # Normalise across tracks within each session (sum-to-1)
            mem_w_norm = mem_w / (
                tf.reduce_sum(mem_w, axis=-1, keepdims=True) + eps)

            # ----------------------------------------------------------
            # 2. Track embeddings
            # ----------------------------------------------------------
            seq_emb = tf.nn.embedding_lookup(
                self.item_embedding_table, self.seqin_ids)
            # seq_emb: [B, seqlen, SESSION_LEN, D]

            # ----------------------------------------------------------
            # 3. Memory-weighted session aggregation
            # ----------------------------------------------------------
            session_embs = tf.reduce_sum(
                seq_emb * tf.expand_dims(mem_w_norm, axis=-1), axis=2)
            # session_embs: [B, seqlen, D]

            # ----------------------------------------------------------
            # 4. Session-level padding mask
            # ----------------------------------------------------------
            n_tracks = tf.reduce_sum(
                tf.compat.v1.to_float(tf.not_equal(self.seqin_ids, 0)),
                axis=-1)
            # n_tracks: [B, seqlen]
            session_mask = tf.expand_dims(
                tf.compat.v1.to_float(tf.not_equal(n_tracks, 0)), axis=-1)
            # session_mask: [B, seqlen, 1]

            # ----------------------------------------------------------
            # 5. SASRec Transformer
            # ----------------------------------------------------------
            if self.input_scale:
                session_embs = session_embs * (self.embedding_dim ** 0.5)

            # Learnable absolute position embeddings
            position_ids = tf.tile(
                tf.expand_dims(
                    tf.range(tf.shape(self.seqin_ids)[1]), 0),
                [tf.shape(self.seqin_ids)[0], 1])
            pos_emb = tf.nn.embedding_lookup(
                self.position_embedding_table, position_ids)
            session_embs = (session_embs + pos_emb) * session_mask

            seq_out = multi_head_attention_blocks(
                input_seq=session_embs,
                num_blocks=self.num_blocks,
                num_heads=self.num_heads,
                embedding_dim=self.embedding_dim,
                dropout_rate=self.dropout_rate,
                mask=session_mask,
                reuse=reuse,
                causality=self.causality,
                is_training=self.is_training,
                nonscale_inseq=None,
                name=name)
            # seq_out: [B, seqlen, D]

            seq_out_norm = seq_out / (
                tf.expand_dims(
                    tf.norm(seq_out + eps, ord=2, axis=-1), axis=-1))

            # ----------------------------------------------------------
            # 6. Global BLL-weighted memory branch
            #    Flattens all sessions' embeddings and weights by BLL,
            #    then takes a weighted sum as a global user memory vector.
            # ----------------------------------------------------------
            B = tf.shape(self.seqin_ids)[0]
            all_bll = tf.reshape(
                self.seqin_actr_bla, [B, self.seqlen * SESSION_LEN])
            all_bll_w = tf.nn.relu(all_bll) + eps
            all_bll_norm = all_bll_w / tf.reduce_sum(
                all_bll_w, axis=-1, keepdims=True)
            # all_bll_norm: [B, seqlen*SESSION_LEN]

            all_seq_emb = tf.reshape(
                seq_emb, [B, self.seqlen * SESSION_LEN, self.embedding_dim])
            mem_flat = tf.reduce_sum(
                all_seq_emb * tf.expand_dims(all_bll_norm, axis=-1), axis=1)
            # mem_flat: [B, D]
            mem_flat = mem_flat / (
                tf.expand_dims(
                    tf.norm(mem_flat + eps, ord=2, axis=-1), axis=-1))

            # Tile to match the sequence dimension
            mem_rep = tf.tile(
                tf.expand_dims(mem_flat, axis=1), [1, self.seqlen, 1])
            # mem_rep: [B, seqlen, D]

            # ----------------------------------------------------------
            # 7. Gated fusion: blend Transformer output with memory
            # ----------------------------------------------------------
            fused_input = tf.concat([seq_out_norm, mem_rep], axis=-1)
            # fused_input: [B, seqlen, 2D]
            with tf.compat.v1.variable_scope(
                    'fusion_gate', reuse=tf.compat.v1.AUTO_REUSE):
                gate = tf.compat.v1.layers.dense(
                    fused_input, self.embedding_dim,
                    activation=tf.nn.sigmoid, name='gate')
            # gate: [B, seqlen, D]

            fused = gate * seq_out_norm + (1.0 - gate) * mem_rep
            # fused: [B, seqlen, D]
            self.fused_rep = fused / (
                tf.expand_dims(
                    tf.norm(fused + eps, ord=2, axis=-1), axis=-1))

            # ----------------------------------------------------------
            # 8. Positive / negative session representations (for loss)
            # ----------------------------------------------------------
            pos_emb_table = tf.nn.embedding_lookup(
                self.item_embedding_table, self.pos_ids)
            # pos_emb_table: [B, seqlen, SESSION_LEN, D]

            pos_bll_w = tf.nn.relu(self.pos_actr_bla) + eps
            pos_bll_norm = pos_bll_w / (
                tf.reduce_sum(pos_bll_w, axis=-1, keepdims=True) + eps)
            pos_rep_raw = tf.reduce_sum(
                pos_emb_table * tf.expand_dims(pos_bll_norm, axis=-1), axis=2)
            # pos_rep_raw: [B, seqlen, D]
            self.pos_rep = pos_rep_raw / (
                tf.expand_dims(
                    tf.norm(pos_rep_raw + eps, ord=2, axis=-1), axis=-1))

            neg_emb_table = tf.nn.embedding_lookup(
                self.item_embedding_table, self.neg_ids)
            neg_rep_raw = tf.reduce_mean(neg_emb_table, axis=2)
            # neg_rep_raw: [B, seqlen, D]
            self.neg_rep = neg_rep_raw / (
                tf.expand_dims(
                    tf.norm(neg_rep_raw + eps, ord=2, axis=-1), axis=-1))

            # Mask: 1 if the positive session contains at least one track
            self.pos_mask = tf.compat.v1.to_float(
                tf.not_equal(tf.reduce_max(self.pos_ids, axis=-1), 0))
            # pos_mask: [B, seqlen]

            # ----------------------------------------------------------
            # 9. Training re-rank inputs (shared MLP weights)
            #    Use last position of each sequence for re-ranker training.
            # ----------------------------------------------------------
            fused_last = self.fused_rep[:, -1:, :]    # [B, 1, D]
            pos_last   = self.pos_rep[:, -1:, :]       # [B, 1, D]
            neg_last   = self.neg_rep[:, -1:, :]       # [B, 1, D]

            rerank_train_input = tf.concat([
                tf.concat([fused_last, pos_last, fused_last * pos_last],
                           axis=-1),
                tf.concat([fused_last, neg_last, fused_last * neg_last],
                           axis=-1)
            ], axis=0)
            # rerank_train_input: [2B, 1, 3D]

            rerank_train_out = self._reranker(rerank_train_input)
            # rerank_train_out: [2B, 1, 1]
            rerank_train_out = tf.squeeze(rerank_train_out, axis=[1, 2])
            # rerank_train_out: [2B]

            train_B = tf.shape(self.fused_rep)[0]
            self.rerank_train_pos = rerank_train_out[:train_B]
            self.rerank_train_neg = rerank_train_out[train_B:]

            # ----------------------------------------------------------
            # 11. Multi-negative pool re-rank loss (forced positive)
            #     Samples n_pool_neg random negatives and guarantees the
            #     positive is in the pool -> sampled softmax loss.
            # ----------------------------------------------------------
            n_pool_neg = 64
            user_last = self.fused_rep[:, -1, :]       # [B, D]
            rand_neg_idx = tf.random.uniform(
                [train_B, n_pool_neg],
                minval=0, maxval=self.n_items, dtype=tf.int32)
            rand_neg_embs = tf.gather(
                self.item_embeddings, rand_neg_idx)    # [B, n_pool_neg, D]

            pos_pool_emb = tf.expand_dims(
                self.pos_rep[:, -1, :], axis=1)        # [B, 1, D]
            pool_embs = tf.concat(
                [rand_neg_embs, pos_pool_emb], axis=1) # [B, n_pool_neg+1, D]

            user_tiled_pool = tf.tile(
                tf.expand_dims(user_last, 1),
                [1, n_pool_neg + 1, 1])
            pool_rerank_input = tf.concat(
                [user_tiled_pool, pool_embs,
                 user_tiled_pool * pool_embs], axis=-1)
            pool_rerank_scores = tf.squeeze(
                self._reranker(pool_rerank_input), axis=-1)
            # pool_rerank_scores: [B, n_pool_neg+1] — positive at index -1

            pos_pool_score = pool_rerank_scores[:, -1]   # [B]
            log_partition = tf.math.reduce_logsumexp(
                pool_rerank_scores, axis=-1)              # [B]
            self._rerank_pool_loss = -tf.reduce_mean(
                pos_pool_score - log_partition)
            self._rerank_pool_loss = tf.where(
                tf.math.is_nan(self._rerank_pool_loss),
                0.0, self._rerank_pool_loss)

            # ----------------------------------------------------------
            # 11. Inference ops: coarse ANN + MLP re-rank
            #     These are only executed during predict(), not during
            #     the training loss computation.
            # ----------------------------------------------------------
            user_rep = self.fused_rep[:, -1, :]
            # user_rep: [B, D]

            # Coarse inner-product scores against all items
            coarse_scores = tf.matmul(
                user_rep, self.item_embeddings, transpose_b=True)
            # coarse_scores: [B, n_items]

            # Top-K candidates (ANN approximation)
            _top_scores, self.topk_indices_op = tf.math.top_k(
                coarse_scores, k=self.ann_k)
            # topk_indices_op: [B, ann_k]

            # Gather top-K item embeddings
            topk_embs = tf.gather(self.item_embeddings, self.topk_indices_op)
            # topk_embs: [B, ann_k, D]

            # Build re-rank input: [user_rep; item_rep; user_rep * item_rep]
            user_rep_tiled = tf.tile(
                tf.expand_dims(user_rep, axis=1), [1, self.ann_k, 1])
            # user_rep_tiled: [B, ann_k, D]
            rerank_inf_input = tf.concat(
                [user_rep_tiled, topk_embs, user_rep_tiled * topk_embs],
                axis=-1)
            # rerank_inf_input: [B, ann_k, 3D]

            # Re-rank (shared weights with training re-ranker via AUTO_REUSE)
            rerank_inf_out = self._reranker(rerank_inf_input)
            # rerank_inf_out: [B, ann_k, 1]
            self.rerank_scores_op = tf.squeeze(rerank_inf_out, axis=-1)
            # rerank_scores_op: [B, ann_k]

    def _create_loss(self):
        self.logger.debug('--> Create loss')
        eps = 1e-8

        # --------------------------------------------------------------
        # Coarse BPR loss across all sequence positions
        # --------------------------------------------------------------
        pos_score = tf.reduce_sum(
            self.fused_rep * self.pos_rep, axis=-1)   # [B, seqlen]
        neg_score = tf.reduce_sum(
            self.fused_rep * self.neg_rep, axis=-1)   # [B, seqlen]
        coarse_bpr = -tf.reduce_mean(
            tf.math.log(
                tf.nn.sigmoid(pos_score - neg_score) + eps
            ) * self.pos_mask)
        coarse_bpr = tf.where(
            tf.math.is_nan(coarse_bpr), 0.0, coarse_bpr)

        # --------------------------------------------------------------
        # Re-rank BPR loss (last position; uses shared MLP weights)
        # --------------------------------------------------------------
        rerank_bpr = -tf.reduce_mean(
            tf.math.log(
                tf.nn.sigmoid(
                    self.rerank_train_pos - self.rerank_train_neg) + eps))
        rerank_bpr = tf.where(
            tf.math.is_nan(rerank_bpr), 0.0, rerank_bpr)

        # Combined loss: lbda_task * coarse + (1-lbda_task) * pool_rerank
        # pool_rerank is a sampled-softmax loss over 64 random negatives +
        # guaranteed positive, which gives the re-ranker harder negatives
        # and ensures it always sees a positive signal.
        self.loss = (self.lbda_task * coarse_bpr
                     + (1.0 - self.lbda_task) * self._rerank_pool_loss)
