from collections import defaultdict
import numpy as np
import scipy.special as ss

from au2actr.models.model import Model


class ACTR(Model):
    def __init__(self, sess, params, n_users, n_items, pretrained_embs):
        super().__init__(sess, params, n_users, n_items, pretrained_embs)
        self.actr_scores = None
        self.user_tracks = None
        self.item_ids_map = None

    def build_feedict(self, batch, is_training=True):
        feedict = {'nxt_indices': batch[1]}
        return feedict

    def build_graph(self, name=None):
        self.logger.info('No need to build graph!')

    def predict(self, feed_dict, top_n=50):
        reco_items = defaultdict(list)
        nxt_indices = feed_dict['model_feed']['nxt_indices']
        for uid, nxt_idx in zip(feed_dict['user_ids'], nxt_indices):
            user_tracks = self.user_tracks[uid]
            actr_scores = self.actr_scores[uid][nxt_idx].toarray()[0]
            # get nonzero entries
            nonzeros = np.exp(actr_scores[np.nonzero(actr_scores)])
            denom = np.sum(nonzeros)
            fav_scores = {tid: np.exp(actr_scores[self.item_ids_map[tid]])/denom
                          for tid in user_tracks}
            top_entries = sorted(fav_scores.items(), key=lambda x: x[1],
                                 reverse=True)[:top_n]
            reco_items[uid].append([e[0] for e in top_entries])
        return reco_items

    def set_user_tracks(self, user_tracks):
        self.user_tracks = user_tracks

    def set_item_ids_map(self, item_ids_map):
        self.item_ids_map = item_ids_map

    def _create_inference(self, name, reuse=None):
        pass

    def _create_loss(self):
        pass
