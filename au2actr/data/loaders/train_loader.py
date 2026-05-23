import numpy as np
import scipy.sparse as sp

from au2actr.data.loaders.loader import DataLoader
from au2actr.data.samplers import one_train_sample


class TrainDataLoader(DataLoader):
    def __init__(self, data, n_users, n_items,
                 batch_size, seqlen, random_seed=2022, **kwargs):
        super(TrainDataLoader, self).__init__(data, n_users, n_items,
                                              batch_size, seqlen, random_seed,
                                              **kwargs)
        self.train_user_tracks = data['user_tracks']['mapped_train']
        self.track_ids_map = data['track_ids_map']
        self.interaction_indexes = \
            kwargs['train_session_indexes']
        self.rng.shuffle(self.interaction_indexes)
        self.n_batches = int(len(self.interaction_indexes) / batch_size)
        if self.n_batches * self.batch_size < len(self.interaction_indexes):
            self.n_batches += 1
        self.track_popularities = data['norm_track_popularities']

    def _batch_sampling(self, batch_index):
        batch_interaction_indexes = self.interaction_indexes[
                                    batch_index * self.batch_size:
                                    (batch_index + 1) * self.batch_size]
        return self._batch_sampling_seq(batch_interaction_indexes)

    def _batch_sampling_seq(self, batch_interaction_indexes):
        """
        Batch sampling
        :param batch_interaction_indexes:
        :return:
        """
        output = []
        for uid, idx in batch_interaction_indexes:
            user_tracks = self.train_user_tracks[uid]
            one_sample = one_train_sample(uid, idx, self.data, self.seqlen,
                                          self.n_items, user_tracks=user_tracks,
                                          norm_track_popularities=self.track_popularities,
                                          **self.kwargs)
            output.append(one_sample)
        return list(zip(*output))


class CFTrainDataLoader(DataLoader):
    def __init__(self, data, n_users, n_items,
                 batch_size, seqlen, random_seed=2025, **kwargs):
        super(CFTrainDataLoader, self).__init__(data, n_users, n_items,
                                                batch_size, seqlen, random_seed,
                                                **kwargs)
        self.user_tracks = data['user_tracks']['train']
        self.track_ids_map = data['track_ids_map']
        self.interactions = sp.lil_matrix(data['interactions_mat'])
        # positive user item pairs
        self.user_item_pairs = np.asarray(self.interactions.nonzero()).T
        self.rng.shuffle(self.user_item_pairs)
        self.n_batches = int(len(self.user_item_pairs) / batch_size)
        if self.n_batches * self.batch_size < len(self.user_item_pairs):
            self.n_batches += 1

    def _batch_sampling(self, batch_index):
        user_pos_items_pairs = self.user_item_pairs[
                                    batch_index * self.batch_size:
                                    (batch_index + 1) * self.batch_size, :]
        # generate batch users
        user_ids = np.array([uid for uid, _ in user_pos_items_pairs])

        # generate batch positives
        pos_ids = np.array([iid for _, iid in user_pos_items_pairs])

        # generate batch negatives
        candidates = np.arange(self.n_items)
        neg_ids = np.random.choice(candidates, size=len(pos_ids))
        for i, uid, neg in zip(range(len(user_ids)), user_ids, neg_ids):
            user_tracks = set([self.track_ids_map[tid]
                               for tid in self.user_tracks[uid]])
            while neg in user_tracks:
                neg_ids[i] = neg = np.random.choice(candidates)
        return user_ids, pos_ids, neg_ids
