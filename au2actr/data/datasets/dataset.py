import os
import pickle
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from au2actr.constants import NEGSAM_POP, NEGSAM_UNIFORM
from au2actr.data.datasets.actr_weights import load_actr_spread_weights, \
    load_actr_spread_weights_for_posout, load_actr_bll_weights, \
    load_test_actr_spread_weights_for_posout
from au2actr.logging import get_logger


class Dataset:
    """
    Dataset
    """
    def __init__(self, params):
        self.logger = get_logger()
        self.command = params['command']
        cache_params = params['cache']
        self.dataset_params = params['dataset']
        self.min_sessions = self.dataset_params.get('min_sessions', 250)
        self.cache_path = os.path.join(cache_params['path'],
                                       self.dataset_params['name'],
                                       f'min{self.min_sessions}sess')
        self.embedding_type = params['training'].get('embedding_type', 'svd')
        self.embedding_dim = params['training'].get('embedding_dim', 128)
        self.samples_step = self.dataset_params.get('samples_step', 5)
        self.normalize_embedding = params['training'].get(
            'normalize_embedding', False)
        self.model_name = params['training']['model']['name']
        self.model_params = params['training']['model']['params']
        self.seqlen = self.model_params.get('seqlen', 30)
        self.negsam_strategy = self.model_params.get('negsam_strategy',
                                                     NEGSAM_UNIFORM)
        self.neg_alpha = self.model_params.get('neg_alpha', 1.0)
        if 'actr' in self.model_params:
            self.hop = self.model_params['actr']['spread'].get('hop', 1)
            self.n_last_sess = self.model_params['actr']['spread'].get(
                'n_last_sess', 1)
        # train/val/test data split
        data_split = self.dataset_params.get('train_val_test_split',
                                             '-1:10:10')
        n_train_sess, n_val_sess, n_test_sess = data_split.split(':')
        split_idx = int(n_val_sess) + int(n_test_sess)
        # set random seed
        np.random.seed(0)
        # valid and test indices are negative integers
        valid_indices = np.random.choice(range(-split_idx, 0),
                                         size=int(n_val_sess), replace=False)
        test_indices = set(range(-split_idx, 0)) - set(valid_indices)
        self.data_split = {'train': int(n_train_sess),
                           'valid': int(n_val_sess),
                           'test': int(n_test_sess),
                           'valid_indices': valid_indices,
                           'test_indices': test_indices}
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        # noinspection PyTypeChecker
        self.entities_path = os.path.join(
            self.cache_path, f'{self.dataset_params["name"]}_entities.npz')

    def fetch_data(self):
        # load tracks
        loaded_tracks = self._load_tracks()
        track_ids = loaded_tracks['track_ids']
        art_ids = loaded_tracks['art_ids']
        # load user sessions
        user_sessions = self._load_stream_sessions()
        user_ids = np.array(list(user_sessions.keys()))
        # mapping entities (user, track, artist) into internal ids
        track_ids_map = {tid: idx + 1 for idx, tid in enumerate(track_ids)}
        user_ids_map = {uid: idx for idx, uid in enumerate(user_ids)}

        # train sessions indexes
        split_idx = -(self.data_split['valid'] + self.data_split['test'])
        train_session_indexes = self._load_train_session_indexes(
            user_sessions, split_idx=split_idx)

        (train_user_tracks, valid_user_tracks, test_user_tracks,
         mapped_train_user_tracks) = self._load_split_user_tracks(
            user_sessions=user_sessions,
            track_ids_map=track_ids_map,
            data_split=self.data_split)
        # calculate track popularities
        pers_track_pops = self._load_pers_track_pops(
            user_sessions=user_sessions)
        glob_track_pops = self._load_glob_track_pops(
            user_tracks=train_user_tracks)
        norm_track_pops = None
        if self.negsam_strategy == NEGSAM_POP:
            norm_track_pops = self._normalize_item_popularities(
                glob_track_pops, track_ids, self.neg_alpha, item_type='track')

        # repeat consumptions dict
        repeat_consumptions = None
        if self.command == 'eval':
            repeat_consumptions = self._load_repeat_consumptions(
                user_tracks=test_user_tracks, pers_track_pops=pers_track_pops)

        track_bll_weights = None
        track_spread_weights = None
        track_pos_spread_weights = None
        if 'actr' in self.model_params:
            # ACT-R BLL weights
            track_bll_weights = self._fetch_bll_weights(user_sessions,
                                                        train_session_indexes)
            # ACT-R Spread weights
            track_embeddings = loaded_tracks[f'{self.embedding_type}_embeddings']
            (track_spread_weights, train_track_pos_spread_weights,
             test_track_pos_spread_weights) = \
                self._fetch_spread_weights(user_sessions,
                                           train_session_indexes,
                                           track_ids, track_embeddings,
                                           hop=self.hop)
            track_pos_spread_weights = {
                'train': train_track_pos_spread_weights,
                'test': test_track_pos_spread_weights
            }
        # interactions matrix, specific for CF models
        interactions_mat = None
        test_actr_scores = None
        if self.model_name == 'actr' or self.model_name == 'actr_bpr':
            test_actr_scores = self._load_test_actr_scores(
                track_bll_weights, user_sessions, track_ids_map,
                self.data_split['test_indices'], n_items=len(track_ids))
            if self.model_name == 'actr_bpr' and self.command == 'train':
                interactions_mat = self._load_interactions_mat(train_user_tracks,
                                                               user_ids_map,
                                                               track_ids_map)
        data = {
            'user_sessions': user_sessions,
            'user_ids': user_ids,
            'user_ids_map': user_ids_map,
            'track_ids': track_ids,
            'track_ids_map': track_ids_map,
            'art_ids': art_ids,
            'svd_embeddings': loaded_tracks['svd_embeddings'],
            'audio_embeddings': loaded_tracks['audio_embeddings'],
            'data_split': self.data_split,
            'train_session_indexes': train_session_indexes,
            'samples_step': self.samples_step,
            'n_users': len(user_sessions),
            'n_items': len(track_ids),
            'n_artists': len(art_ids),
            'glob_track_popularities': glob_track_pops,
            'pers_track_popularities': pers_track_pops,
            'norm_track_popularities': norm_track_pops,
            'user_tracks': {
                'train': train_user_tracks,
                'valid': valid_user_tracks,
                'test': test_user_tracks,
                'mapped_train': mapped_train_user_tracks
            },
            'repeat_consumptions': repeat_consumptions,
            'track_bll_weights': track_bll_weights,
            'track_spread_weights': track_spread_weights,
            'track_pos_spread_weights': track_pos_spread_weights,
            'interactions_mat': interactions_mat,
            'test_actr_scores': test_actr_scores
        }
        return data

    def _fetch_bll_weights(self, user_sessions, train_session_indexes):
        bll_weights = {
            'train': load_actr_bll_weights(self.cache_path, user_sessions,
                                           seqlen=self.seqlen,
                                           train_session_indexes=train_session_indexes,
                                           bll_type='ts',
                                           data_split=self.data_split,
                                           item_type='track',
                                           mode='train',
                                           samples_step=self.samples_step,
                                           logger=self.logger),
            'valid': load_actr_bll_weights(self.cache_path, user_sessions,
                                           seqlen=self.seqlen,
                                           bll_type='ts',
                                           data_split=self.data_split,
                                           item_type='track',
                                           mode='valid',
                                           samples_step=self.samples_step,
                                           logger=self.logger),
            'test': load_actr_bll_weights(self.cache_path, user_sessions,
                                          seqlen=self.seqlen,
                                          bll_type='ts',
                                          data_split=self.data_split,
                                          item_type='track',
                                          mode='test',
                                          samples_step=self.samples_step,
                                          logger=self.logger)
        }
        return bll_weights

    def _fetch_spread_weights(self, user_sessions, train_session_indexes,
                              track_ids, track_embeddings, hop=1):
        track_spread_weights = load_actr_spread_weights(
            self.cache_path, user_sessions, train_session_indexes,
            data_split=self.data_split,
            item_type='track', track_ids=track_ids,
            logger=self.logger, seqlen=self.seqlen,
            item_embeddings=track_embeddings, hop=hop,
            samples_step=self.samples_step,
            n_last_sess=self.n_last_sess)
        track_pos_spread_weights = load_actr_spread_weights_for_posout(
            self.cache_path, user_sessions, train_session_indexes,
            seqlen=self.seqlen,
            item_type='track', track_ids=track_ids,
            logger=self.logger, item_embeddings=track_embeddings,
            hop=hop, samples_step=self.samples_step,
            n_last_sess=self.n_last_sess)
        test_track_pos_spread_weights = load_test_actr_spread_weights_for_posout(
            self.cache_path, user_sessions,
            seqlen=self.seqlen, data_split=self.data_split,
            item_type='track', track_ids=track_ids,
            logger=self.logger, item_embeddings=track_embeddings,
            hop=hop, samples_step=self.samples_step,
            n_last_sess=self.n_last_sess
        )
        return (track_spread_weights, track_pos_spread_weights,
                test_track_pos_spread_weights)

    def _load_tracks(self):
        raise NotImplementedError('_load_tracks should be '
                                  'implemented in concrete class')

    def _load_stream_sessions(self):
        raise NotImplementedError('_load_stream_sessions should be '
                                  'implemented in concrete class')

    def _load_train_session_indexes(self, user_sessions, split_idx):
        # noinspection PyTypeChecker
        train_session_indexes_path = os.path.join(
            self.cache_path,
            f'train_session_indexes_'
            f'samples-step{self.samples_step}_seqlen{self.seqlen}_'
            f'{self.data_split["valid"]}v_{self.data_split["test"]}t.pkl')
        if not os.path.exists(train_session_indexes_path):
            self.logger.info('Extract training session indexes')
            train_session_indexes = []
            for uid, sessions in user_sessions.items():
                last_idx = len(sessions) + (split_idx - 2)
                first_idx = 0
                # train_session_indexes.append((uid, last_idx))
                if self.samples_step > 0:
                    offsets = list(range(last_idx, first_idx + self.seqlen,
                                         -self.samples_step))
                    for offset in offsets:
                        train_session_indexes.append((uid, offset))
            # noinspection PyTypeChecker
            pickle.dump(train_session_indexes,
                        open(train_session_indexes_path, 'wb'))
        else:
            self.logger.info(f'Load training session indexes from '
                             f'{train_session_indexes_path}')
            train_session_indexes = pickle.load(open(train_session_indexes_path, 'rb'))
        return train_session_indexes

    def _load_split_user_tracks(self, user_sessions, track_ids_map, data_split):
        # noinspection PyTypeChecker
        train_user_tracks_path = os.path.join(
            self.cache_path,
            f'train_user_tracks_'
            f'{data_split["valid"]}v_{data_split["test"]}t.pkl')
        # noinspection PyTypeChecker
        mapped_train_user_tracks_path = os.path.join(
            self.cache_path,
            f'train_mapped_user_tracks_'
            f'{data_split["valid"]}v_{data_split["test"]}t.pkl')
        # noinspection PyTypeChecker
        valid_user_tracks_path = os.path.join(
            self.cache_path,
            f'valid_user_tracks_'
            f'{data_split["valid"]}v_{data_split["test"]}t.pkl')
        # noinspection PyTypeChecker
        test_user_tracks_path = os.path.join(
            self.cache_path,
            f'test_user_tracks_'
            f'{data_split["valid"]}v_{data_split["test"]}t.pkl')
        if not os.path.exists(train_user_tracks_path) or not \
                os.path.exists(valid_user_tracks_path) or not \
                os.path.exists(test_user_tracks_path):
            self.logger.info('Extract user tracks...')
            train_user_tracks = defaultdict(set)
            valid_user_tracks = test_user_tracks = defaultdict(dict)
            split_idx = data_split['valid'] + data_split['test']
            for uid, sessions in user_sessions.items():
                for s in sessions[:-split_idx]:
                    for tid in s['track_ids']:
                        train_user_tracks[uid].add(tid)
                for idx in data_split['valid_indices']:
                    s = sessions[idx]
                    valid_user_tracks[uid][s['session_id']] = set(
                        s['track_ids'])
                for idx in data_split['test_indices']:
                    s = sessions[idx]
                    test_user_tracks[uid][s['session_id']] = set(s['track_ids'])
            # noinspection PyTypeChecker
            pickle.dump(train_user_tracks, open(train_user_tracks_path, 'wb'))
            # noinspection PyTypeChecker
            pickle.dump(valid_user_tracks, open(valid_user_tracks_path, 'wb'))
            # noinspection PyTypeChecker
            pickle.dump(test_user_tracks, open(test_user_tracks_path, 'wb'))
        else:
            self.logger.info('Load user tracks...')
            train_user_tracks = pickle.load(open(train_user_tracks_path, 'rb'))
            valid_user_tracks = pickle.load(open(valid_user_tracks_path, 'rb'))
            test_user_tracks = pickle.load(open(test_user_tracks_path, 'rb'))
        # mapped to internal tracks ids
        if not os.path.exists(mapped_train_user_tracks_path):
            self.logger.info('Extract mapped user tracks...')
            mapped_train_user_tracks = {
                uid: set([track_ids_map[tid] for tid in tracks])
                for uid, tracks in train_user_tracks.items()}
            # noinspection PyTypeChecker
            pickle.dump(mapped_train_user_tracks,
                        open(mapped_train_user_tracks_path, 'wb'))
        else:
            self.logger.info('Load mapped train user tracks...')
            mapped_train_user_tracks = pickle.load(
                open(mapped_train_user_tracks_path, 'rb'))
        return (train_user_tracks, valid_user_tracks, test_user_tracks,
                mapped_train_user_tracks)

    def _load_pers_track_pops(self, user_sessions):
        # noinspection PyTypeChecker
        pers_track_pops_path = os.path.join(
            self.cache_path,
            f'pers_track_popularities_'
            f'{self.data_split["valid"]}v_{self.data_split["test"]}t.pkl')
        if not os.path.exists(pers_track_pops_path):
            pers_track_pops = defaultdict(dict)
            for uid, sessions in tqdm(user_sessions.items(),
                                      desc='Calculate perso track POP...'):
                for ss in sessions[:-self.data_split['valid'] - self.data_split['test']]:
                    for tid in ss['track_ids']:
                        if tid not in pers_track_pops[uid]:
                            pers_track_pops[uid][tid] = 1.0
                        else:
                            pers_track_pops[uid][tid] += 1.0
            for uid, popdict in pers_track_pops.items():
                sum_pop = sum(list(popdict.values()))
                pers_track_pops[uid] = defaultdict(
                    float, {tid: pop / sum_pop for tid, pop in popdict.items()})
            # noinspection PyTypeChecker
            pickle.dump(pers_track_pops, open(pers_track_pops_path, 'wb'))
        else:
            self.logger.info(
                f'Load perso track POP from {pers_track_pops_path}...')
            pers_track_pops = pickle.load(open(pers_track_pops_path, 'rb'))
        return pers_track_pops

    def _load_glob_track_pops(self, user_tracks):
        # noinspection PyTypeChecker
        glob_track_pops_path = os.path.join(
            self.cache_path,
            f'glob_track_popularities_'
            f'{self.data_split["valid"]}v_{self.data_split["test"]}t.pkl')
        if not os.path.exists(glob_track_pops_path):
            self.logger.info('Calculate global track POP...')
            glob_track_pops = defaultdict(float)
            for uid, track_ids in user_tracks.items():
                for tid in track_ids:
                    glob_track_pops[tid] += 1.0
            n_users = len(user_tracks)
            glob_track_pops = {tid: pop / n_users
                               for tid, pop in glob_track_pops.items()}
            # noinspection PyTypeChecker
            pickle.dump(glob_track_pops, open(glob_track_pops_path, 'wb'))
        else:
            self.logger.info(f'Load global track popularities '
                             f'from {glob_track_pops_path}...')
            glob_track_pops = pickle.load(open(glob_track_pops_path, 'rb'))
        return glob_track_pops

    def _load_repeat_consumptions(self, user_tracks, pers_track_pops):
        # noinspection PyTypeChecker
        repeat_consumption_path = os.path.join(
            self.cache_path,
            f'repeat_consumptions_'
            f'{self.data_split["valid"]}v_{self.data_split["test"]}t.pkl')
        if not os.path.exists(repeat_consumption_path):
            self.logger.info('Extract repeat consumptions...')
            repeat_consumptions = defaultdict(list)
            for uid, track_dict in user_tracks.items():
                repeat_tracks = []
                for sid, track_ids in track_dict.items():
                    for tid in track_ids:
                        if tid in pers_track_pops[uid]:
                            repeat_tracks.append(tid)
                    repeat_consumptions[uid].append(set(repeat_tracks))
            # noinspection PyTypeChecker
            pickle.dump(repeat_consumptions, open(repeat_consumption_path, 'wb'))
        else:
            self.logger.info(f'Load repeat consumptions '
                             f'from {repeat_consumption_path}')
            repeat_consumptions = pickle.load(open(repeat_consumption_path, 'rb'))
        return repeat_consumptions

    def _load_interactions_mat(self, user_tracks, user_ids_map,
                               item_ids_map):
        # noinspection PyTypeChecker
        root_path = os.path.join(self.cache_path, self.model_name)
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        # noinspection PyTypeChecker
        interactions_path = os.path.join(
            root_path,
            f'train_user_interactions_{self.data_split["valid"]}v_'
            f'{self.data_split["test"]}t.npz')
        if not os.path.exists(interactions_path):
            interaction_mat = sp.dok_matrix((len(user_ids_map), len(item_ids_map)),
                                            dtype=np.int32)
            for uid, tracks in tqdm(user_tracks.items(),
                                    desc='Extract UI interaction matrix...'):
                for tid in tracks:
                    interaction_mat[user_ids_map[uid], item_ids_map[tid]-1] = 1
            sp.save_npz(interactions_path, interaction_mat.tocsr())
        else:
            self.logger.info(f'Load UI interactions from {interactions_path}')
            interaction_mat = sp.load_npz(interactions_path)
        return interaction_mat

    def _load_test_actr_scores(self, track_bll_weights, user_sessions,
                               track_ids_map, test_indices, n_items):
        # noinspection PyTypeChecker
        root_path = os.path.join(self.cache_path, 'actr')
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        # noinspection PyTypeChecker
        actr_scores_path = os.path.join(
            root_path, f'test_actr_scores_{self.data_split["valid"]}v_'
                       f'{self.data_split["test"]}t.pkl')
        if not os.path.exists(actr_scores_path):
            actr_scores = {}
            # noinspection PyTypeChecker
            adj_matrix_path = os.path.join(
                self.cache_path, f'track_sess-level_adj_matrix_{self.hop}hop_'
                                 f'last{self.n_last_sess}sessctx.npz')
            self.logger.info(f'Load track cooccurences from {adj_matrix_path}')
            # scipy.sparse._csr.csr_matrix
            track_coorc = sp.load_npz(adj_matrix_path)
            for uid, sessions in tqdm(user_sessions.items(),
                                      desc='Extract ACTR-scores...'):
                # ACTR scores on test
                test_user_scores = self._user_actr_scores_indices(
                    uid, test_indices, sessions,
                    track_bll_weights, track_ids_map,
                    track_coorc, n_items, mode='test')
                # convert ACTR scores to sparse matrix format
                actr_scores[uid] = test_user_scores
            # noinspection PyTypeChecker
            pickle.dump(actr_scores, open(actr_scores_path, 'wb'))
        else:
            self.logger.info(f'Load ACTR scores from {actr_scores_path}')
            actr_scores = pickle.load(open(actr_scores_path, 'rb'))
        return actr_scores

    @classmethod
    def _user_actr_scores_indices(cls, uid, ref_indices, sessions,
                                  track_bll_weights, track_ids_map,
                                  track_coorc, n_items, mode='test'):
        user_scores = {}
        for nxt_idx in ref_indices:
            context_sess = sessions[nxt_idx - 1]
            context_tracks = [track_ids_map[tid] - 1 for tid in
                              context_sess['track_ids']]
            bll = np.zeros(n_items, dtype=np.float32)
            for tid, weight in track_bll_weights[mode][uid][nxt_idx].items():
                bll[track_ids_map[tid] - 1] = weight
            spread = np.zeros(n_items, dtype=np.float32)
            for tid in context_tracks:
                row = track_coorc.getrow(tid)
                candidate_tracks = row.indices
                row_spreads = row.data
                for idx, cid in enumerate(candidate_tracks):
                    spread[cid] += row_spreads[idx]
            actr = bll + spread
            user_scores[nxt_idx] = sp.csr_matrix(actr)
        return user_scores

    def _normalize_item_popularities(self, glob_item_pops, item_ids,
                                     alpha=1.0, item_type='track'):
        # noinspection PyTypeChecker
        norm_item_pops_path = os.path.join(
            self.cache_path,
            f'norm_{item_type}_popularities_'
            f'{self.data_split["valid"]}v_{self.data_split["test"]}t_alpha{alpha}.npy')
        if not os.path.exists(norm_item_pops_path):
            self.logger.info(f'Normalize global {item_type} popularities')
            if alpha != 1.0:
                glob_item_pops = {iid: np.power(freq, alpha)
                                   for iid, freq in glob_item_pops.items()}
            total_count = np.sum(list(glob_item_pops.values()))
            item_popularities = np.zeros(len(item_ids), dtype=np.float32)
            for idx in range(len(item_ids)):
                iid = item_ids[idx]
                if iid in glob_item_pops:
                    item_popularities[idx] = glob_item_pops[iid] / total_count
            with open(norm_item_pops_path, 'wb') as f:
                np.save(f, item_popularities)
        else:
            self.logger.info(f'Load normalized {item_type} from '
                             f'{norm_item_pops_path}...')
            with open(norm_item_pops_path, 'rb') as f:
                item_popularities = np.load(f)
        return item_popularities
