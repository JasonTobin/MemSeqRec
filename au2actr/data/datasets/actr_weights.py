import os
import pickle
from scipy.special import softmax

from au2actr.constants import SESSION_LEN
from au2actr.utils.adjmat import *


def load_actr_bll_weights(cache_path, user_sessions, data_split, seqlen,
                          train_session_indexes=None,
                          item_type='track',
                          mode='train',
                          decay=0.5, **kwargs):
    logger = kwargs['logger']
    samples_step = kwargs['samples_step']
    bll_weights_path = os.path.join(cache_path,
                                    f'{mode}_{item_type}_'
                                    f'blldecay{decay}_weights_'
                                    f'seqlen{seqlen}_step{samples_step}.pkl')
    if not os.path.exists(bll_weights_path):
        logger.info(f'Build {mode} BLL weights dictionary...')
        # reference time for ACT-R BLL calculation
        t_refs = _load_t_refs(user_sessions, data_split,
                              train_session_indexes=train_session_indexes,
                              mode=mode)
        track_art_map = None if 'track_art_map' not in kwargs \
            else kwargs['track_art_map']
        if mode == 'train':
            user_item_seq_hist = _load_train_user_item_seq_hist(
                cache_path, user_sessions, data_split,
                seqlen=seqlen,
                samples_step=samples_step,
                train_session_indexes=train_session_indexes,
                logger=logger,
                item_type=item_type,
                track_art_map=track_art_map)
        else:
            user_item_seq_hist = _load_test_user_item_seq_hist(
                cache_path,
                user_sessions,
                data_split,
                mode=mode,
                logger=logger,
                item_type=item_type,
                track_art_map=track_art_map,
                seqlen=seqlen,
                samples_step=samples_step)
        bll_weights = {}
        for uid, item_seq_hist in tqdm(user_item_seq_hist.items(),
                                       desc=f'Calculate {mode}-{item_type} BLL weights...'):
            user_bll_weights = {}
            for nxt_idx, hist in item_seq_hist.items():
                user_nxtidx_bll_weights = {
                    iid: _calculate_bll_weights(seq_hist, t_refs[uid][nxt_idx],
                                                decay)
                    for iid, seq_hist in hist.items()}
                # normalize
                k = list(user_nxtidx_bll_weights.keys())
                v = np.array(list(user_nxtidx_bll_weights.values()))
                v = softmax(v)
                user_bll_weights[nxt_idx] = dict(zip(k, v))
            bll_weights[uid] = user_bll_weights
        # noinspection PyTypeChecker
        pickle.dump(bll_weights, open(bll_weights_path, 'wb'))
    else:
        logger.info(f'Load {mode}-{item_type} BLL weights '
                    f'step {samples_step} dictionary...')
        bll_weights = pickle.load(open(bll_weights_path, 'rb'))
    return bll_weights


def load_actr_spread_weights(cache_path, user_sessions, train_session_indexes,
                             data_split, item_type='track', **kwargs):
    logger = kwargs['logger']
    hop = kwargs['hop']
    samples_step = kwargs['samples_step']
    seqlen = kwargs['seqlen']
    track_ids = kwargs['track_ids']
    n_last_sess = kwargs['n_last_sess']
    spread_weights_path = os.path.join(cache_path,
                                       f'{item_type}_spread_weights_'
                                       f'{hop}hop_seqlen{seqlen}_'
                                       f'step{samples_step}_'
                                       f'nlastsess{n_last_sess}.pkl')
    if not os.path.exists(spread_weights_path):
        logger.info(f'Load {item_type} {hop}-hop adjacency matrices...')
        track_art_map = None if 'track_art_map' not in kwargs else kwargs[
            'track_art_map']
        adj_matrix_path = os.path.join(cache_path,
                                       f'{item_type}_sess-level_'
                                       f'adj_matrix_{hop}hop_'
                                       f'last{n_last_sess}sessctx.npz')
        if os.path.exists(adj_matrix_path):
            adj_matrix = sp.load_npz(adj_matrix_path).toarray()
        else:
            logger.info(f'Build {item_type} session-'
                        f'level adjacency matrix...')
            adj_matrix = _build_adj_matrix(user_sessions=user_sessions,
                                           level='sess',
                                           data_split=data_split,
                                           track_art_map=track_art_map,
                                           track_ids=track_ids,
                                           n_last_sess=n_last_sess,
                                           hop=hop)
            sp.save_npz(adj_matrix_path, adj_matrix)
            logger.info(f'Save {item_type} {hop}-hop '
                        f'adjacency matrix to {adj_matrix_path}...')

        spread_weights = {'train': {}, 'test': {}}
        # Train corpus
        for uid, nxt_idx in tqdm(
                train_session_indexes,
                desc=f'Build train spread weights from {item_type} '
                     f'{hop}-hop adj matrix...'):
            if uid not in spread_weights['train']:
                spread_weights['train'][uid] = {
                    nxt_idx: _build_spread_weights_for_user_index(
                        nxt_idx, user_sessions[uid], item_type,
                        adj_matrix, **kwargs)}
            else:
                spread_weights['train'][uid][nxt_idx] = \
                    _build_spread_weights_for_user_index(
                        nxt_idx, user_sessions[uid], item_type,
                        adj_matrix, **kwargs)
        # Val + Test corpus
        for uid, sessions in tqdm(
                user_sessions.items(),
                desc=f'Build test spread weights from {item_type} '
                     f'{hop}-hop adj matrix...'):
            for nxt_idx in range(-(data_split['valid'] + data_split['test']), 0):
                if uid not in spread_weights['test']:
                    spread_weights['test'][uid] = {
                        nxt_idx: _build_spread_weights_for_user_index(
                            nxt_idx, sessions, item_type, adj_matrix, **kwargs)}
                else:
                    spread_weights['test'][uid][nxt_idx] = \
                        _build_spread_weights_for_user_index(
                            nxt_idx, sessions, item_type, adj_matrix, **kwargs)
        # noinspection PyTypeChecker
        pickle.dump(spread_weights, open(spread_weights_path, 'wb'))
    else:
        logger.info(f'Load {item_type} spread weights '
                    f'from {spread_weights_path}...')
        spread_weights = pickle.load(open(spread_weights_path, 'rb'))
    return spread_weights


def load_actr_spread_weights_for_posout(cache_path, user_sessions,
                                        train_session_indexes,
                                        item_type='track', **kwargs):
    """This is a patch for the function _load_actr_spread_weights """
    logger = kwargs['logger']
    hop = kwargs['hop']
    samples_step = kwargs['samples_step']
    seqlen = kwargs['seqlen']
    n_last_sess = kwargs['n_last_sess']
    spread_weights_path = os.path.join(cache_path,
                                       f'pos_{item_type}_spread_weights_'
                                       f'{hop}hop_seqlen{seqlen}_'
                                       f'step{samples_step}_'
                                       f'nlastsess{n_last_sess}.pkl')
    if not os.path.exists(spread_weights_path):
        logger.info(f'Load {item_type} adjacency matrices...')
        adj_matrix_path = os.path.join(cache_path,
                                       f'{item_type}_sess-level_'
                                       f'adj_matrix_{hop}hop_'
                                       f'last{n_last_sess}sessctx.npz')
        adj_matrix = sp.load_npz(adj_matrix_path).toarray()
        spread_weights = {}
        for uid, nxt_idx in tqdm(
                train_session_indexes,
                desc=f'Build train POS-OUTPUT spread weights '
                     f'from {item_type} adj matrix...'):
            if uid not in spread_weights:
                spread_weights[uid] = {
                    nxt_idx: _build_spread_weights_for_user_next_index(
                        nxt_idx, user_sessions[uid], item_type, adj_matrix, **kwargs)}
            else:
                spread_weights[uid][nxt_idx] = \
                    _build_spread_weights_for_user_next_index(
                        nxt_idx, user_sessions[uid], item_type, adj_matrix, **kwargs)
        # noinspection PyTypeChecker
        pickle.dump(spread_weights, open(spread_weights_path, 'wb'))
    else:
        logger.info(f'Load POS OUTPUT {item_type} spread weights '
                    f'from {spread_weights_path}...')
        spread_weights = pickle.load(open(spread_weights_path, 'rb'))
    return spread_weights


def load_test_actr_spread_weights_for_posout(cache_path, user_sessions,
                                             item_type='track',
                                             **kwargs):
    """This is a patch for the function _load_actr_spread_weights """
    logger = kwargs['logger']
    hop = kwargs['hop']
    samples_step = kwargs['samples_step']
    seqlen = kwargs['seqlen']
    n_last_sess = kwargs['n_last_sess']
    data_split = kwargs['data_split']
    spread_weights_path = os.path.join(cache_path,
                                       f'test_pos_{item_type}_spread_weights_'
                                       f'{hop}hop_seqlen{seqlen}_'
                                       f'step{samples_step}_'
                                       f'nlastsess{n_last_sess}.pkl')
    if not os.path.exists(spread_weights_path):
        logger.info(f'Load {item_type} adjacency matrices...')
        adj_matrix_path = os.path.join(cache_path,
                                       f'{item_type}_sess-level_'
                                       f'adj_matrix_{hop}hop_'
                                       f'last{n_last_sess}sessctx.npz')
        adj_matrix = sp.load_npz(adj_matrix_path).toarray()
        spread_weights = {}
        for uid, sessions in tqdm(
                user_sessions.items(),
                desc=f'Build TEST POS-OUTPUT spread weights '
                     f'from {item_type} adj matrix...'):
            for nxt_idx in range(-(data_split['valid'] + data_split['test']), 0):
                if uid not in spread_weights:
                    spread_weights[uid] = {
                        nxt_idx: _build_spread_weights_for_user_next_index(
                            nxt_idx, user_sessions[uid], item_type, adj_matrix,
                            **kwargs)}
                else:
                    spread_weights[uid][nxt_idx] = \
                        _build_spread_weights_for_user_next_index(
                            nxt_idx, user_sessions[uid], item_type, adj_matrix,
                            **kwargs)
        # noinspection PyTypeChecker
        pickle.dump(spread_weights, open(spread_weights_path, 'wb'))
    else:
        logger.info(f'Load TEST POS OUTPUT {item_type} spread weights '
                    f'from {spread_weights_path}...')
        spread_weights = pickle.load(open(spread_weights_path, 'rb'))
    return spread_weights


def _load_t_refs(user_sessions, data_split, train_session_indexes=None,
                 mode='train'):
    # Reference time is the time of each begining test session
    t_refs = defaultdict(dict)
    if mode == 'train':
        for uid, nxt_idx in tqdm(train_session_indexes,
                                 desc='Extract train t_ref...'):
            t_refs[uid][nxt_idx] = user_sessions[uid][nxt_idx]['context']['ts']
    else:
        ref_indices = data_split[f'{mode}_indices']
        for uid, sessions in tqdm(user_sessions.items(),
                                  desc='Extract test t_ref...'):
            for ref_idx in ref_indices:
                t_refs[uid][ref_idx] = sessions[ref_idx]['context']['ts']
    return t_refs


def _calculate_bll_weights(seq_hist, t_ref, d):
    weight = 0.
    for ts in seq_hist:
        weight += np.power(t_ref - ts, -d)
    return np.log(weight)


def _load_train_user_item_seq_hist(cache_path, user_sessions, data_split,
                                   seqlen, train_session_indexes,
                                   item_type='track', **kwargs):
    logger = kwargs['logger']
    user_item_interactions_path = os.path.join(
        cache_path,
        f'train_user_{item_type}_interactions_dict_'
        f'{data_split["valid"]}v_{data_split["test"]}t_seqlen{seqlen}_'
        f'step{kwargs["samples_step"]}.pkl')
    if not os.path.exists(user_item_interactions_path):
        logger.info(f'Extract train user-{item_type} sequential '
                    f'history for ACT-R...')
        output = defaultdict(dict)
        # calculate for each nxt_idx
        for uid, nxt_idx in tqdm(train_session_indexes,
                                 desc=f'Build user-{item_type} history...'):
            sessions = user_sessions[uid]
            for idx, ss in enumerate(sessions[:nxt_idx]):
                for tid in ss['track_ids']:
                    rid = tid if item_type == 'track' else kwargs['track_art_map'][tid]
                    if uid not in output or nxt_idx not in output[uid]:
                        output[uid][nxt_idx] = {}
                    if rid not in output[uid][nxt_idx]:
                        output[uid][nxt_idx][rid] = [ss['context']['ts']]
                    else:
                        output[uid][nxt_idx][rid].append(ss['context']['ts'])
        # noinspection PyTypeChecker
        pickle.dump(output, open(user_item_interactions_path, 'wb'))
    else:
        logger.info(f'Load train user-{item_type} sequential history for ACT-R from '
                    f'{user_item_interactions_path}')
        output = pickle.load(open(user_item_interactions_path, 'rb'))
    return output


def _load_test_user_item_seq_hist(cache_path, user_sessions, data_split,
                                  mode='test',
                                  item_type='track', **kwargs):
    logger = kwargs['logger']
    ref_indices = data_split[f'{mode}_indices']
    user_item_interactions_path = os.path.join(
        cache_path,
        f'{mode}_user_{item_type}_interactions_dict_'
        f'{data_split["valid"]}v_{data_split["test"]}t_'
        f'seqlen{kwargs["seqlen"]}_step{kwargs["samples_step"]}.pkl')
    if not os.path.exists(user_item_interactions_path):
        logger.info(f'Extract {mode} user-{item_type} sequential '
                    f'history for ACT-R...')
        output = defaultdict(dict)
        for uid, sessions in tqdm(user_sessions.items(),
                                  desc=f'Build user-{item_type} history...'):
            for nxt_idx in ref_indices:
                for idx, ss in enumerate(sessions[:nxt_idx]):
                    for tid in ss['track_ids']:
                        rid = tid if item_type == 'track' else \
                        kwargs['track_art_map'][tid]
                        if uid not in output or nxt_idx not in output[uid]:
                            output[uid][nxt_idx] = {}
                        if rid not in output[uid][nxt_idx]:
                            output[uid][nxt_idx][rid] = [ss['context']['ts']]
                        else:
                            output[uid][nxt_idx][rid].append(
                                ss['context']['ts'])
        # noinspection PyTypeChecker
        pickle.dump(output, open(user_item_interactions_path, 'wb'))
    else:
        logger.info(f'Load {mode} user-{item_type} sequential history for ACT-R from '
                    f'{user_item_interactions_path}')
        output = pickle.load(open(user_item_interactions_path, 'rb'))
    return output


def _build_spread_weights_for_user_index(nxt_idx, user_sessions, item_type,
                                         adj_matrix, **kwargs):
    seqlen = kwargs['seqlen']
    sessions = user_sessions[nxt_idx - seqlen:nxt_idx]
    prev_sessions = user_sessions[nxt_idx - seqlen - 1:nxt_idx - 1]
    item_ids_list = _extract_item_ids(sessions, item_type, **kwargs)
    prev_item_ids_list = _extract_item_ids(prev_sessions, item_type, **kwargs)
    return _build_spread_weights(item_ids_list, prev_item_ids_list, adj_matrix)


def _build_spread_weights_for_user_next_index(nxt_idx, user_sessions,
                                              item_type, adj_matrix, **kwargs):
    sessions = [user_sessions[nxt_idx]]
    prev_sessions = [user_sessions[nxt_idx - 1]]
    item_ids_list = _extract_item_ids(sessions, item_type, **kwargs)
    prev_item_ids_list = _extract_item_ids(prev_sessions, item_type, **kwargs)
    return _build_spread_weights(item_ids_list, prev_item_ids_list, adj_matrix)


def _extract_item_ids(sessions, item_type='track', **kwargs):
    output = []
    if item_type == 'track':
        track_ids_map = {tid: idx for idx, tid in enumerate(kwargs['track_ids'])}
        for ss in sessions:
            track_ids = [track_ids_map[tid] for tid in ss['track_ids']]
            output.append(track_ids)
    else:
        art_ids_map = {aid: idx for idx, aid in enumerate(kwargs['art_ids'])}
        track_art_map = kwargs['track_art_map']
        for ss in sessions:
            art_ids = [art_ids_map[track_art_map[tid]] for tid in ss['track_ids']]
            output.append(art_ids)
    return output


def _build_spread_weights(item_ids_list, prev_item_id_list, adj_matrix):
    spread_weights = []
    for curr_list_idx, item_ids in enumerate(item_ids_list):
        weights = np.zeros(SESSION_LEN, dtype=np.float32)
        for idx, iid_1 in enumerate(item_ids):
            w = 0.
            for iid_2 in prev_item_id_list[curr_list_idx]:
                w += adj_matrix[iid_1, iid_2]
            weights[idx] = w
        spread_weights.append(weights)
    return np.array(spread_weights, dtype=np.float32)


def _build_spread_weights_with_item_embs(item_ids_list, adj_matrix,
                                         item_embeddings):
    spread_weights = []
    embeddings = np.array(list(item_embeddings.values()))
    for item_ids in item_ids_list:
        weights = np.zeros(SESSION_LEN, dtype=np.float32)
        for idx, iid_1 in enumerate(item_ids):
            w = 0.
            for iid_2 in item_ids:
                if iid_2 != iid_1:
                    sim = np.dot(embeddings[iid_1, :],
                                 embeddings[iid_2, :])
                    w += sim * adj_matrix[iid_1, iid_2]
            weights[idx] = w
        spread_weights.append(weights)
    return np.array(spread_weights, dtype=np.float32)


def _build_adj_matrix(user_sessions, level,
                      track_ids, data_split, n_last_sess,
                      track_art_map=None, hop=1):
    adj_matrix = build_sparse_adjacency_matrix(user_sessions=user_sessions,
                                               level=level,
                                               track_ids=track_ids,
                                               data_split=data_split,
                                               track_art_map=track_art_map,
                                               n_last_sess=n_last_sess)
    adj_matrix = normalize_adj(adj_matrix)
    adj_matrix = adj_matrix.toarray()
    mul = adj_matrix
    w_mul = adj_matrix
    coeff = 1.0
    if hop > 1:
        for w in range(1, hop):
            coeff *= 0.85
            w_mul *= adj_matrix
            w_mul = remove_diag(w_mul)
            w_adj_matrix = normalize_adj(w_mul)
            mul += coeff * w_adj_matrix
    adj_matrix = mul
    adj_matrix = sp.csr_matrix(adj_matrix)
    return adj_matrix
