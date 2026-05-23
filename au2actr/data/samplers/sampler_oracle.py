import numpy as np

from au2actr.constants import SESSION_LEN


def train_sample(uid, nxt_idx, data, seqlen, n_items, user_tracks,
                 **kwargs):
    pass


def test_sample(uid, data, seqlen, nxt_idx, **kwargs):
    track_ids_map = data['track_ids_map']
    user_sessions = data['user_sessions'][uid]
    orac_sess = user_sessions[nxt_idx]
    last_track_ids = [track_ids_map[tid] - 1 for tid in orac_sess['track_ids']]
    out = uid, last_track_ids
    # TODO: Change ACTR with Oracle tracks
    if kwargs['aggregate_type'] == 'actr':
        bll_weights = data['track_bll_weights']['test']
        spread_weights = data['track_spread_weights']['test']
        last_actr_bll = np.zeros(shape=[SESSION_LEN], dtype=np.float32)
        last_actr_bll[:len(last_track_ids)] = [
            bll_weights[uid][nxt_idx][tid] if tid in bll_weights[uid][nxt_idx] \
                else 0.0 for tid in orac_sess['track_ids']]
        last_actr_spread = np.array(spread_weights[uid][nxt_idx][-1])
        out = out + (last_actr_bll, last_actr_spread)
    return out
