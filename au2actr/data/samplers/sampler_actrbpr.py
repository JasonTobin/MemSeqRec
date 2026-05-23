import numpy as np


def test_sample(uid, data, seqlen, nxt_idx, **kwargs):
    user_ids_map = data['user_ids_map']
    activate_actr = kwargs['activate_actr']
    if activate_actr is True:
        track_ids_map = data['track_ids_map']
        user_sessions = data['user_sessions'][uid]
        context_sess = user_sessions[nxt_idx - 1]
        context_tracks = [track_ids_map[tid] - 1 for tid in
                          context_sess['track_ids']]
        out = user_ids_map[uid], nxt_idx, context_tracks, uid
    else:
        out = user_ids_map[uid], nxt_idx, uid
    return out
