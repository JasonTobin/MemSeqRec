def test_sample(uid, data, seqlen, nxt_idx, **kwargs):
    user_ids_map = data['user_ids_map']
    actr = data['test_actr_scores'][uid][nxt_idx]
    out = user_ids_map[uid], nxt_idx, actr, uid
    return out
