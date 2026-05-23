from au2actr import Au2ActrError
from au2actr.eval.metrics.ndcg import NDCG
from au2actr.eval.metrics.recall import RECALL
from au2actr.eval.metrics.repr import REPR
from au2actr.eval.metrics.pop import POP


_SUPPORTED_METRICS = {
    'ndcg': NDCG,
    'ndcg_rep': NDCG,
    'ndcg_exp': NDCG,
    'recall': RECALL,
    'recall_rep': RECALL,
    'recall_exp': RECALL,
    'repr': REPR,
    'pop': POP
}


def get_metric(name, k, **kwargs):
    """
    Get metric object from configuration
    :param name:
    :param k:
    :return:
    """
    if name not in _SUPPORTED_METRICS:
        raise Au2ActrError(f'Not supported metric `{name}`. '
                        f'Must one of {list(_SUPPORTED_METRICS.keys())}.')
    if 'rep' in name:
        kwargs['consumption_mode'] = 'rep'
    elif 'exp' in name:
        kwargs['consumption_mode'] = 'exp'
    else:
        kwargs['consumption_mode'] = 'all'
    return _SUPPORTED_METRICS[name](k=k, **kwargs)
