from au2actr import Au2ActrError
from au2actr.models import UNTRAINED_MODELS


def one_train_sample(uid, nxt_idx, data, seqlen, n_items, user_tracks,
                     **kwargs):
    model_name = kwargs['model_name']
    if 'pisa' in model_name or 'au2actr' in model_name or model_name == 'memseq':
        from au2actr.data.samplers.sampler_pisa import train_sample
    else:
        raise Au2ActrError(f'Not support train sampler for '
                        f'{kwargs["model_name"]} model')
    return train_sample(uid, nxt_idx, data, seqlen, n_items, user_tracks,
                        **kwargs)


def one_valid_sample(uid, data, seqlen, nxt_idx, n_items, user_tracks,
                     **kwargs):
    model_name = kwargs['model_name']
    if 'pisa' in model_name or 'au2actr' in model_name or model_name == 'memseq':
        from au2actr.data.samplers.sampler_pisa import valid_sample
    else:
        raise Au2ActrError(f'Not support test sampler for '
                        f'{kwargs["model_name"]} model')
    return valid_sample(uid, data, seqlen, nxt_idx, n_items, user_tracks,
                        **kwargs)


def one_test_sample(uid, data, seqlen, nxt_idx, **kwargs):
    model_name = kwargs['model_name']
    if 'pisa' in model_name or 'au2actr' in model_name or model_name == 'memseq':
        from au2actr.data.samplers.sampler_pisa import test_sample
    elif 'top' in model_name:
        from au2actr.data.samplers.sampler_top import test_sample
    elif 'last' in model_name:
        from au2actr.data.samplers.sampler_last import test_sample
    elif model_name == 'actr':
        from au2actr.data.samplers.sampler_actr import test_sample
    elif model_name == 'actr_bpr':
        from au2actr.data.samplers.sampler_actrbpr import test_sample
    else:
        raise Au2ActrError(f'Not support test sampler for '
                        f'{kwargs["model_name"]} model')
    return test_sample(uid, data, seqlen, nxt_idx, **kwargs)
