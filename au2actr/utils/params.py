import os

from au2actr import Au2ActrError
from au2actr.models import UNTRAINED_MODELS
from au2actr.constants import *


def process_params(params):
    dataset_params = params['dataset']
    training_params = params['training']
    model_params = training_params['model']['params']

    # data specification
    dataset_spec = f'{dataset_params["name"]}_' \
                   f'minsess{dataset_params["min_sessions"]}_' \
                   f'step{dataset_params["samples_step"]}'
    # model specification
    model_spec = gen_model_spec(training_params, model_params)

    n_epoch = training_params["num_epochs"] if 'num_epochs' in training_params else 0
    training_params['model_dir'] = os.path.join(
        training_params['model_dir'],
        dataset_spec,
        f'nepoch{n_epoch}',
        model_spec)
    return training_params, model_params


def gen_model_spec(training_params, model_params):
    model_name = training_params['model']['name']
    if model_name not in UNTRAINED_MODELS:
        # training specifications
        training_spec = _get_training_spec(training_params, model_params)
        # embedding trainable
        embedding_trainable = training_params.get('embedding_trainable', True)
        if embedding_trainable is not True:
            training_spec = f'{training_spec}_frozenin'
        if model_name != 'actr_bpr':
            # transformers blocks
            transformers_spec = _get_transformers_spec(model_params)
            # ACTR
            actr_spec = _get_actr_spec(model_params)
            model_spec = f'{training_spec}_{transformers_spec}_{actr_spec}'
            # OTHERS
            lbda_pos = model_params.get('lbda_pos', 0.)
            if lbda_pos > 0:
                model_spec = f'{model_spec}_lbda-pos{lbda_pos}'
            # lambda multi task
            lbda_task = model_params.get('lbda_task', 0.)
            model_spec = f'{model_spec}_lbda-task{lbda_task}'
            # lambda long-short term
            lbda_ls = model_params.get('lbda_ls', 0.)
            model_spec = f'{model_spec}_lbdals{lbda_ls}'
            # num of favorites for long-term
            num_favs = model_params.get('num_favs', 0)
            model_spec = f'{model_spec}_nfavs{num_favs}'
            # negative sampling
            negsam_strategy = model_params.get('negsam_strategy', NEGSAM_UNIFORM)
            if negsam_strategy == NEGSAM_POP:
                neg_alpha = model_params.get('neg_alpha', 1.0)
                model_spec = f'{model_spec}_neg-pop{neg_alpha}'
            if model_name == 'pisa' or 'au2actr' in model_name:
                lbda_actrpred = model_params.get('lbda_actrpred', 0.)
                actr_dropout = model_params.get('actr_dropout', 0.)
                hidden_dim = model_params.get('hidden_dim', 512)
                au_enc_dropout = model_params.get('au_enc_dropout', 0.)
                lbda_auenc = model_params.get('lbda_auenc', 0.)
                model_spec = f'{model_spec}_lda-actrpred{lbda_actrpred}_' \
                             f'actrdrop{actr_dropout}_lda-auenc{lbda_auenc}_' \
                             f'audrop{au_enc_dropout}_l2enc_hdim{hidden_dim}'
        else:
            model_spec = f'{model_name}_{training_spec}'
            # negative sampling
            negsam_strategy = model_params.get('negsam_strategy',
                                               NEGSAM_UNIFORM)
            if negsam_strategy == NEGSAM_POP:
                neg_alpha = model_params.get('neg_alpha', 1.0)
                model_spec = f'{model_spec}_neg-pop{neg_alpha}'
            else:
                model_spec = f'{model_spec}_neg-uni'
    elif 'top' in model_name or model_name == 'actr':
        model_spec = model_name
    elif 'last' in model_name:
        emb_type = training_params.get('embedding_type', 'svd')
        agg_type = training_params['model']['params'].get('aggregate_type',
                                                          'mean')
        model_spec = f'{model_name}_{emb_type}_{agg_type}-agg'
    else:
        raise Au2ActrError(f'Unknown model name {model_name}')
    return model_spec


def _get_training_spec(training_params, model_params):
    model_name = training_params['model']['name']
    seqlen = model_params.get('seqlen', 0)
    # training specifications
    training_spec = f'lr{training_params["learning_rate"]}_' \
                    f'b{training_params["batch_size"]}_' \
                    f'op-{training_params["optimizer"].lower()}_' \
                    f'slen{seqlen}_' \
                    f'dim{training_params["embedding_dim"]}'
    dropout = model_params.get('dropout_rate', 0)
    if dropout > 0:
        training_spec = f'{training_spec}_drop{dropout}'
    pretrained = model_params.get('pretrained', 'svd')
    training_spec = f'{training_spec}_{pretrained}'
    model_spec = f'{model_name}_{training_spec}_' \
                 f'l2emb{model_params["l2_emb"]}'
    return model_spec


def _get_transformers_spec(model_params):
    input_scale = model_params.get('input_scale', False)
    model_spec = f'nonscale' if input_scale is False else ''
    causality = model_params["sab"].get('causality', True)
    if causality is not True:
        model_spec = f'{model_spec}_noncausal'
    n_negatives = 1 if 'n_negatives' not in model_params \
        else model_params["n_negatives"]
    model_spec = f'{model_spec}_nb{model_params["sab"]["num_blocks"]}_' \
                 f'nh{model_params["sab"]["num_heads"]}_' \
                 f'neg{n_negatives}'
    return model_spec


def _get_actr_spec(model_params):
    model_spec = f'ACTR-bll+{model_params["actr"]["bll"]["type"]}'
    activate_spread = model_params['actr']['spread']['activate']
    if activate_spread:
        model_spec = f'{model_spec}-spr'
        hop = model_params['actr']['spread'].get('hop', 1)
        if hop > 1:
            model_spec = f'{model_spec}+{hop}hop'
        n_last_sess = model_params['actr']['spread'].get(
            'n_last_sess', 1)
        model_spec = f'{model_spec}+last{n_last_sess}sess'
    activate_pm = model_params['actr']['pm']['activate']
    if activate_pm:
        model_spec = f'{model_spec}-pm'
    flatten_actr = model_params.get('flatten_actr', 1)
    if flatten_actr != 1:
        model_spec = f'{model_spec}-flat{flatten_actr}'
    return model_spec
