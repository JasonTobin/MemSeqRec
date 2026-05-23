import os
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import defaultdict

from au2actr.logging import get_logger
from au2actr.utils.params import process_params, gen_model_spec
from au2actr.data.datasets import dataset_factory
from au2actr.models import ModelFactory, UNTRAINED_MODELS
from au2actr.data.loaders import dataloader_factory
from au2actr.eval.evaluator import Evaluator
# from au2actr.constants import *


def entrypoint(params):
    """ Command entrypoint
    :param params: Deserialized JSON configuration file
                   provided in CLI args.
    """
    logger = get_logger()
    tf.compat.v1.disable_eager_execution()
    # process params
    training_params, model_params = process_params(params)
    dataset_params = params['dataset']
    training_params = params['training']
    model_name = training_params['model']['name']
    cache_path = params['cache']['path']
    eval_params = params['eval']
    min_sessions = dataset_params.get('min_sessions', 250)
    cache_path = os.path.join(cache_path,
                              dataset_params['name'],
                              f'min{min_sessions}sess')
    embedding_dim = training_params.get('embedding_dim', 128)
    logger.info(training_params['model_dir'])
    params['command'] = 'eval'

    # load datasets
    data = dataset_factory(params=params)
    logger.info(f'Number of users: {data["n_users"]}')
    logger.info(f'Number of items: {data["n_items"]}')

    pretrained_embs = {
        'item_ids': np.array(list(data['svd_embeddings'].keys())),
        'svd_embeddings': np.array(list(data['svd_embeddings'].values())),
        'audio_embeddings': np.array(list(data['audio_embeddings'].values()))
    }

    # reco result path
    # noinspection PyTypeChecker
    reco_parent_path = os.path.join(
        cache_path,
        f'result_{data["data_split"]["valid"]}v_{data["data_split"]["test"]}t')
    # noinspection PyTypeChecker
    reco_parent_path = os.path.join(reco_parent_path, 'norm') \
        if training_params['normalize_embedding'] is True \
        else os.path.join(reco_parent_path, 'unnorm')
    if not os.path.exists(reco_parent_path):
        os.makedirs(reco_parent_path, exist_ok=True)

    # start model eval
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    with tf.compat.v1.Session(config=sess_config) as sess:
        model = ModelFactory.generate_model(sess=sess,
                                            params=training_params,
                                            n_users=data['n_users'],
                                            n_items=data['n_items'],
                                            command='eval',
                                            pretrained_embs=pretrained_embs)
        aggregate_type = None
        if 'last' in model_name:
            sess.run(tf.compat.v1.global_variables_initializer())
            aggregate_type = model_params.get('aggregate_type', 'mean')
        # generate users for test
        scores = defaultdict(list)
        batch_size = eval_params.get('batch_size', 256)
        model.batch_size = batch_size
        num_scored_users = eval_params.get('n_users')
        seqlen = model_params.get('seqlen', 20)
        num_favs = model_params.get('num_favs', 0)
        eval_mode = eval_params.get('mode', 'mean')
        mode = 'test'
        user_tracks = data['user_tracks']['train']
        random_seeds = eval_params.get('random_seeds')
        if model_name == 'actr':
            model.set_user_tracks(data['user_tracks']['train'])
            model.set_item_ids_map(
                {iid: idx for idx, iid in enumerate(data['track_ids'])})
        if model.model_name == 'actr_bpr':
            model.activate_actr = True
        for step, seed in enumerate(random_seeds):
            logger.info(f'EVALUATION for #{step + 1} COHORT')
            test_dataloader = dataloader_factory(
                data=data,
                batch_size=batch_size,
                seqlen=seqlen,
                mode=mode,
                num_scored_users=num_scored_users,
                model_name=model_name,
                embedding_dim=embedding_dim,
                random_seed=seed,
                num_favs=num_favs,
                command='eval',
                aggregate_type=aggregate_type,
                activate_actr=True)
            # Evaluate
            ref_user_items = test_dataloader.get_ref_user_items()
            evaluator = Evaluator(config=eval_params,
                                  ref_user_items=ref_user_items,
                                  user_tracks=user_tracks,
                                  track_popularities=data[
                                      'glob_track_popularities'],
                                  num_sess_test=data['data_split'][mode],
                                  mode=mode)
            # model specification
            model_spec = gen_model_spec(training_params, model_params)
            # noinspection PyTypeChecker
            reco_outpath = os.path.join(
                reco_parent_path, f'{model_spec}_seed{seed}.pkl')
            reco_items = _recommend(dataloader=test_dataloader,
                                    model=model,
                                    top_n=evaluator.max_k,
                                    reco_outpath=reco_outpath,
                                    seed=seed)
            curr_scores = evaluator.eval(reco_items, mode=eval_mode)
            for metric, val in curr_scores.items():
                scores[metric].append(val)
        # Display final result
        message = ['RESULTS:']
        for metric, score_arr in scores.items():
            message.append(
                f'{metric}: {np.mean(score_arr):8.5f} +/- {np.std(score_arr):8.5f}')
        logger.info('\n'.join(message))


def _recommend(dataloader, model, reco_outpath,
               top_n=10, seed=0):
    logger = get_logger()
    if not os.path.exists(reco_outpath):
        n_batches = dataloader.get_num_batches()
        reco_items = {}
        # for each batch
        for _ in tqdm(range(1, n_batches), desc=f'Evaluating cohort generated '
                                                f'with random seed = {seed}...'):
            feed_dict = {}
            # get batch data
            batch_data = dataloader.next_batch()
            feed_dict['model_feed'] = model.build_feedict(batch_data,
                                                          is_training=False)
            if 'top' in model.model_name:
                feed_dict['user_ids'] = batch_data[0]
                feed_dict['item_pops'] = dataloader.get_item_pops()
                feed_dict['n_test'] = dataloader.get_num_test_sessions()
            else:
                feed_dict['item_ids'] = dataloader.item_ids
                feed_dict['user_ids'] = batch_data[-1]
            if model.model_name == 'actr':
                feed_dict['nxt_indices'] = batch_data[1]
                actr_scores = defaultdict(dict)
                for idx in range(len(batch_data[-1])):
                    uid = batch_data[-1][idx]
                    nxt_idx = batch_data[1][idx]
                    actr = batch_data[-2][idx]
                    actr_scores[uid][nxt_idx] = actr
                model.actr_scores = actr_scores
            if model.model_name == 'actr_bpr':
                feed_dict['nxt_indices'] = batch_data[1]
                actr_scores = []
                for idx in range(len(batch_data[-1])):
                    uid = batch_data[-1][idx]
                    nxt_idx = batch_data[1][idx]
                    actr = dataloader.data['test_actr_scores'][uid][nxt_idx].toarray()[0]
                    actr_scores.append(actr)
                feed_dict['actr_scores'] = np.array(actr_scores)
            # get prediction from model
            batch_reco_items = model.predict(feed_dict, top_n=top_n)
            for uid, items in batch_reco_items.items():
                reco_items[uid] = items
        logger.info(f'Write prediction to {reco_outpath}')
        # noinspection PyTypeChecker
        pickle.dump(reco_items, open(reco_outpath, 'wb'))
    else:
        logger.info(f'Load prediction from {reco_outpath}')
        reco_items = pickle.load(open(reco_outpath, 'rb'))
    return reco_items
