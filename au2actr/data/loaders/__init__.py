from au2actr import Au2ActrError
from au2actr.data.loaders.train_loader import TrainDataLoader, CFTrainDataLoader
from au2actr.data.loaders.valid_loader import ValidDataLoader
from au2actr.data.loaders.test_loader import TestDataLoader

_SUPPORTED_DATALOADERS = {
    'train': TrainDataLoader,
    'cf_train': CFTrainDataLoader,
    'valid': ValidDataLoader,
    'test': TestDataLoader
}


def dataloader_factory(data, batch_size, seqlen, mode='train',
                       random_seed=2025, **kwargs):
    kwargs['mode'] = mode
    kwargs['track_ids_map'] = data['track_ids_map']
    kwargs['user_ids_map'] = data['user_ids_map']
    reco_enable = kwargs.get('reco_enable', False)
    if reco_enable is True:
        _SUPPORTED_DATALOADERS['valid'] = TestDataLoader
    else:
        _SUPPORTED_DATALOADERS['valid'] = ValidDataLoader
    if mode == 'train':
        kwargs['train_session_indexes'] = data['train_session_indexes']
    try:
        return _SUPPORTED_DATALOADERS[mode](data,
                                            n_users=data['n_users'],
                                            n_items=data['n_items'],
                                            batch_size=batch_size,
                                            seqlen=seqlen,
                                            random_seed=random_seed,
                                            **kwargs)
    except KeyError as err:
        raise Au2ActrError(f'{err}')
