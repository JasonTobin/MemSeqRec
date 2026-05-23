from au2actr.data.datasets.deezer import DeezerDataset

_SUPPORTED_DATASETS = {
    'deezer': DeezerDataset
}


def dataset_factory(params):
    """
    Factory that generate dataset
    :param params:
    :return:
    """
    dataset_name = params['dataset'].get('name', 'deezer')
    try:
        dataset = _SUPPORTED_DATASETS[dataset_name](params)
        data = dataset.fetch_data()
        return data
    except KeyError:
        raise KeyError(f'Not support {dataset_name} dataset')
