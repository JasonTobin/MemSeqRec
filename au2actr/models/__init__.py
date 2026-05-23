from au2actr import Au2ActrError
from au2actr.models.baselines.freq.topfreq import TopFreq
from au2actr.models.baselines.cf.actr_bpr import ACTR_BPR
from au2actr.models.baselines.actr import ACTR
from au2actr.models.baselines.pisa import PISA
from au2actr.models.baselines.memseq import MemSeqRec
from au2actr.models.net import AU2ACTR


_TRAINED_MODELS = {
    'au2actr': AU2ACTR,
    'pisa': PISA,
    'actr': ACTR,
    'actr_bpr': ACTR_BPR,
    'memseq': MemSeqRec,
    'gtop': TopFreq,
    'ptop': TopFreq
}

UNTRAINED_MODELS = {'gtop', 'ptop', 'last', 'actr'}


class ModelFactory:
    @classmethod
    def generate_model(cls, sess, params, n_users, n_items,
                       pretrained_embs=None, command='train'):
        """
        Factory method to generate a model
        :param sess:
        :param params:
        :param n_users:
        :param n_items:
        :param pretrained_embs: dictionary of pretrained embeddings
        :param command:
        :return:
        """
        model_name = params['model']['name']
        try:
            # create a new model
            mdl = _TRAINED_MODELS[model_name](sess=sess,
                                              params=params,
                                              n_users=n_users,
                                              n_items=n_items,
                                              pretrained_embs=pretrained_embs)
            if model_name not in UNTRAINED_MODELS:
                if command == 'train':
                    # build computation graph
                    mdl.build_graph(name=model_name)
                elif command == 'eval' or command == 'score':
                    mdl.restore(name=model_name)
            elif model_name == 'last' or model_name == 'actr':
                mdl.build_graph(name=model_name)
            return mdl
        except KeyError as e:
            raise Au2ActrError(f'{model_name}: {e}')
