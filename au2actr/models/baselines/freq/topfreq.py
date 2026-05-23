from au2actr.logging import get_logger


class TopFreq:
    """
    Recommendation based on top frequency (global & personal)
    """

    def __init__(self, sess, params, n_users, n_items, pretrained_embs):
        self.logger = get_logger()
        self.sess = sess
        self.model_name = params['model']['name']
        self.n_users = n_users
        self.n_items = n_items

    @classmethod
    def build_feedict(cls, batch, is_training=True):
        return {
            'user_ids': batch[0]
        }

    def predict(self, feed_dict, top_n=50):
        reco_items = {}
        if self.model_name == 'gtop':
            sorted_items_pops = sorted(feed_dict['item_pops']['glob'].items(),
                                       key=lambda x: (-x[1], x[0]))
            item_ids = [iid for iid, _ in sorted_items_pops][:top_n]
            for uid in feed_dict['model_feed']['user_ids']:
                reco_items[uid] = feed_dict['n_test'] * [item_ids]
        else:
            for uid in feed_dict['model_feed']['user_ids']:
                sorted_items_pops = sorted(feed_dict['item_pops']['pers'][uid].items(),
                                           key=lambda x: (-x[1], x[0]))
                item_ids = [iid for iid, _ in sorted_items_pops][:top_n]
                reco_items[uid] = feed_dict['n_test'] * [item_ids]
        return reco_items
