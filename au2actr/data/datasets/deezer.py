import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

from au2actr.data.datasets.dataset import Dataset
from au2actr.utils.tempo import high_level_tempo_from_ts


class DeezerDataset(Dataset):

    def _load_stream_sessions(self):
        output_path = os.path.join(self.cache_path, 'user_sessions.pkl')
        if not os.path.exists(output_path):
            session_path = os.path.join(self.dataset_params['path'],
                                        self.dataset_params['name'],
                                        f'min{self.min_sessions}sess',
                                        self.dataset_params['files']['streams'])
            self.logger.info(f'Read user session streams from {session_path}')
            # noinspection PyTypeChecker
            streams_df = pd.read_parquet(session_path)
            grouped_streams = streams_df.sort_values(['ts']).groupby(
                ['user_id', 'session_id'])
            user_sessions = defaultdict(list)
            user_sess_index_map = defaultdict(dict)
            # get context infos
            self.logger.info('Get context infos')
            first_rows_grouped_streams = grouped_streams.first().reset_index()
            first_rows_grouped_streams = first_rows_grouped_streams[
                ['user_id', 'session_id', 'ts']]
            first_rows_grouped_streams = first_rows_grouped_streams.sort_values(
                ['ts']).groupby('user_id')

            for user_id, df_group in first_rows_grouped_streams:
                session_ids = df_group['session_id'].tolist()
                user_sess_index_map[user_id] = {sid: idx for idx, sid in
                                                enumerate(session_ids)}
                timestamps = df_group['ts'].tolist()
                for idx, (sid, ts) in enumerate(zip(session_ids, timestamps)):
                    if idx == 0:
                        time_since_last_session = 0
                    else:
                        time_since_last_session = ts - timestamps[idx - 1]
                    day_of_week, hour_of_day = high_level_tempo_from_ts(ts)
                    user_sessions[user_id].append({
                        'session_id': sid,
                        'context': {
                            'time_since_last_session': time_since_last_session,
                            'ts': ts,
                            'day_of_week': day_of_week,
                            'hour_of_day': hour_of_day
                        }
                    })
            # tracks infos
            self.logger.info('Get track list in each session')
            for group_name, df_group in grouped_streams:
                user_id, session_id = group_name
                track_ids = df_group['track_id'].astype(
                    'int32').tolist()
                idx = user_sess_index_map[user_id][session_id]
                # noinspection PyTypeChecker
                user_sessions[user_id][idx]['track_ids'] = track_ids
            # write result to cache
            self.logger.info(f'Write user session streams to {output_path}')
            # noinspection PyTypeChecker
            pickle.dump(user_sessions, open(output_path, 'wb'))
        else:
            self.logger.info(f'Load user session streams from {output_path}')
            user_sessions = pickle.load(open(output_path, 'rb'))
        return user_sessions

    def _load_tracks(self):
        if self.normalize_embedding:
            svd_embeddings_path = os.path.join(
                self.cache_path, f'norm_track_svd_embeddings.pkl')
            audio_embeddings_path = os.path.join(
                self.cache_path, f'norm_track_audio_embeddings.pkl')
        else:
            svd_embeddings_path = os.path.join(self.cache_path,
                                               f'track_svd_embeddings.pkl')
            audio_embeddings_path = os.path.join(self.cache_path,
                                                 f'track_audio_embeddings.pkl')
        if not os.path.exists(svd_embeddings_path) or not \
            os.path.exists(audio_embeddings_path):
            input_track_embeddings_path = os.path.join(
                self.dataset_params['path'], self.dataset_params['name'],
                f'min{self.min_sessions}sess',
                self.dataset_params['files']['track_embeddings'])
            self.logger.info(f'Read track embeddings from '
                             f'{input_track_embeddings_path}')
            # noinspection PyTypeChecker
            track_embs_df = pd.read_parquet(input_track_embeddings_path)
            track_ids = track_embs_df['track_id'].tolist()
            art_ids = track_embs_df['art_id'].tolist()
            svd_embeddings_arr = np.array(
                [self._convert_track_embeddings_to_array(e)
                 for e in track_embs_df['svd'].tolist()])
            audio_embeddings_arr = np.array(track_embs_df['audio'].tolist())
            if self.normalize_embedding is True:
                self.logger.info('Normalize track embeddings')
                l2_svd_norm = np.linalg.norm(svd_embeddings_arr, ord=2, axis=1,
                                         keepdims=True)
                svd_embeddings_arr = svd_embeddings_arr / l2_svd_norm
                l2_audio_norm = np.linalg.norm(audio_embeddings_arr, ord=2,
                                               axis=1, keepdims=True)
                audio_embeddings_arr = audio_embeddings_arr / l2_audio_norm
            # SVD & audio embeddings
            svd_embeddings = dict(zip(track_ids, svd_embeddings_arr))
            audio_embeddings = dict(zip(track_ids, audio_embeddings_arr))
            art_ids = list(set(art_ids))
            # noinspection PyTypeChecker
            pickle.dump(svd_embeddings, open(svd_embeddings_path, 'wb'))
            # noinspection PyTypeChecker
            pickle.dump(audio_embeddings, open(audio_embeddings_path, 'wb'))
            np.savez(self.entities_path, track_ids=track_ids, art_ids=art_ids)
        else:
            self.logger.info(f'Load SVD embeddings from '
                             f'{svd_embeddings_path}')
            svd_embeddings = pickle.load(open(svd_embeddings_path, 'rb'))
            self.logger.info(f'Load AUDIO embeddings from '
                             f'{audio_embeddings_path}')
            audio_embeddings = pickle.load(open(audio_embeddings_path, 'rb'))
            entities = np.load(self.entities_path, allow_pickle=True)
            track_ids = entities['track_ids']
            art_ids = entities['art_ids']
        out = {
            'track_ids': track_ids,
            'art_ids': art_ids,
            'svd_embeddings': svd_embeddings,
            'audio_embeddings': audio_embeddings
        }
        return out

    @classmethod
    def _convert_track_embeddings_to_array(cls, embeddings):
        output = [it['item'] for it in embeddings['list']]
        return output
