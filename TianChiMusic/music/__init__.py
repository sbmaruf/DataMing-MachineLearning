#!/usr/bin/env python
# coding=utf-8
import datetime
import collections

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import utils
import data

if __name__ == '__main__':

    songs, actions, day_total_play, artists_songs = data.basic_data()
    songs_Ds_actions = actions.groupby(['song_id', 'Ds', 'action_type']
                             )\
                            .action_type.sum()\
                            .unstack(['action_type', 'Ds'])\
                            .fillna(0)\
                            .sortlevel(level=[0, 1], axis=1)
    users_action = actions.groupby(['user_id', 'action_type']
                            ).action_type.sum().unstack(['action_type']).fillna(0)
    songs_action = actions.groupby(['song_id', 'action_type']
                            ).action_type.sum().unstack(['action_type']).fillna(0)

    songs_actions_sum = songs_action.assign(sum=songs_action.sum(axis=1)).sort_values('sum', ascending=True)
    users_actions_sum = users_action.assign(sum=users_action.sum(axis=1)).sort_values('sum', ascending=True)

    artists = {}
    for _id in np.unique(songs.artist_id):
        _temp = (songs[songs.artist_id==_id]).song_id
        artists[_id] = [_temp, len(_temp)]

    _a = sorted(artists.iteritems(), key=lambda x:x[1][1])
    _data = np.array([[key for key, _ in _a], [val[1] for _, val in _a]]).T
    artists_plays = pd.DataFrame(_data, columns=['artist_id', 'init_play'])

