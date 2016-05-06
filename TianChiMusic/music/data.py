import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import utils


def basic_data():
    """
    load basic data
    """
    songs = pd.read_csv('../data/mars_tianchi_songs.csv', engine='c', header=None)
    actions = pd.read_csv('../data/mars_tianchi_user_actions.csv', engine='c', header=None)
    
    songs.columns = ['song_id', 'artist_id', 'publish_time', 'song_init_plays', 'Language', 'Gender']
    actions.columns = ['user_id', 'song_id', 'gmt_create', 'action_type', 'Ds']
    
    actions['Ds'] = actions['Ds'].apply(utils.num_to_date)
    actions['gmt_create'] = actions['gmt_create'].apply(utils.timestamp2datatime)
    
    day_total_play = {}
    day_singer_play = {}
    
    for _time in actions['Ds'].tolist():
        day_total_play[_time] = day_total_play.get(_time, 0) + 1
    
    # plot time and total play
    day_total_play = pd.Series(day_total_play)
    fig = plt.plot(day_total_play.index, day_total_play.values)
    plt.show(block=False)
    
    artists_songs = {}
    uniquevalues = np.unique(songs['artist_id'].values)
    for _id in uniquevalues:
            artists_songs[_id] = songs[songs['artist_id'] == _id]
    #sonns_info = {}
    #for _id in np.unique(actions['song_id'].values):
    #    sonns_info[_id] = actions[actions['song_id'] == _id]
    return songs, actions, day_total_play, artists_songs

