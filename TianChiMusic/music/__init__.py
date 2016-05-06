import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import utils
import data

if __name__ == '__main__':

    songs, actions, day_total_play, artists_songs = data.basic_data()
    new_df = actions.groupby(['song_id', 'Ds', 'action_type']
                             )\
                            .action_type.sum()\
                            .unstack(['action_type', 'Ds'])\
                            .fillna(0)\
                            .sortlevel(level=[0, 1], axis=1)
