#!/usr/bin/env python
# coding=utf-8
import datetime


import numpy as np


class ProcessData(object):
    """"""
    def __init__(self, actions, songs):
        self.actions = actions
        self.songs = songs

    def hot_singers(self, top=10, criteria=None):
        """
        找出top-k的热门歌手
        """
        pass

    def hot_songs(self, top=10, action_type=1):
        """
        action_type = 1, 2, 3
        """
        pass

    def active_users(self, top=10, action_type=1):
        """"""
        pass

    def get_user_hist_action(self, user_id, time_value, action_type=1):
        """
        获取用户的历史播放记录
        action_type =1, 2, 3, all
        """
        _user = self.actions[self.actions.user_id==user_id]
        _user_time = _user[_user.Ds <= time_value]
        if action_type != 'all':
            _user_time_action = _user_time[_user_time.action_type==action_type]
        else:
            _user_time_action = _user_time
        return _user_time_action.ix[:, ['song_id', 'gmt_create', 'Ds', 'action_type']]\
                                    .sort_values('gmt_create')

