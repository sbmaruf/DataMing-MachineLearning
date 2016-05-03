# coding utf-8
'''
Created on 2016.3.31

@author: qiuwei
'''
import datetime


def num_to_date(int_value):
    """
    >>> import utils
    >>> utils.num_to_date(20150302)
    datetime.date(2015, 3, 2)
    >>> utils.num_to_date(20151202)
    datetime.date(2015, 12, 2)
    >>> utils.num_to_date(20151222)
    datetime.date(2015, 12, 22)
    >>> utils.num_to_date(20150222)
    datetime.date(2015, 2, 22)
    >>> utils.num_to_date(20050222)
    datetime.date(2005, 2, 22)
    >>> utils.num_to_date(10050222)
    datetime.date(1005, 2, 22)
    """
    _day_mon = int_value % 10000
    _year = (int_value - _day_mon) / 10000
    _mon = _day_mon / 100
    _day = _day_mon - _mon * 100
    return datetime.date(_year, _mon, _day)
