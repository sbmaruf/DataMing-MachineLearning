#!/usr/bin/env python
# coding=utf-8


def check_type(input_value, expected_type):
    if not isinstance(input_value, expected_type):
        raise TypeError('invalid type %s, type %s is needed' % (type(input_value), expected_type))
