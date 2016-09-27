#!/usr/bin/env python
# coding=utf-8
import sys
import math

sys.path.append('..')
from utils import utils


class Tree(object):
    """
    树的父类
    """
    def __init__(self, data_set):
        self.data_set = data_set

    def train(self):
        pass

    def prune(self):
        pass

    def get_entropy(self):
        _num_entries = len(self.data_set)
        _label_counts = dict()
        for vec in self.data_set:
            _curr_label = vec[-1]
            _label_counts[_curr_label] = _label_counts.get(_curr_label, 0) + 1
        entropy = 0
        for key in _label_counts.keys():
            _prob = float(_label_counts[key]) / _num_entries
            entropy -= _prob * math.log(_prob, 2)
        return entropy

    def split_data(self, split_data, axis, value):
        ret_data = []
        for _vec in split_data:
            if _vec[axis] == value:
                _reduced_feature_vec = _vec[:axis]
                _reduced_feature_vec.extend(_vec[axis+1:])
                ret_data.append(_reduced_feature_vec)
        return ret_data


class TreeNode(Tree):
    """
    决策树中的节点
    """
    def __init__(self, data, left_child, right_child, is_leaf):
        self.data = data
        utils.check_type(left_child, TreeNode)
        utils.check_type(right_child, TreeNode)
        self.l_child = left_child
        self.r_child = right_child
        utils.check_type(is_leaf, bool)
        self.leaf = is_leaf

        super(self, TreeNode).__init__()


class RegressionTee(Tree):
    """
    回归树
    """
    pass


class ClassifierTree(Tree):
    """
    分类树
    """
    pass


def create_data_set():
    data_set = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels

