#!/usr/bin/env python
# coding=utf-8
import sys
import math
import operator
import collections

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

    def get_entropy(self, data_set):
        _num_entries = len(data_set)
        _label_counts = dict()
        for vec in data_set:
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

    def choose_best_feature(self, data_set):
        _num_features, _base_entropy = len(data_set[0]) - 1, self.get_entropy(data_set)
        _max_info_gain, _best_feature = 0.0, -1
        for _axis in range(_num_features):
            _feature_list = [_vec[_axis] for _vec in data_set]
            _unique_values, _conditional_entropy = set(_feature_list), 0
            # 得到条件熵
            for _val in _unique_values:
                _sub_data_set = self.split_data(data_set, _axis, _val)
                _prob = len(_sub_data_set) / float(len(data_set))
                _conditional_entropy += _prob * self.get_entropy(_sub_data_set)
            _info_gain = _base_entropy - _conditional_entropy
            if _info_gain > _max_info_gain:
                _max_info_gain, _best_feature = _info_gain, _axis
        return _best_feature

    def majority_count(self, class_list):
        if len(class_list) == 0:
            raise ValueError('empty class list')
        return sorted(collections.Counter(class_list).iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]

    def create_tree(self, data_set, labels):
        _class_list = [_vec[-1] for _vec in data_set]
        if _class_list.count(_class_list[0]) == len(_class_list):
            return _class_list
        if len(data_set[0]) == 1:
            return self.majority_count(_class_list)
        _best_feature = self.choose_best_feature(data_set)
        _best_feat_label = labels[_best_feature]
        _my_tree = {_best_feat_label: {}}
        del(labels[_best_feature])
        feat_values = [_vec[_best_feature] for _vec in data_set]
        unique_values = set(feat_values)
        for _value in unique_values:
            sub_labels = labels[:]
            _my_tree[_best_feat_label][_value] = self.create_tree(
                self.split_data(data_set, _best_feature, _value), sub_labels)
        return _my_tree

    def devide_data(self, divide_data, column, value):
        _func = None
        if isinstance(value, int) or isinstance(value, float):
            _func = lambda x: x >= value
        else:
            _func = lambda x: x == value
        set_1 = [_row for _row in divide_data if _func(_row[column])]
        set_2 = [_row for _row in divide_data if not _func(_row[column])]
        return set_1, set_2


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

        super(TreeNode, self).__init__()


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


def load_iris_data():
    labels = ['sepal length', 'sepal width', 'petal length', 'petal width']
    with open('iris.txt') as f:
        return [[_v.split('\n')[0] for _i, _v in enumerate(_line.split(','))] for _line in f.readlines()], labels


my_tree = {'petal length': {'1.0': ['Iris-setosa'],
                            '1.1': ['Iris-setosa'],
                            '1.2': ['Iris-setosa', 'Iris-setosa'],
                            '1.3': ['Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa'],
                            '1.4': ['Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa'],
                            '1.5': ['Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa'],
                            '1.6': ['Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa',
                                    'Iris-setosa'],
                            '1.7': ['Iris-setosa', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa'],
                            '1.9': ['Iris-setosa', 'Iris-setosa'],
                            '3.0': ['Iris-versicolor'],
                            '3.3': ['Iris-versicolor', 'Iris-versicolor'],
                            '3.5': ['Iris-versicolor', 'Iris-versicolor'],
                            '3.6': ['Iris-versicolor'],
                            '3.7': ['Iris-versicolor'],
                            '3.8': ['Iris-versicolor'],
                            '3.9': ['Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor'],
                            '4.0': ['Iris-versicolor',
                                    'Iris-versicolor',
                                    'Iris-versicolor',
                                    'Iris-versicolor',
                                    'Iris-versicolor'],
                            '4.1': ['Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor'],
                            '4.2': ['Iris-versicolor',
                                    'Iris-versicolor',
                                    'Iris-versicolor',
                                    'Iris-versicolor'],
                            '4.3': ['Iris-versicolor', 'Iris-versicolor'],
                            '4.4': ['Iris-versicolor',
                                    'Iris-versicolor',
                                    'Iris-versicolor',
                                    'Iris-versicolor'],
                            '4.5': {'sepal length': {'4.9': ['Iris-virginica'],
                                                     '5.4': ['Iris-versicolor'],
                                                     '5.6': ['Iris-versicolor'],
                                                     '5.7': ['Iris-versicolor'],
                                                     '6.0': ['Iris-versicolor', 'Iris-versicolor'],
                                                     '6.2': ['Iris-versicolor'],
                                                     '6.4': ['Iris-versicolor']}},
                            '4.6': ['Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor'],
                            '4.7': ['Iris-versicolor',
                                    'Iris-versicolor',
                                    'Iris-versicolor',
                                    'Iris-versicolor',
                                    'Iris-versicolor'],
                            '4.8': {'sepal length': {'5.9': ['Iris-versicolor'],
                                                     '6.0': ['Iris-virginica'],
                                                     '6.2': ['Iris-virginica'],
                                                     '6.8': ['Iris-versicolor']}},
                            '4.9': {'sepal width': {'2.5': ['Iris-versicolor'],
                                                    '2.7': ['Iris-virginica'],
                                                    '2.8': ['Iris-virginica'],
                                                    '3.0': ['Iris-virginica'],
                                                    '3.1': ['Iris-versicolor']}},
                            '5.0': {'sepal length': {'5.7': ['Iris-virginica'],
                                                     '6.0': ['Iris-virginica'],
                                                     '6.3': ['Iris-virginica'],
                                                     '6.7': ['Iris-versicolor']}},
                            '5.1': {'sepal length': {'5.8': ['Iris-virginica',
                                                             'Iris-virginica',
                                                             'Iris-virginica'],
                                                     '5.9': ['Iris-virginica'],
                                                     '6.0': ['Iris-versicolor'],
                                                     '6.3': ['Iris-virginica'],
                                                     '6.5': ['Iris-virginica'],
                                                     '6.9': ['Iris-virginica']}},
                            '5.2': ['Iris-virginica', 'Iris-virginica'],
                            '5.3': ['Iris-virginica', 'Iris-virginica'],
                            '5.4': ['Iris-virginica', 'Iris-virginica'],
                            '5.5': ['Iris-virginica', 'Iris-virginica', 'Iris-virginica'],
                            '5.6': ['Iris-virginica',
                                    'Iris-virginica',
                                    'Iris-virginica',
                                    'Iris-virginica',
                                    'Iris-virginica',
                                    'Iris-virginica'],
                            '5.7': ['Iris-virginica', 'Iris-virginica', 'Iris-virginica'],
                            '5.8': ['Iris-virginica', 'Iris-virginica', 'Iris-virginica'],
                            '5.9': ['Iris-virginica', 'Iris-virginica'],
                            '6.0': ['Iris-virginica', 'Iris-virginica'],
                            '6.1': ['Iris-virginica', 'Iris-virginica', 'Iris-virginica'],
                            '6.3': ['Iris-virginica'],
                            '6.4': ['Iris-virginica'],
                            '6.6': ['Iris-virginica'],
                            '6.7': ['Iris-virginica', 'Iris-virginica'],
                            '6.9': ['Iris-virginica']}}
