#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('..')
from utils import utils


class Tree(object):
    """
    树的父类
    """
    def __init__(self):
        pass

    def train(self):
        pass

    def prune(self):
        pass


class TreeNode(object):
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
