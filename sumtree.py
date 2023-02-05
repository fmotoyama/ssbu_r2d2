# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 14:04:30 2022

@author: motoyama
"""
import math, random

class Node:
    def __init__(self, priority, parent):
        self.priority = priority
        self.left = None
        self.right = None
        self.parent = parent

class Leaf:
    def __init__(self, parent):
        self.index = None
        self.priority = 0
        self.value = None
        self.parent = parent


class SumTree:
    def __init__(self, l):
        self.l = l
        self.root = Node(priority=0, parent=None)
        self.all_leaves = []        # all_leaves[index] = leaf
        self.next_leaf_index = 0
        self.count_add = 0          # 追加したデータの個数をカウント
        
        # 空の木を作成
        depth = math.ceil(math.log2(self.l))   # 葉を除く木の段数
        floor = [self.root]
        for _ in range(1,depth):
            floor2 = []
            for node in floor:
                left_node = Node(0, parent=node)
                right_node = Node(0, parent=node)
                node.left = left_node
                node.right = right_node
                floor2.append(left_node)
                floor2.append(right_node)
            floor = floor2
        
        for node in floor:
            node.left = Leaf(parent=node)
            node.right = Leaf(parent=node)
            self.all_leaves.append(node.left)
            self.all_leaves.append(node.right)
                
        
    def add(self, value, priority):
        # value,priorityをもつ葉を作成
        leaf = self.all_leaves[self.next_leaf_index]
        leaf.index = self.next_leaf_index
        leaf.value = value
        self.propagate(self.next_leaf_index, priority)
        self.next_leaf_index += 1
        self.count_add += 1
        if self.l <= self.next_leaf_index:
            self.next_leaf_index = 0
    
    
    def propagate(self, leaf_index, priority):
        # 葉の優先度の値をpriorityにする
        node = self.all_leaves[leaf_index]
        delta = priority - node.priority
        while node:
            node.priority += delta
            node = node.parent
    
    
    def get(self):
        # 優先度に従って葉を1つ抽出する
        priority = self.root.priority * random.uniform(0, 1)
        node = self.root
        while True:
            if priority <= node.left.priority:
                node = node.left
            else:
                priority -= node.left.priority
                node = node.right
            if isinstance(node, Leaf):
                break
        return node.value, node.index, node.priority/self.root.priority
    




import numpy as np

class SequencialMemory:
    """"""
    def __init__(self, l:int, actors:int):
        # l: 各actorにより保存される経験の個数
        self.l = l
        self.priorities = np.zeros(l*actors, dtype='f4')
        self.memory = [[None for _ in range(l)] for _ in range(actors)]
        self.index = [0,0,0]
    
    def add(self, actor_id, value, priority):
        if self.l <= self.index[actor_id]:
            self.index[actor_id] =  0
        
        self.memory[actor_id][self.index[actor_id]] = value
        self.priorities[self.index[actor_id] + actor_id*self.l] = priority
        
        self.index[actor_id] += 1
    
    def get_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch,None,1
    
    def update(self, indices, td_errors):
        pass










