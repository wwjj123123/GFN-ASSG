"""
Basic operations on trees.
"""

import numpy as np
from collections import defaultdict

import copy

class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self,'_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self,'_depth'):
            return self._depth
        count = 0
        if self.num_children>0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

def head_to_tree(head, tokens, len_):
    """
    Convert a sequence of head indexes into a tree object.
    """
    if isinstance(head, list) == False:
        tokens = tokens[:len_].tolist()
        head = head[:len_].tolist()
    root = None

    nodes = [Tree() for _ in head]

    for i in range(len(nodes)):
        h = head[i]
        nodes[i].idx = i
        nodes[i].dist = -1 # just a filler
        if h == 0:
            root = nodes[i]
        else:
            try:
                nodes[h-1].add_child(nodes[i])
            except:
                print(len_)
                exit()

    assert root is not None
    return root

def tree_to_adj(sent_len, tree, directed=False, self_loop=True):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret




import numpy as np
import heapq

def dijkstra(adj_matrix, start_node):
    sent_len = adj_matrix.shape[0] # 获取邻接矩阵的行数（或列数），即图中节点的数量，并将其存储在 sent_len 中
    distances = np.full(sent_len, np.inf) # 创建一个名为 distances 的数组，初始化为无穷大（np.inf），表示从起始节点到所有其他节点的初始距离都是无穷大。
    distances[start_node] = 0 # 将起始节点到自身的距离设置为 0，因为从一个节点到自身的距离是 0。
    priority_queue = [(0, start_node)]  # (distance, node)创建一个优先队列（使用列表），初始时只包含一个元组 (0, start_node)，表示起始节点的当前距离为 0。

    while priority_queue:# 开始一个循环，只要优先队列不为空，就继续执行
        current_distance, current_node = heapq.heappop(priority_queue)# 从优先队列中弹出距离最小的节点，并将其距离和节点索引分别存储在 current_distance 和 current_node 中

        if current_distance > distances[current_node]:# 如果当前弹出的距离大于已知的最短距离，说明这个节点已经被处理过，跳过该节点
            continue

        for neighbor in range(sent_len): # 遍历所有邻接节点（从 0 到 sent_len - 1）。
            if adj_matrix[current_node][neighbor] > 0:  # 检查当前节点 current_node 和邻接节点 neighbor 之间是否存在连接（即邻接矩阵中的值大于 0）。
                distance = current_distance + adj_matrix[current_node][neighbor]# 计算从起始节点到邻接节点的潜在新距离，即当前距离加上连接的权重
                if distance < distances[neighbor]: # 如果计算出的新距离小于已知的到邻接节点的距离，则更新距离。
                    distances[neighbor] = distance # 更新到邻接节点的最短距离
                    heapq.heappush(priority_queue, (distance, neighbor)) # 将更新后的邻接节点和其新距离推入优先队列，以便后续处理

    return distances # 返回从起始节点到所有其他节点的最短距离数组

def tree_to_adj_D(sent_len, tree, aspect_range, directed=False, self_loop=True):
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)
    queue = [tree]
    
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]
        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    # # 将 aspect_range 转换为 0 基索引
    # aspect_range = [i - 1 for i in aspect_range]  # 假设 aspect_range 是一个列表
    # start_indices = aspect_range[:-1]  # 取出左闭右开范围的起始索引
    # D_min = np.zeros((sent_len, sent_len), np.float32)
    
    # 将 aspect_range 转换为 0 基索引（本来就是0 基索引，已修正）
    # aspect_range = [i for i in aspect_range]  # 假设 aspect_range 是一个列表 
    #     (又错一版本，没加range迭代器)
    # 也不算错，一个方面词里面有多个token，目前我想不到如何处理后几个token，继续沿用对aspect头词计算最短路径的方法
    
    start_indices =[aspect_range[0]]

    D_min = np.zeros((sent_len, sent_len), np.float32)


    # 计算从每个 start 到所有节点的最短路径
    for start in start_indices:
        distances = dijkstra(ret, start)
        for i in range(sent_len):
            if i not in start_indices:  # 排除 start_indices 中的节点
                D_min[start][i] = distances[i]
                D_min[i][start] = distances[i]

    # 将 aspect_range 内的节点的距离设为 1
    for j in start_indices:
        for k in start_indices:
            D_min[j][k] = 1
            D_min[k][j] = 1

    return D_min



def calculate_shortest_paths(ret, start_indices, end_indices):

    sent_len = ret.shape[0]

    start_indices =[start_indices]

    D_min = np.zeros(ret.shape, np.float32)


    # 计算从每个 start 到所有节点的最短路径(其实就算了一个aspect头词的)
    for start in start_indices:
        distances = dijkstra(ret, start)
        for i in range(sent_len):
            if i not in start_indices:  # 排除 start_indices 中的节点
                D_min[start][i] = distances[i]
                D_min[i][start] = distances[i]

    # 将 aspect_range 内的节点的距离设为 1
    for j in start_indices:
        for k in start_indices:
            D_min[j][k] = 1
            D_min[k][j] = 1

    return D_min