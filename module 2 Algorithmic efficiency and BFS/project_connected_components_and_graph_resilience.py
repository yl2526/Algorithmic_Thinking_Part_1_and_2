# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 22:04:53 2016
@author: Yi

This script is for project 2
Connected components and graph resilience
"""

from collections import deque
from copy import deepcopy
        
EX_GRAPH0 = {0: set([1, 2]), 
             1: set([0, 2]), 
             2: set([0, 1]), 
             3: set([])
            }

EX_GRAPH1 = {0: set([1, 2]), 
             1: set([0, 2]), 
             2: set([0, 1]), 
             3: set([4]), 
             4: set([3]), 
             5: set([])
            }

def bfs_visited(ugraph, start_node):
    '''
    the bfs visited function 
    return set of nodes visted
    '''
    check_queue = deque([start_node])
    visited = set([start_node])
    while check_queue:
        current_node = check_queue.pop()
        for neighbor in ugraph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                check_queue.append(neighbor)
    return visited
        
bfs_visited(ugraph = EX_GRAPH0, start_node = 3)
bfs_visited(ugraph = EX_GRAPH1, start_node = 1)  
bfs_visited(ugraph = EX_GRAPH1, start_node = 3)
bfs_visited(ugraph = EX_GRAPH1, start_node = 5)
    
def cc_visited(ugraph):
    '''
    the connected component visited function 
    return list of sets, each sets is a connected nodes
    '''
    remained = set(ugraph.keys())
    connected_components = []
    while remained:
        start_node = remained.pop()
        connected = bfs_visited(ugraph, start_node)
        connected_components.append(connected)
        remained = remained - connected
    return connected_components

cc_visited(ugraph = EX_GRAPH0)
cc_visited(ugraph = EX_GRAPH1)

def largest_cc_size(ugraph):
    '''
    returns the size (an integer) of the largest connected component 
    in ugraph
    '''
    connected_components = cc_visited(ugraph)
    if connected_components: 
        largest_connected = max(map(len, connected_components))
    else:
        return 0
    return largest_connected

largest_cc_size(ugraph = EX_GRAPH0)
largest_cc_size(ugraph = EX_GRAPH1)

def compute_resilience(ugraph, attack_order):
    '''
    Takes the undirected graph ugraph, a list of nodes attack_order and 
    iterates through the nodes in attack_order. For each node in the list, 
    the function removes the given node and its edges from the graph and 
    then computes the size of the largest connected component for the 
    resulting graph.
    The function should return a list whose k+1th entry is the size of 
    the largest connected component in the graph after the removal of the 
    first k nodes in attack_order. The first entry (indexed by zero) is 
    the size of the largest connected component in the original graph.
    '''
    graph_copy = deepcopy(ugraph)
    #graph_copy = ugraph
    resilience = [largest_cc_size(graph_copy)]
    for attack in attack_order:
        connecting_nodes = graph_copy.pop(attack)
        #return graph_copy, connecting_nodes, attack
        for connecting_node in connecting_nodes:
                graph_copy[connecting_node].remove(attack)
        resilience.append(largest_cc_size(graph_copy))
    return resilience

#compute_resilience(ugraph = EX_GRAPH0, attack_order = [1])
#compute_resilience(ugraph = EX_GRAPH0, attack_order = [1, 2])
#compute_resilience(ugraph = EX_GRAPH1, attack_order = [1, 4])
















