# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:13:23 2016

@author: yliu
"""
from __future__ import division

from project_connected_components_and_graph_resilience import EX_GRAPH0, EX_GRAPH1, bfs_visited, cc_visited, largest_cc_size, compute_resilience
from DPAtrial_undirected import UPATrial
from plot_helper import bar_with_line
from collections import deque, defaultdict
import itertools
import pandas as pd

"""
Provided code for Application portion of Module 2
"""

# general imports
import urllib2
import random
import time
import math

# CodeSkulptor import
#import simpleplot
#import codeskulptor
#codeskulptor.set_timeout(60)

# Desktop imports
#import matplotlib.pyplot as plt


############################################
# Provided code

def copy_graph(graph):
    """
    Make a copy of a graph
    """
    new_graph = {}
    for node in graph:
        new_graph[node] = set(graph[node])
    return new_graph

def delete_node(ugraph, node):
    """
    Delete a node from an undirected graph
    """
    neighbors = ugraph[node]
    ugraph.pop(node)
    for neighbor in neighbors:
        ugraph[neighbor].remove(node)
    
def targeted_order(ugraph):
    """
    Compute a targeted attack order consisting
    of nodes of maximal degree
    
    Returns:
    A list of nodes
    """
    # copy the graph
    new_graph = copy_graph(ugraph)
    
    order = []    
    while len(new_graph) > 0:
        max_degree = -1
        for node in new_graph:
            if len(new_graph[node]) > max_degree:
                max_degree = len(new_graph[node])
                max_degree_node = node
        
        neighbors = new_graph[max_degree_node]
        new_graph.pop(max_degree_node)
        for neighbor in neighbors:
            new_graph[neighbor].remove(max_degree_node)

        order.append(max_degree_node)
    return order

##########################################################
# Code for loading computer network graph

NETWORK_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_rf7.txt"


def load_graph(graph_url):
    """
    Function that loads a graph given the URL
    for a text representation of the graph
    
    Returns a dictionary that models a graph
    """
    graph_file = urllib2.urlopen(graph_url)
    graph_text = graph_file.read()
    graph_lines = graph_text.split('\n')
    graph_lines = graph_lines[ : -1]
    
    print "Loaded graph with", len(graph_lines), "nodes"
    
    answer_graph = {}
    for line in graph_lines:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1 : -1]:
            answer_graph[node].add(int(neighbor))

    return answer_graph

##########################################################
# Code that I wrote myself

computer_network = load_graph(NETWORK_URL)
computer_network_node = 1239
computer_network_edge = 3047
   
def make_ER_ugraph(num_nodes, p = 0.5):
    '''
    Takes the number of nodes num_nodes and returns a 
    dictionary corresponding to a ramdon undirected graph 
    with the specified number of nodes
    implement the ER algorith in HW 10
    '''
    graph_dict = defaultdict(lambda: set([]))
    for key in xrange(num_nodes):
        other_notes = range(key) + range(key+1, num_nodes)
        for node in other_notes:
            if random.random() < p:
                graph_dict[key].add(node)
                graph_dict[node].add(key)
    return graph_dict

def make_complete_ugraph(num_nodes):
    '''
    Takes the number of nodes num_nodes and returns a 
    dictionary corresponding to a complete directed graph 
    with the specified number of nodes
    '''
    graph_dict = defaultdict(lambda: set([]))
    for key in xrange(num_nodes):
        graph_dict[key] = set(range(key) + range(key+1, num_nodes))
    return graph_dict

def random_order(ugraph):
    '''
    generate a list of nodes in random order 
    '''
    old_nodes = ugraph.keys()
    random_nodes = []
    while old_nodes:
        random_one = random.choice(old_nodes)
        random_nodes.append(random_one)
        old_nodes.remove(random_one)
    return random_nodes
    
def check_undirected_graph(graph):
    '''
    check if a graph is undirected or not
    '''
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if node not in graph[neighbor]:
                return False
    return True


computer_network_p = computer_network_edge*2/computer_network_node/(computer_network_node-1)
computer_network_ER = make_ER_ugraph(computer_network_node, p = computer_network_p)

UPA_m = int(computer_network_edge / computer_network_node)
computer_network_UPA = make_complete_ugraph(UPA_m) 
DPA_trial = UPATrial(UPA_m)
for key in xrange(UPA_m, computer_network_node):
    new_heads = DPA_trial.run_trial(UPA_m)
    computer_network_UPA[key] = set(new_heads)
    for head in new_heads:
        computer_network_UPA[head].add(key)

check_undirected_graph(computer_network_UPA)


remaining_nodes_list = [node for node in reversed(xrange(computer_network_node+1))]
three_resilience = pd.DataFrame(remaining_nodes_list)
three_resilience.columns = ['remaining nodes']
three_resilience['computer network'] = compute_resilience(computer_network, random_order(computer_network))
three_resilience['ER graph p of ' + str(round(computer_network_p, 5))] = compute_resilience(computer_network_ER, random_order(computer_network_ER))
three_resilience['UPA graph m of 2'] = compute_resilience(computer_network_UPA, random_order(computer_network_UPA))

plot_name = bar_with_line(three_resilience, x_axis = 'remaining nodes', limit_x = 10, 
                          bar_list = None, rotation = 0,
                          line_list = ['computer network', 
                          'ER graph p of ' + str(round(computer_network_p, 5)), 
                          'UPA graph m of 2'],
                          line_color = None, line_anchor = (0.99, 0.86), line_ms = 1,
                          anno_number = True, save = True,
                          title = 'Resilience for Three Graphs',
                          xlabel = 'Remaining Nodes',
                          bar_label = '',
                          line_label = 'Resilience'
                          )


def fast_targeted_order(ugraph):
    """
    the faster version Compute a targeted attack order consisting
    of nodes of maximal degree
    
    Returns:
    A list of nodes
    """
    new_graph = copy_graph(ugraph)
    
    degree_sets = defaultdict(lambda: set([]))
    for node, neighbors in new_graph.iteritems():
        degree = len(neighbors)
        degree_sets[degree].add(node) 

    order = []
    for degree in reversed(xrange(len(new_graph))):
        while degree_sets[degree]:
            high_degree_node = degree_sets[degree].pop()
            for neighbor in new_graph[high_degree_node]:
                neighbor_degree = len(new_graph[neighbor])
                degree_sets[neighbor_degree].remove(neighbor)
                degree_sets[neighbor_degree-1].add(neighbor)
            order.append(high_degree_node)
            delete_node(new_graph, high_degree_node)
    return order

# question 3
nodes_size = range(10, 1000, 10)
original_time = []
fast_time = []
for n in nodes_size:
    UPA_m = 5
    temp_UPA = make_complete_ugraph(UPA_m) 
    DPA_trial = UPATrial(UPA_m)
    for key in xrange(UPA_m, UPA_m * n):
        new_heads = DPA_trial.run_trial(UPA_m)
        temp_UPA[key] = set(new_heads)
        for head in new_heads:
            temp_UPA[head].add(key)
    start_time = time.clock()
    targeted_order(temp_UPA)
    end_time = time.clock()
    original_time.append(end_time - start_time)
    start_time = time.clock()
    fast_targeted_order(temp_UPA)
    end_time = time.clock()
    fast_time.append(end_time - start_time)

running_time = pd.DataFrame(nodes_size)
running_time.columns = ['# of nodes']
running_time['provided'] = original_time 
running_time['fast'] = fast_time 

plot_name = bar_with_line(running_time, x_axis = '# of nodes', limit_x = 50,
                          bar_list = None, rotation = 90,
                          line_list = ['provided', 'fast'],
                          line_color = None, line_anchor = (0.35, 0.86), line_ms = 1,
                          anno_number = True, save = True,
                          title = 'Running Time Comparsion in Desktop Python',
                          xlabel = '# of Nodes',
                          bar_label = '',
                          line_label = 'Running Time'
                          )

three_resilience = pd.DataFrame(remaining_nodes_list)
three_resilience.columns = ['remaining nodes']
three_resilience['computer network'] = compute_resilience(computer_network, fast_targeted_order(computer_network))
three_resilience['ER graph p of ' + str(round(computer_network_p, 5))] = compute_resilience(computer_network_ER, fast_targeted_order(computer_network_ER))
three_resilience['UPA graph m of 2'] = compute_resilience(computer_network_UPA, fast_targeted_order(computer_network_UPA))

plot_name = bar_with_line(three_resilience, x_axis = 'remaining nodes', limit_x = 10, 
                          bar_list = None, rotation = 0,
                          line_list = ['computer network', 'ER graph p of 0.00397', 'UPA graph m of 2'],
                          line_color = None, line_anchor = (0.99, 0.86), line_ms = 1,
                          anno_number = True, save = True,
                          title = 'Resilience for Three Graphs in Targed Order',
                          xlabel = 'Remaining Nodes',
                          bar_label = '',
                          line_label = 'Resilience'
                          )

























