'''
This script is some basic implementation for the citation
'''
from __future__ import devision

from M1_Project_Graphs_and_brute_force_algorithms import make_complete_graph, compute_in_degrees, in_degree_distribution



"""
Provided code for Application portion of Module 1

Imports physics citation graph 
"""
# general imports
import urllib2
# Set timeout for CodeSkulptor if necessary
#import codeskulptor
#codeskulptor.set_timeout(20)

###################################
# Code for loading citation graph

CITATION_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_phys-cite.txt"

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
citation_graph = load_graph(CITATION_URL)

#############################
## Question 1
citation_in_dist = in_degree_distribution(citation_graph)

total_in = sum(citation_in_dist.values())

for key, value in citation_in_dist.items():
    citation_in_dist[key] = float(value) / total_in

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16, 8))
title = 'Citation In Degree Distribution in loglog'
xlabel = 'In Degree'
ylabel = 'Normailized In Degree Frequency'

plt.title(title, size = 25)
plt.xlabel(xlabel, size = 18)
plt.ylabel(ylabel, size = 18)

ax = plt.gca()
ax.set_axis_bgcolor('#fffefb')
ax.tick_params(labelsize = 16)
ax.grid(b=True, which='major', axis='both', color='#A6A6A6', alpha = 0.85)

ax.loglog(citation_in_dist.keys(), citation_in_dist.values(), mfc='g', alpha = 0.65, ls = '', marker = 'o')

fig.savefig(ax.get_title(), facecolor=fig.get_facecolor(), edgecolor='w', bbox_inches='tight')
plt.close(fig)

 
#############################
## Question 2
import random

def make_ER_random_graph(num_nodes, p = 0.5):
    '''
    Takes the number of nodes num_nodes and returns a 
    dictionary corresponding to a ramdon directed graph 
    with the specified number of nodes
    implement the ER algorith in HW 10
    '''
    graph_dict = {}
    for key in xrange(num_nodes):
        graph_dict[key] = set([])
        other_notes = range(key) + range(key+1, num_nodes)
        for node in other_notes:
            if p < random.random():
                graph_dict[key].add(node)
    return graph_dict

total_nodes = 1000

for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
    random_graph = make_ER_random_graph(total_nodes, p = p)
    random_in_dist = in_degree_distribution(random_graph)
    total_in = sum(random_in_dist.values())
    for key, value in random_in_dist.items():
        random_in_dist[key] = float(value) / total_in
    fig = plt.figure(figsize=(16, 8))
    title = 'Random In Degree Distribution with p of ' + str(p) + ' in loglog'
    xlabel = 'In Degree'
    ylabel = 'Normailized In Degree Frequency'
    plt.title(title, size = 25)
    plt.xlabel(xlabel, size = 18)
    plt.ylabel(ylabel, size = 18)
    ax = plt.gca()
    ax.set_axis_bgcolor('#fffefb')
    ax.tick_params(labelsize = 16)
    ax.grid(b=True, which='major', axis='both', color='#A6A6A6', alpha = 0.85)
    ax.loglog(random_in_dist.keys(), random_in_dist.values(), mfc='g', alpha = 0.65, ls = '', marker = 'o')
    fig.savefig(ax.get_title()+'.png', facecolor=fig.get_facecolor(), edgecolor='w', bbox_inches='tight')
    plt.close(fig)


#############################
## Question 3
import numpy as np

def make_DPA_random_graph(num_nodes, initial_nodes):
    '''
    Takes the number of nodes num_nodes and returns a 
    dictionary corresponding to a ramdon directed graph 
    with the specified number of nodes
    implement the DPA algorith in Application 3
    '''
    graph_dict = make_complete_graph(initial_nodes)
    for index, key in enumerate(xrange(initial_nodes, num_nodes)):
        in_degree = compute_in_degrees(graph_dict)
        total_in_degree = sum(in_degree.values())
        choose_p = (np.array(in_degree.values(), dtype = float) + 1) / (total_in_degree + index + initial_nodes)
        choose_key = np.random.choice(in_degree.keys(), size=initial_nodes, replace=False, p=choose_p)
        graph_dict[key] = set(choose_key)
    return graph_dict


from DPATrial import DPATrial



total_citation_nodes = len(citation_graph)
mean_citation_out_degree = np.mean(map(len, citation_graph.values()))

DPA_graph = make_DPA_random_graph(num_nodes = 27770, initial_nodes = 13)

DPA_in_dist = in_degree_distribution(citation_graph)
total_in = sum(DPA_in_dist.values())

for key, value in DPA_in_dist.items():
    DPA_in_dist[key] = float(value) / total_in

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16, 8))
title = 'DPA In Degree Distribution in loglog'
xlabel = 'In Degree'
ylabel = 'Normailized In Degree Frequency'

plt.title(title, size = 25)
plt.xlabel(xlabel, size = 18)
plt.ylabel(ylabel, size = 18)

ax = plt.gca()
ax.set_axis_bgcolor('#fffefb')
ax.tick_params(labelsize = 16)
ax.grid(b=True, which='major', axis='both', color='#A6A6A6', alpha = 0.85)

ax.loglog(DPA_in_dist.keys(), DPA_in_dist.values(), mfc='g', alpha = 0.65, ls = '', marker = 'o')

fig.savefig(ax.get_title(), facecolor=fig.get_facecolor(), edgecolor='w', bbox_inches='tight')
plt.close(fig)












