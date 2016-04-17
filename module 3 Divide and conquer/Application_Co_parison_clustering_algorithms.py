# -*- coding: utf-8 -*-
"""
Created on Fri Mar 04 19:10:44 2016

@author: Yi
"""
import urllib2
import random
import time
import math
import matplotlib.pyplot as plt

from alg_cluster import Cluster
from project_closest_pairs_and_clustering_algorithms import *

from plot_helper import ax_formater

###############
#Q1

def gen_random_clusters(num_clusters): 
    '''
    that creates a list of clusters where each cluster in 
    this list corresponds to one randomly generated point 
    in the square with corners (±1,±1)
    '''
    clusters = []
    get_random = lambda : random.random()*2-1
    for _ in xrange(num_clusters):
        clusters.append(Cluster(set([]),
                                get_random(),
                                get_random(),
                                1, 0))
    return clusters

slow_time, fast_time = [], []
num_clusters_list = range(2, 201)
for num_clusters in num_clusters_list:
    clusters = gen_random_clusters(num_clusters)
    begin = time.time()
    slow_closest_pair(clusters)
    end = time.time()
    slow_time.append(end-begin)
    begin = time.time()
    fast_closest_pair(clusters)
    end = time.time()
    fast_time.append(end-begin)

fig, ax = ax_formater(title = 'Running Time Comparsion',
                      xlabel = 'Cluster Size 2-200',
                      ylabel = 'Runing Time in Sec',
                      title_size = 25, xlabel_size = 15, ylabel_size = 15)
ax.plot(num_clusters_list, slow_time, label = 'Slow Closest Pair', lw =2)
ax.plot(num_clusters_list, fast_time, label = 'Fast Closest Pair', lw =2)
ax.legend(loc = 2, bbox_to_anchor = (0.08, 0.95), prop={'size':16})
fig.savefig(ax.get_title(), layout = 'tight', dpi = 200)
plt.close(fig)
      
      
###############
#Q7-10
from alg_project3_viz import load_data_table, DATA_3108_URL, DATA_111_URL, DATA_290_URL, DATA_896_URL
import project_closest_pairs_and_clustering_algorithms as alg_project3_solution

def compute_distortion(cluster_list, original_data):
    '''
    distortion(L)=∑C∈Lerror(C)
    takes a list of clusters and uses cluster_error to compute its distortion
    '''
    return sum(map(lambda c: c.cluster_error(original_data), cluster_list))
    
def run_example():
    """
    Modified to do question 7 showing distoration
    Load a data table, compute a list of clusters and 
    plot a list of clusters

    Set DESKTOP = True/False to use either matplotlib or simplegui
    """
    # DATA_3108_URL DATA_111_URL DATA_290_URL
    data_table = load_data_table(DATA_111_URL) 
    
    singleton_list = []
    for line in data_table:
        singleton_list.append(alg_cluster.Cluster(set([line[0]]), line[1], line[2], line[3], line[4]))
        
    print "___________"
    cluster_list = alg_project3_solution.hierarchical_clustering(singleton_list, 9)
    print "Displaying", len(cluster_list), "hierarchical clusters"
    print "with Distoration of {0}".format(compute_distortion(cluster_list, data_table))
    
    print "___________"
    cluster_list = alg_project3_solution.kmeans_clustering(singleton_list, 9, 5)	
    print "Displaying", len(cluster_list), "k-means clusters"
    print "with Distoration of {0}".format(compute_distortion(cluster_list, data_table))

run_example()

def run_example(data_dir,  num_clusters):
    """
    Modified to do question 10 loops and save time plot
    Load a data table, compute a list of clusters and 
    plot a list of clusters

    Set DESKTOP = True/False to use either matplotlib or simplegui
    """
    # DATA_3108_URL DATA_111_URL DATA_290_URL
    data_table = load_data_table(data_dir) 
    
    singleton_list = []
    for line in data_table:
        singleton_list.append(alg_cluster.Cluster(set([line[0]]), line[1], line[2], line[3], line[4]))
        
    begin = time.time()
    cluster_list = alg_project3_solution.hierarchical_clustering(singleton_list, num_clusters)
    end = time.time()
    hierarchical_dur = end - begin
    hierarchical_dist = compute_distortion(cluster_list, data_table) * 10e10
    
    begin = time.time()
    cluster_list = alg_project3_solution.kmeans_clustering(singleton_list, num_clusters, 5)
    end = time.time()
    kmeans_dur = end - begin
    kmeans_dist = compute_distortion(cluster_list, data_table) * 10e10
    
    return hierarchical_dur, hierarchical_dist, kmeans_dur, kmeans_dist


fig, axes = plt.subplots(3, 2, figsize=(16*2.1, 10*3.2))
num_clusters_list = range(20, 5, -1)
data_set_name = ['DATA_111', 'DATA_290', 'DATA_896']
for row_index, data_dir in enumerate([DATA_111_URL, DATA_290_URL, DATA_896_URL]):
    h_time, h_dist, k_time, k_dist = [], [], [], []
    for num_clusters in num_clusters_list:
        h_k_tuple = run_example(data_dir, num_clusters)
        for index, list_plot in enumerate([h_time, h_dist, k_time, k_dist]):
            list_plot.append(h_k_tuple[index])
            
    ax_formater(ax = axes[row_index, 0],
                title = 'Running Time Comparsion Using Data Set {}'.format(data_set_name[row_index]),
                xlabel = 'Cluster Size {0}-{1}'.format(min(num_clusters_list), max(num_clusters_list)),
                ylabel = 'Runing Time in Sec',
                title_size = 25, xlabel_size = 15, ylabel_size = 15)
    axes[row_index, 0].plot(num_clusters_list, h_time, label = 'Hierarchical', lw =2)
    axes[row_index, 0].plot(num_clusters_list, k_time, label = 'K Means', lw =2)
    axes[row_index, 0].legend(loc = 2, bbox_to_anchor = (0.08, 0.95), prop={'size':16})

    ax_formater(ax = axes[row_index, 1],
                title = 'Distortion Comparsion Using Data Set {}'.format(data_set_name[row_index]),
                xlabel = 'Cluster Size {0}-{1}'.format(min(num_clusters_list), max(num_clusters_list)),
                ylabel = 'Distoration in $10^10$',
                title_size = 25, xlabel_size = 15, ylabel_size = 15)
    axes[row_index, 1].plot(num_clusters_list, h_dist, label = 'Hierarchical', lw =2)
    axes[row_index, 1].plot(num_clusters_list, k_dist, label = 'K Means', lw =2)
    axes[row_index, 1].legend(loc = 2, bbox_to_anchor = (0.08, 0.95), prop={'size':16}) 
   
fig.savefig('Distoration and Running Time Comparsion', layout = 'tight')
plt.close(fig)
