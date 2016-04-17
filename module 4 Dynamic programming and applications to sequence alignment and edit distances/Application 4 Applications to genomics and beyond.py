# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 00:58:50 2016

@author: Yi
"""

########################################
## Application 4
########################################
from __future__ import division
from provide_code import *
from project_compute_alignments_of_sequences import *

# Question 1
amino_score = read_scoring_matrix(PAM50_URL)

human_eyeless = read_protein(HUMAN_EYELESS_URL)
fly_eyeless = read_protein(FRUITFLY_EYELESS_URL )

eyeless_align_matrix = compute_alignment_matrix(human_eyeless, fly_eyeless, amino_score, False)
eyeless_align_detail = compute_local_alignment(human_eyeless, fly_eyeless, amino_score, eyeless_align_matrix)

print 'alignmnet score is {0}\n'.format(eyeless_align_detail[0])
print 'human eyeless alignment is below\n\t{0}\n'.format(eyeless_align_detail[1])
print 'fruit fly eyeless alignment is below\n\t{0}'.format(eyeless_align_detail[2])

consensus = read_protein(CONSENSUS_PAX_URL)

human_align = eyeless_align_detail[1].replace('-', '')
fly_align = eyeless_align_detail[2].replace('-', '')

def agree_ratio(seq_x, seq_y):
    '''
    compute aggre ratio for two string
    '''
    agree_count = sum(map(lambda (x, y): x==y, zip(seq_x, seq_y))) 
    total = len(seq_x)   
    return float(agree_count) / total


consensus_human_global_matrix = compute_alignment_matrix(consensus, human_align, amino_score, True)
consensus_human_global_detail = compute_global_alignment(consensus, human_align, amino_score, consensus_human_global_matrix)
print 'global alignment agree ratio between consensus and human is {0:.3f}'.format(agree_ratio(consensus_human_global_detail[1], consensus_human_global_detail[2]))

consensus_fly_global_matrix = compute_alignment_matrix(consensus, fly_align, amino_score, True)
consensus_fly_global_detail = compute_global_alignment(consensus, fly_align, amino_score, consensus_fly_global_matrix)
print 'global alignment agree ratio between consensus and fly is {0:.3f}'.format(agree_ratio(consensus_fly_global_detail[1], consensus_fly_global_detail[2]))


consensus_human_local_matrix = compute_alignment_matrix(consensus, human_align, amino_score, False)
consensus_human_local_detail = compute_local_alignment(consensus, human_align, amino_score, consensus_human_local_matrix)
print 'local alignment agree ratio between consensus and human is {0:.3f}'.format(agree_ratio(consensus_human_local_detail[1], consensus_human_local_detail[2]))

consensus_fly_local_matrix = compute_alignment_matrix(consensus, fly_align, amino_score, False)
consensus_fly_local_detail = compute_local_alignment(consensus, fly_align, amino_score, consensus_fly_local_matrix)
print 'local alignment agree ratio between consensus and fly is {0:.3f}'.format(agree_ratio(consensus_fly_local_detail[1], consensus_fly_local_detail[2]))


# Question 3
import random
import matplotlib.pyplot as plt
from plot_helper import ax_formater
from collections import defaultdict


def generate_null_distribution(seq_x, seq_y, scoring_matrix, num_trials):
    '''
    function described in question 3
    '''
    score_dist = defaultdict(lambda :0)
    for _ in range(num_trials):
        rand_y = list(seq_y)
        random.shuffle(rand_y)
        local_matrix = compute_alignment_matrix(seq_x, rand_y,  scoring_matrix, False)
        score = max([max(column) for column in local_matrix])
        score_dist[score] += 1
    return dict(score_dist)

score_dist = generate_null_distribution(human_eyeless, fly_eyeless, amino_score, 1000)


fig, ax = ax_formater(title = 'Normalized Score Distribution Local Alignment Human & Permuted Fly Eyeless',
                      xlabel = 'Score Of Local Alignment',
                      ylabel = 'Fraction Of Total Trials Corresponding To Each Score',
                      title_size = 25, xlabel_size = 15, ylabel_size = 15)

score_dist_normalized = {s: float(count)/1000 for s, count in score_dist.items()}

ax.bar(score_dist_normalized.keys(), score_dist_normalized.values(), 0.6, color = 'g', alpha= 0.75)
fig.savefig(ax.get_title(), layout = 'tight')
plt.close(fig)

# Question 5
import numpy as np
import scipy as sp

all_samples = np.array(score_dist.keys()).repeat(score_dist.values())

sample_mean = sp.mean(all_samples)
sample_std = sp.std(all_samples)

human_fly_local_matrix = compute_alignment_matrix(human_eyeless, fly_eyeless, amino_score, False)
human_fly_score = max([max(column) for column in human_fly_local_matrix])
z_score = (human_fly_score - sample_mean) / sample_std

print 'mu = {0:.3f}, sigma = {1: .3f} and Z = {2:.3f}'.format(sample_mean, sample_std, z_score)

# Question 7

print 'diag_score = {0}, off_diag_score = {1}, and dash_score = {0}'.format(2, 1, 0)

edit_score = build_scoring_matrix(alphabet, 2,1,0)

word_list = read_words(WORD_LIST_URL)

def check_spelling(checked_word, dist, word_list):
    '''
    that iterates through word_list and returns the set of all words that
    are within edit distance dist of the string checked_word
    '''
    near_set = set([])
    alphabet = set('a b c d e f g h i j k l m n o p q r s t u v w x y z'.split())
    edit_score = build_scoring_matrix(alphabet, 2,1,0)
    check_len = len(checked_word)
    for word in word_list:
        global_matrix = compute_alignment_matrix(checked_word, word, edit_score, True)
        edit_dist = check_len + len(word) - global_matrix[-1][-1]
        if edit_dist <= dist:
            near_set.add(word)
    return near_set
        
humble_1 = check_spelling('humble', 1, word_list)

firefly_2 = check_spelling('firefly', 2, word_list)

print humble_1

print firefly_2









