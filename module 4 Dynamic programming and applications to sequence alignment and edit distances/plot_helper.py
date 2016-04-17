# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 10:18:01 2015

@author: yliu

This scrit includes some ploting functions
"""
from __future__ import division

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 
import math

import os
import shutil

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import re
from time import time
from datetime import datetime
from datetime import date, timedelta

def ax_formater(ax = None, **kwargs):
    '''
    This function will return a ax with certain formating style
    you can pass folowing 
    title, title_size, xlabel, xlabel_size, ylabel, ylabel_size, xlim, ylim
    ''' 
    if ax == None:
        fig, ax = plt.subplots(figsize=(16, 10))
        return_ax = True
    else:
        return_ax = False
    ax.set_axis_bgcolor('#fffefb')
    ax.tick_params(labelsize = 16)
    ax.grid(b=True, which='major', axis='both', color='#A6A6A6', alpha = 0.85)
    
    def apply_setting(setting_name, setting_func, size_para_name = None, ax = ax, kwargs = kwargs):
        '''
        this is to apply the setting if exist
        '''
        if setting_name in kwargs:
            if size_para_name in kwargs:
                setting_func(kwargs.pop(setting_name), size = kwargs.pop(size_para_name) )
            else:
                setting_func(kwargs.pop(setting_name))
                
    setting_list = [('title', ax.set_title, 'title_size'),
                    ('xlabel', ax.set_xlabel, 'xlabel_size'),
                    ('ylabel', ax.set_ylabel, 'ylabel_size'),
                    ('xlim', ax.set_xlim, None),
                    ('ylim', ax.set_ylim, None)
                   ]
    
    for setting_name, setting_func, size_para_name in setting_list:
        apply_setting(setting_name = setting_name, setting_func = setting_func, size_para_name = size_para_name)
    
    if return_ax:
        return fig, ax

def closed_main(data, color_dict, category, duration, annotate_dict, 
                begin = 'Date Created', end = 'Date Closed', middle = None, 
                log10 = False, 
                save = True, **kwargs):
    '''
    This function will create a plot to show how the duration for all the closed tickets or projects.
    data: pandas dataframe contains all the data
    color_dict: This is the color dict for different category of things
    category: this  is the column name for the cateogry
    annotate_dict: is dict for all the annotate lines to draw, the sla list only consider the first two, first for end, second for middle
    back_days: number of days to go back from to focus
    duration: column name for the duration
    begin: column name for the created date
    end: column name for the cloased date
    middle: optional column name for duration to some period in the middle such as first response or ...
    '''       
    closed = data.copy(deep=True)
    
    closed = closed.sort_values(by = duration, ascending = True).reset_index(drop=True)
    total_closed = closed.shape[0]
    closed['Duration Rank'] = range(total_closed)
    
    if log10 == 'auto':
        max_duration = (closed[duration]).iloc[total_closed-1]
        mid_duration = (closed[duration]).iloc[total_closed//2]
        if (max_duration / mid_duration) > 5:
            log10 = True
        else:
            log10 = False
    
    fig = plt.figure(figsize=(16, 8))
    plt.xlim(-0.5, total_closed)
    if log10:
        closed[duration] = closed[duration].map(lambda x: math.log(x+10, 10))
        plt.ylim(closed[duration].min(), closed[duration].max())  
        if middle != None:
            closed[middle] = closed[middle].map(lambda x: math.log(x+10, 10))
    else:
        plt.ylim(-0.5, int(closed[duration].max()/5+1)*5)
    title = kwargs.pop('title' )
    xlabel = kwargs.pop('xlabel')
    ylabel = kwargs.pop('ylabel')
    if log10:
        ylabel += ' in log10 scale'
    plt.title(title, size = 25)
    plt.xlabel(xlabel, size = 18)
    plt.ylabel(ylabel, size = 18)
    
    ax = plt.gca()
    ax.set_axis_bgcolor('#fffefb')
    ax.tick_params(labelsize = 16)
    ax.grid(b=True, which='major', axis='both', color='#A6A6A6', alpha = 0.85)
    all_category = closed[category].unique()

    for certain_category in all_category:
        color_category = color_dict[certain_category]
        category_rank = closed['Duration Rank'][closed[category] == certain_category]
        category_duration = closed[duration][closed[category] == certain_category]
        ax.scatter(category_rank , category_duration, label = certain_category, color = color_category, 
                   alpha = 0.9, marker = '8', s = 45)
    ax.legend(bbox_to_anchor=(0.05, 0.96), loc=2, borderaxespad=0., prop={'size':16})
    
    if middle != None:
        for certain_category in all_category:
            color = color_dict[certain_category]
            category_rank = closed['Duration Rank'][closed[category] == certain_category]
            category_duration = closed[middle][closed[category] == certain_category]
            ax.scatter(category_rank , category_duration, color = color,
                       alpha = 0.75, marker = r'$\clubsuit$', s = 45)
    
    colors = closed[category].map(color_dict)
    plt.plot(closed['Duration Rank'],  closed[duration], 'g-', alpha = 0.3)
    plt.vlines(closed['Duration Rank'], ymin = np.zeros(total_closed), ymax = closed[duration],
               color = colors, alpha = 0.5)
    
    xmin, xmax = ax.get_xlim()
    x_range = xmax - xmin
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    h_just = total_closed * 0.05    
    
    try:
        detail = annotate_dict['SLA']
        anno_to_added = 0
        for sla_name, sla_unit, sla_time in detail:
            if log10:
                sla_time_use = math.log(sla_time+10, 10)
            else:
                sla_time_use = sla_time
            if anno_to_added == 0: # first sla is for the duration
                if sla_time_use > ymax:
                    ymax = sla_time_use*1.1
                    y_range = ymax - ymin
                    ax.set_ylim(ymin, ymax)
                
                    
                sla_within = (closed[closed[duration] <= sla_time_use]['Duration Rank']).max()
                sla_break = sla_within + 0.5
            elif (anno_to_added == 1) & (middle != None): # second sla is for the middle
                sla_within = closed[closed[middle] <= sla_time_use].shape[0]
                sla_break = sla_break - 6 * h_just
            else:
                break # all the other sla will be ignored
            sla_describe = '{0:.0f}/{1:.0f}({2:.1%}) within {3:.0f} {4} {5} SLA'.format(sla_within, total_closed, sla_within/total_closed, sla_time, sla_unit, sla_name)
            plt.axhline(y = sla_time_use, color = '#1f477e', ls = '--', linewidth = 2, alpha = 0.6)
            ax.annotate(sla_describe, xy=(sla_break, sla_time_use), 
                        arrowprops=dict(facecolor='#D2B584', shrink=0.02, alpha = 0.5),
                        xytext=(0.75*x_range + xmin, (14-anno_to_added)*0.05*y_range + ymin), ha="right", va="bottom", size = 13, 
                        textcoords='data'
                        )
            anno_to_added += 1
    except KeyError:
        pass

    try:
        detail =  annotate_dict['Percentile']
        for index, (unit, q) in enumerate(detail):
            quantile = closed[duration].quantile(q)
            if log10:
                quantile_show = 10**quantile - 10
            else:
                quantile_show = quantile
            quantile_break =  int(total_closed * q)
            quantile_describe = '{0:.1f} {1} {2:.0%} percentile'.format(quantile_show, unit, q)
            plt.axhline(y = quantile, color = '#1f477e', ls = '--', linewidth = 2, alpha = 0.6)
            ax.annotate(quantile_describe, xy=(quantile_break, quantile), 
                        arrowprops=dict(facecolor='#009396', shrink=0.02, alpha = 0.5),
                        xytext=(0.35*x_range + xmin, (7+index)*0.05*y_range + ymin), ha="right", va="bottom", size = 13, 
                        textcoords='data'
                        )
    except KeyError:
        pass          
    #ytick_place = ax.get_yticks()[ax.get_yticks() <= closed[duration].max()]
    ytick_place = ax.get_yticks()[(ax.get_yticks() <= ymax) & (ax.get_yticks() >= ymin)]
    if log10:
        plt.yticks(ytick_place, (10**ytick_place - 10).round(1))
    else:
        y_tick_show = [int_with_commas(tick) for tick in ytick_place]
        plt.yticks(ytick_place, y_tick_show) 
    if save:                    
        fig.savefig(ax.get_title(), dpi = 100, facecolor='w', edgecolor='w', bbox_inches='tight')
        plt.close(fig)
        return ax.get_title()
    else:
        return ax.get_title(), fig, ax

def closed_main_split(data, color_dict, category, duration, annotate_dict, unit_median = 'Hours',
                begin = 'Date Created', end = 'Date Closed', middle = None, 
                log10 = False, 
                save = True, **kwargs):
    '''
    This function will create two closed_main plot and split the data in the middle
    ''' 
    title = kwargs.pop('title' )
    xlabel = kwargs.pop('xlabel')
    ylabel = kwargs.pop('ylabel')
    
    median = data[duration].quantile(0.5)
    left_data = data[data[duration] <= median]
    right_data = data[data[duration] > median]
    
    left_title = '{0} Below Median of {1:.1f} {2}'.format(title, median, unit_median)
    right_title = '{0} Above Median of {1:.1f} {2}'.format(title, median, unit_median)
    
    left_annotate = {}
    right_annotate = {}
    for key, detail in annotate_dict.items():
        if key == 'Percentile':
            left_annotate['Percentile'] = [('Hours', 0.5)]
            right_annotate['Percentile'] = [('Hours', 0.5)]
        elif key == 'SLA':
            for sla_type, sla_unit, sla_time in detail:
                if sla_time < median:
                    if 'SLA' in left_annotate:
                        left_annotate['SLA'].append((sla_type, sla_unit, sla_time))
                    else:
                        left_annotate['SLA'] = [(sla_type, sla_unit, sla_time)]
                else:
                    if 'SLA' in right_annotate:
                        right_annotate['SLA'].append((sla_type, sla_unit, sla_time))
                        
                    else:
                        right_annotate['SLA'] = [(sla_type, sla_unit, sla_time)]
    
    left_name, left_fig, left_ax = closed_main(data = left_data, color_dict = color_dict, category = category, 
                                     duration = duration, annotate_dict = left_annotate, 
                                     begin = begin, end = end, middle = middle, 
                                     log10 = False, 
                                     save = False, title = left_title, xlabel = xlabel, ylabel = ylabel)
                
    right_name, right_fig, right_ax = closed_main(data = right_data, color_dict = color_dict, category = category, 
                                      duration = duration, annotate_dict = right_annotate, 
                                      begin = begin, end = end, middle = middle, 
                                      log10 = log10, 
                                      save = False, title = right_title, xlabel = xlabel,  ylabel = ylabel)
    if save:
        left_fig.savefig(left_name + '.png', dpi = 100, facecolor='w', edgecolor='w', bbox_inches='tight')
        plt.close(left_fig)
        right_fig.savefig(right_name + '.png', dpi = 100, facecolor='w', edgecolor='w', bbox_inches='tight')
        plt.close(right_fig)        
        return {'left': left_name, 'right': right_name}
    else:
        return {'left': (left_name, left_fig, left_ax), 'right': (right_name, right_fig, right_ax)}

def closed_pie(data, color_dict, category, duration, sla = 48, goal = 0.95, save = True, **kwargs):
    '''
    This function will create a pir chart for all the closed tickets or projects.
    data: pandas dataframe contains all the data
    color_dict: This is the color dict for different category of things
    category: this  is the column name for the cateogry
    duration: column name for the duration
    sla: the sla dictionary with key of category or scalar
    goal: the goal percentage for the within sla rate
    ''' 
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_axes([0, 0, 1, 1], polar=True)
    
    title  = kwargs.pop( 'title'  )
    ax.set_title(title, size = 18)
    
    data = data.copy(deep=True)    
    
    N_category = len(data[category].unique())
    total_ticket = data.shape[0]
    if isinstance(sla, (int, long, float, complex)):
        category_within_sla = (data[category][data[duration] <= sla]).value_counts().reset_index(drop=False)
    elif isinstance(sla, dict):
        data_grouped = data.groupby(category)[duration]
        within_sla_dict = {}
        for certain_category, all_durations in data_grouped:
            within_sla_dict[certain_category] = (all_durations[all_durations <= sla[certain_category]]).size
        category_within_sla = pd.Series( within_sla_dict).reset_index(drop=False)    
    category_total = data[category].value_counts().reset_index(drop=False)
    category_total_within =  category_total.merge(category_within_sla, how = 'left', on = ['index'])
    category_total_within.columns = ['category', 'total', 'within']
    category_total_within.fillna(0, inplace = True)
    # to put total in big samll alternating order, to avoid overlap

    index_old = list(category_total_within.index)
    order_old = list(category_total_within.index)
    index_new = []
    for order in order_old:
        if (order % 2) == 0:
            index_new.append(index_old.pop(0))
        else:
            index_new.append(index_old.pop(-1))
    category_total_within = (category_total_within.T[index_new]).T

    width = category_total_within['total'] / total_ticket * 2 * np.pi
    theta = np.hstack((np.zeros(1), np.cumsum(width[:-1])))
    
    radii_out = np.ones(N_category)*100
    bars_out = ax.bar(left=theta, height=radii_out, width=width)
    for bar in bars_out:
        certain_category = category_total_within['category'].iloc[abs(theta - bar.get_x()).argmin()] 
        bar.set_facecolor(color_dict[certain_category])
        bar.set_alpha(0.2)
    
    radii_in = category_total_within['within'] / category_total_within['total'] * 100
    bars_in = ax.bar(left=theta, height=radii_in, width=width)
    for bar in bars_in:
        certain_category = category_total_within['category'].iloc[abs(theta - bar.get_x()).argmin()] 
        bar.set_facecolor(color_dict[certain_category])
        bar.set_alpha(0.8)

    ax.plot(np.linspace(0, 2*np.pi, 1000), np.ones(1000)*100*goal, color='r', linestyle='--', alpha = 0.7)

    anno_x = theta + width / 2
    anno_y_text = radii_out * 0.7
    anno_y_arrow = radii_in

    sla_percent = (radii_in / 100).map(lambda ratio: "{0:.1%}".format(ratio ) )
    anno_text_describe = category_total_within['within'].astype(int).astype(str) + '/' + category_total_within['total'].astype(int).astype(str) + '(' + sla_percent + ')'
    anno_text = category_total_within['category'] + '\n' + anno_text_describe
    for x, y_text, y_arrow, text in zip(anno_x, anno_y_text, anno_y_arrow, anno_text):
        ax.annotate(text, xy=(x, y_arrow), ha="center", va="center",size = 12, 
                    alpha = kwargs.get('anno_alpha', 0.85),
                    xytext=(x, y_text),   
                    textcoords='data', 
                    arrowprops=dict(facecolor='#1f477e', shrink=0.03, alpha = 0.6)
                    )

    plt.tick_params(labelsize = 12)
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.grid(b=True, which='major', axis='both', color='#A6A6A6', alpha = 0.6)
    
    if save:
        fig.savefig(ax.get_title(), dpi = 100, facecolor='w', edgecolor='w', bbox_inches='tight')
        plt.close(fig)
        return ax.get_title()
    else:  
        return ax.get_title(), fig, ax

def closed_weekly(file_list, color_dict, category, duration, sla = 48, sla_unit = 'Hours',
                  save = True, **kwargs):
    '''
    This function will create a plot to show the weekly trend of within or out sla
    file_list: a file manager object to certian files
    sla: the sla dictionary with key of category or scalar
    '''
    n = len(file_list)
    weeks = []
    unit = 1. / (n+1) 
    ytick_week = []    
    
    for index, file_name in enumerate(file_list.walk_files()):
        week_date = file_list.get_core_name(file_name)
        weeks.append(week_date)
        closed = pd.read_csv(file_name, encoding='utf-8')
        if isinstance(sla, (int, long, float, complex)):
            closed_within_sla = (closed[category][closed[duration] <= sla]).value_counts().reset_index(drop=False)
        elif isinstance(sla, dict):
            data_grouped = closed.groupby(category)[duration]
            within_sla_dict = {}
            for certain_category, all_durations in data_grouped:
                within_sla_dict[certain_category] = (all_durations[all_durations <= sla[certain_category]]).size
            closed_within_sla = pd.Series(within_sla_dict).reset_index(drop=False)    
        closed_total = closed[category].value_counts().reset_index(drop=False)
        closed_total_within =  closed_total.merge(closed_within_sla, how = 'left', on = ['index'])
        closed_total_within.columns = ['category', week_date + ' total tickets', week_date + ' within sla']
        if index == 0:
            closed_all_weeks = closed_total_within.copy(deep=True)
        if index > 0:
                closed_all_weeks = closed_all_weeks.merge(closed_total_within, how = 'outer', on = ['category'])
    closed_all_weeks.fillna(0)
    closed_all_weeks_melted = pd.melt(closed_all_weeks, id_vars=['category'], var_name='type', value_name='count')
    closed_all_weeks_melted.sort_values(by = ['category', 'type'], ascending=[False, True], inplace = True)
    closed_all_weeks_melted_total = closed_all_weeks_melted[closed_all_weeks_melted['type'].map(lambda x: x[-1]) == 's'].reset_index(drop=True)
    closed_all_weeks_melted_sla = closed_all_weeks_melted[closed_all_weeks_melted['type'].map(lambda x: x[-1]) == 'a'].reset_index(drop=True)
    closed_all_weeks_melted_out_sla = closed_all_weeks_melted_total['count'] - closed_all_weeks_melted_sla['count']
    
    categories = closed_all_weeks_melted['category'].unique()
    n_categories = len(categories)

    fig = plt.figure(figsize=(16, 8))
    title = kwargs.pop('title' )
    xlabel = kwargs.pop('xlabel')
    ylabel = kwargs.pop('ylabel')
    plt.title(title, size = 25)
    plt.xlabel(xlabel, size = 18)
    plt.ylabel(ylabel, size = 18)
    plt.ylim(ymin = 0.5, ymax = n_categories + 0.5)
    ax = plt.gca()
    ax.set_axis_bgcolor('#fffefb')
    ax.tick_params(labelsize = 16)
    ax.grid(b=True, which='major', axis='x', color='#A6A6A6', alpha = 0.7)
    ax.grid(axis='y', alpha = 0.01)

    for index, certain_category in enumerate(categories):
        certain_category_only = closed_all_weeks_melted_total[closed_all_weeks_melted_total['category'] == certain_category]['count']
        certain_category_ybar_bottom = np.linspace(n_categories - index + unit*(n/2-1), n_categories - index - unit*n/2, n)
        ax.barh(bottom = certain_category_ybar_bottom,
                width = certain_category_only, 
                height = unit,
                label = certain_category,
                alpha=0.75, 
                color = color_dict[certain_category])
        ytick_week.extend(certain_category_ybar_bottom  + unit/2) # make the ticks to be the middle of each bar
    ax.legend(loc='upper left', bbox_to_anchor=kwargs.get('legend_anchor', (1.02, 0.45)), 
              borderaxespad=0., fancybox=True, shadow=True, prop={'size':16})    
    
    for index, out in enumerate(closed_all_weeks_melted_out_sla):
        if out > 0:
            total = closed_all_weeks_melted_total.iloc[index]['count']
            if isinstance(sla, (int, long, float, complex)):
                out_describe = ' {0:.0f}/{1:.0f}({2:.1%}) {3} {4} SLA'.format(out, total, out/total, sla, sla_unit)
            elif isinstance(sla, dict):
                sla_from_dict = sla[closed_all_weeks_melted_total['category'].iloc[index]]
                out_describe = ' {0:.0f}/{1:.0f}({2:.1%}) {3} {4} SLA'.format(out, total, out/total, sla_from_dict, sla_unit)
            ax.annotate(out_describe, 
                        xy=(total - out, ytick_week[index]), 
                        arrowprops=dict(facecolor='#E5005A', shrink=0.02, alpha = 0.9),
                        xytext=(total, ytick_week[index]), 
                        ha='left', va='center', size = 12, textcoords='data')

    plt.yticks(ytick_week, weeks*n_categories)
    if save:                    
        plt.savefig(ax.get_title(), dpi = 100, facecolor='w', edgecolor='w', bbox_inches='tight')
        plt.close(fig)
        return ax.get_title()
    else:
        return ax.get_title(), fig, ax

def aging_hbar(file_list,  color_dict, aging_range, aging_range_to_show, 
               category = 'all', specific_category = 'None',  save = True, **kwargs):
    '''
    this function will create a horizontal bar chart for the aging tickets or projects
    file_list: filemanger of all the file name to plot, each bar for each file
    color_dict: This is the color dict for different aging range
    aging_range: colname of the aging name
    aging_range_to_show: select the duration range to show
    category: 'all' will include all the category form the local files and will ifnore the value in specific_category
    specific_category: will only be used when category is the colname of the category instead of 'all'
    ''' 
    weeks = [] # usually it is weekly thing but can be whatever the period you want
    periods = len(file_list)
    
    fig = plt.figure(figsize=(16, 8))
    plt.ylim(ymin = -0.5, ymax = periods - 0.5) # plot start at y = 0 so periodes -1 + 0.5
    
    title = kwargs.pop('title')
    xlabel = kwargs.pop('xlabel')
    ylabel = kwargs.pop('ylabel')
    if category == 'all':
        title += ' All Categories'
    else:
        title += ' ' + specific_category
    plt.title(title, size = 25)
    plt.xlabel(xlabel, size = 18)
    plt.ylabel(ylabel, size = 18)
    
    ax = plt.gca()
    ax.set_axis_bgcolor('#fffefb')
    ax.tick_params(labelsize = 16)
    ax.grid(b=True, which='major', axis='both', color='#A6A6A6', alpha = 0.75)
    
    for index, file_name in enumerate(file_list.walk_files()):
        weeks.append(file_list.get_core_name(file_name))
        open_before_break = pd.read_csv(file_name, encoding='utf-8')
        if category != 'all':
            open_before_break = open_before_break[open_before_break[category] == specific_category]
        open_before_break_ranged =open_before_break.groupby([aging_range]).size()
        left = 0.
        for open_range in aging_range_to_show:
            try:
                width = open_before_break_ranged[open_range]
            except:
                width = 0.
            ax.barh(bottom = index-0.3, width = width, height = 0.6,left = left,
                    label = open_range, alpha=0.75, color = color_dict[open_range])
            left += width
            
        if index == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 0.76), 
                      borderaxespad=0., fancybox=True, shadow=True, prop={'size':16})
    
    plt.yticks(range(periods), weeks)
    if save:
        fig.savefig(ax.get_title(), dpi = 100, facecolor='w', edgecolor='w', bbox_inches='tight')
        plt.close(fig)
        return ax.get_title()
    else:
        return ax.get_title(), fig, ax

def bar_with_line(data, x_axis, hbar = False,
                  bar_list = None, bar_color = None, bar_legend_show = True,
                  line_list = None, line_color = None, line_legend_show = True,
                  anno_number = False, limit_x = False,
                  save = True, **kwargs):
    '''
    x_axis: name for the columns containing the labels of the x axis
    bar_list: list of names for the (stacked) bars or just string for one column name
    bar_color, line_color: color dictionary 
    line_list: list of names for the lines or just string for one column name
    anno_number: is show the number at the bars or not
    kwargs could contain: title, xlabel, bar_label, line_label, bar_alpha, line_alpha, bar_anchor, line_anchor rotation
    '''
    ax = {}
    
    x_labels = data[x_axis]
    x_n = len(x_labels)
    
    fig = plt.figure(figsize=(16, 8))
    if hbar:
        plt.yticks(rotation = kwargs.get('rotation', 35))
    else:
        plt.xticks(rotation = kwargs.get('rotation', 35))
    
    title = kwargs.get('title', 'Stacked Bars with Lines')
    xlabel = kwargs.get('xlabel', 'xlabel')
    bar_label = kwargs.get('bar_label', 'bar_label')
    line_label = kwargs.get('line_label', 'line_label')

    plt.title(title, size = 25)
    ax_bar = plt.gca()
    ax_bar.set_axis_bgcolor('#fffefb')
    if hbar:
        ax_bar.set_ylabel(xlabel, size = 18)
    else:
        ax_bar.set_xlabel(xlabel, size = 18)    
        
    if bar_list == None:
        if hbar:
            ax_bar.xaxis.set_major_formatter(plt.NullFormatter())
        else:
            ax_bar.yaxis.set_major_formatter(plt.NullFormatter())  
    else:
        if hbar:
            ax_bar.set_xlabel(bar_label, size = 18)
            ax_bar.set_ylim([-0.5, x_n-0.5])
        else:
            ax_bar.set_ylabel(bar_label, size = 18)
            ax_bar.set_xlim([-0.5, x_n-0.5]) 
        if bar_color == None:
            colors = cm.rainbow(np.linspace(0, 1, len(bar_list)))
            bar_color = {bar_name: color for bar_name, color in zip(bar_list, colors)}
        ax_bar.tick_params(labelsize = 16)
        ax['bar'] = ax_bar
        for index, x_label in enumerate(x_labels):
            that_label = pd.Series(data.iloc[index])
            bottom = 0
            if isinstance(bar_list, str):
                bar_list = [bar_list]
            for bar_index, stacked_bar in enumerate(bar_list):
                height = that_label[stacked_bar]
                if hbar:
                    ax_bar.barh(index-0.3, height, 0.6, bottom,  label = stacked_bar,
                                alpha=kwargs.get('bar_alpha', 0.75), color = bar_color[stacked_bar])
                else:
                    ax_bar.bar(index-0.3, height, 0.6, bottom,  label = stacked_bar,
                               alpha=kwargs.get('bar_alpha', 0.75), color = bar_color[stacked_bar])
                if anno_number:
                    if hbar:
                        h_just = -0.23 if bar_index%2 == 0 else 0.1
                        ax_bar.text(bottom + height/3, index + h_just, str(height), ha='center', va='bottom')
                    else:
                        ax_bar.text(index, bottom + height/3, str(height), ha='center', va='bottom')
                bottom += height
            if (index == 0) & bar_legend_show:
                bar_legend = ax_bar.legend(bbox_to_anchor=kwargs.get('bar_anchor', (0.01, -0.1)), loc=2, 
                                           borderaxespad=0., prop={'size':15})
                bar_legend.get_frame().set_alpha(0.6)
    if line_list != None:
        if hbar:
            ax_line = ax_bar.twiny()
            ax_line.set_ylim([-0.5, x_n-0.5])
            ax_line.set_xlabel(line_label, size = 18)
        else:
            ax_line = ax_bar.twinx()
            ax_line.set_xlim([-0.5, x_n-0.5])
            ax_line.set_ylabel(line_label, size = 18)
        ax_line.tick_params(labelsize = 16)
        ax['line'] = ax_line
        if isinstance(line_list, str):
            line_list = [line_list]
        for line_name in line_list:
            if hbar:
                x_y = (data[line_name], range(x_n))
            else:
                x_y = (range(x_n), data[line_name])
            if line_color == None:
                ax_line.plot(x_y[0], x_y[1], lw = 3, marker = 'D', ms = kwargs.get('line_ms', 10),
                             label = line_name, alpha = kwargs.get('line_alpha', 0.75))
            else:
                ax_line.plot(x_y[0], x_y[1], c = line_color[line_name], lw = 3, marker = 'D', ms = kwargs.get('line_ms', 10),
                             label = line_name, alpha = kwargs.get('line_alpha', 0.75))
        if hbar & line_legend_show:
            line_legend = ax_line.legend(bbox_to_anchor=kwargs.get('line_anchor', (0.01, 1.1)), loc=6, 
                                         borderaxespad=0., prop={'size':15})
        elif (not hbar) & line_legend_show:
            line_legend = ax_line.legend(bbox_to_anchor=kwargs.get('line_anchor', (0.99, -0.1)), loc=1, 
                                         borderaxespad=0., prop={'size':15})
        if line_legend_show: 
            line_legend.get_frame().set_alpha(0.6)
    x_positions = pd.Series(range(x_n))
    if limit_x:
        while x_n > limit_x:
            x_limited_to = map(lambda x: bool(x%2) , range(x_n))
            x_labels = x_labels[x_limited_to]
            x_positions = x_positions[x_limited_to]
            x_n = len(x_labels)
    if hbar:
        plt.yticks(x_positions, x_labels)
    else:
        plt.xticks(x_positions, x_labels)
    plt.grid(b=True, which='major', axis='both', color='#A6A6A6', alpha = 0.75) 

    if save:                    
        fig.savefig(ax_bar.get_title(), dpi = 100, facecolor='w', edgecolor='w', bbox_inches='tight')
        plt.close(fig)
        return ax_bar.get_title()
    else:
        return ax_bar.get_title(), fig, ax
        
        