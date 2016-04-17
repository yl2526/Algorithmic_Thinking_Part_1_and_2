# -*- coding: utf-8 -*-
"""
This script contains all the formating functions

@author: yliu
"""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from datetime import date, datetime

def seperate_underline_issue(issue):
    '''
    this function will seperate the issue for support into products and cause tuple
    some issues such as call transfer has no product associated with, 
    then return tuple ('unknown', cause).
    '''
    parts = issue.split('_', 1)
    if len(parts) == 1:
        return ('Unknown', parts[0])
    else:
        if parts[0] == 'Rcopia':
            parts[0] = 'RC'
        return (parts[0], parts[1])
        
def all_hour(x):
    '''
    all hours includeing both bussiness day and non_bussiness day between created and closed
    '''
    return (x[0] - x[1]).total_seconds()/3600
    
def business_hour(x):
    '''
    only count the hours in the work days
    '''
    hours = 0
    us_business_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    work_days = (pd.DatetimeIndex(start=x[1].date(), end=x[0].date(), freq=us_business_day)).map(lambda x: x.date())
    for day in work_days:
        if (day == x[1].date()) & (day < x[0].date()):
            hours += 24 - (x[1].hour + x[1].minute / 60. + x[1].second / 3600.)
        elif (day == x[1].date()) & (day == x[0].date()):
            return (x[0] - x[1]).total_seconds()/3600
        elif (day > x[1].date()) & (day < x[0].date()):
            hours += 24
        elif (day > x[1].date()) & (day == x[0].date()):
            hours += x[0].hour + x[0].minute / 60. + x[0].second / 3600.   
    return hours

def workable_day(df):
    '''
    count the number of workable days, exclude weekends, on-hold days, invited days
    '''
    if (df['New Value'] == 'COMPLETE').any():
        num_days = np.busday_count(begindates = df['Date Created'].iloc[0].date(), 
                                   enddates = df[df['New Value'] == 'COMPLETE']['Date'].iloc[0].date())
    elif (df['New Value'] == 'CANCELLED').any():
        num_days = np.busday_count(begindates = df['Date Created'].iloc[0].date(), 
                                   enddates = df[df['New Value'] == 'CANCELLED']['Date'].iloc[0].date())
        return num_days
    else:
        num_days = np.busday_count(begindates = df['Date Created'].iloc[0].date(), 
                                   enddates = date.today())
    if (df['New Value'] == 'ON HOLD').any():
        begin = df[df['New Value'] == 'ON HOLD']['Date'].iloc[0].date()
        try:
            bus_days = np.busday_count(begindates = begin, 
                                       enddates = df[df['Old Value'] == 'ON HOLD']['Date'].iloc[0].date())
        except IndexError:  # sometime project can be changed back to on hold after complete
            bus_days = np.busday_count(begindates = begin, 
                                       enddates = date.today())
        finally:
            num_days = num_days - bus_days
    if (df['New Value'] == 'INVITED').any():
        begin = df[df['New Value'] == 'INVITED']['Date'].iloc[0].date()
        try:
            bus_days = np.busday_count(begindates = begin, 
                                       enddates = df[df['Old Value'] == 'INVITED']['Date'].iloc[0].date())
        except IndexError:
            bus_days = np.busday_count(begindates = begin, 
                                       enddates = date.today())
        finally:
            num_days = num_days - bus_days
    return num_days
    
def date_closed(df):
    '''
    define data closed for implementation projects to be the date when deployment status is changed to canceled or complete
    '''
    if (df['New Value'] == 'COMPLETE').any():
        return df[df['New Value'] == 'COMPLETE']['Date'].iloc[0]
    elif (df['New Value'] == 'CANCELLED').any():
        return df[df['New Value'] == 'CANCELLED']['Date'].iloc[0]
    else:
        return np.nan
    
def categorize_pipeline(category):
    '''
    categorize pipeline based on deployment cateogry
    '''
    category = unicode(category) #.replace("\u2013", "-")  #replace utf-8 symbol (ndash) to ascii (-)
    if category.startswith('nan'):
        return 'Integration Services'
    elif category.find('EPCS') >= 0:
        return 'EPCS'
    elif category.startswith('MedHx'):
        return 'MedHX'
    elif category.startswith('Rcopia') | category.startswith('Partner') | category.startswith('RETAIL'):
        return 'Rcopia RcopiaMU'
    elif category.startswith('Akario'):
        return 'Backline'
    elif category.startswith('Smart'):
        return 'Smart Strings'
    else:
        return 'Others'

def categorize_customer_category_group(x, group_dict):
    '''
    this funciton group the customer category
    '''
    try:
        return group_dict[x]
    except:
        return 'Unknown'

def aging_workable_day_range(x):
    '''
    take a row series of ['Workable Days', '% Complete', 'Deployment Status']
    return a string represent the range of opened days
    '''
    if x[2] == 'INVITED':
        return 'Invited'
    elif x[2] == 'ON HOLD':
        return 'On Hold'
    elif x[1] >= 75:
        return 'Scheduled'
    num = x[0]
    if num <=6:
        return '0-6'
    elif num <= 13:
        return '7-13'
    elif num <= 20:
        return '14-20'
    elif num <= 27:
        return '21-27'
    else:
        return '28+'
    
def int_with_commas(x):
    '''
    this funciton is adding commans to numbers
    '''
    #if type(x) not in [type(0), type(0L)]:
        #raise TypeError("Parameter must be an integer.")
    if x < 0:
        return '-' + int_with_commas(-x)
    result = ''
    while x >= 1000:
        x, r = divmod(x, 1000)
        result = ",%03d%s" % (r, result)
    return "%d%s" % (x, result)
    
def month_name_to_date(month_name):
    '''
    format the 12-Mar-15 string to date, basically for epcs list only
    '''
    return datetime.strptime(month_name.replace(' ', ''),"%d-%b-%y").date()

def time_being_currnet_status(row):
    '''
    calculate the days being at the current date, 
    need to go to different column based epcs status
    '''
    epcs_status = row['EPCS Status']
    try:
        if epcs_status == 'TOKENBOUND':
            return (date.today() - month_name_to_date(row['IDP ProvDate'])).days, epcs_status
        elif (epcs_status == 'ENROLLED') & (row['Grant'] == 'INACTIVE'):
            return (date.today() - month_name_to_date(row['EnrolledActiveProvDate'])).days, epcs_status + ' INACTIVE'
        elif epcs_status == 'IDPFAILED':
            return (date.today() - month_name_to_date(row['IDP Faile'])).days, epcs_status
        elif epcs_status == 'INVITED':
            return (date.today() - month_name_to_date(row['IDP invit'])).days, epcs_status
        else:
            return np.nan, epcs_status
    except AttributeError:
        return -999, epcs_status + ' Weird'

def outreach_method(row, duration, category, contract, vendor, outreach_dict):
    if not (row[contract] is not np.nan) | (row[vendor] is not np.nan):
        return 'Not Qualified'
    try:
        criteria = outreach_dict[row[category]]
    except KeyError:
        return 'No Need' # 'No Need'
    duration_value = row[duration]
    for n_days, method, in criteria:
        if duration_value == n_days:
            return method
    if duration_value < n_days:
        return 'Not Yet'
    else:
        return 'Long Ago'
    
    
    
