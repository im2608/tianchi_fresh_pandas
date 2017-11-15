'''
Created on Aug 4, 2017

@author: Heng.Zhang
'''

import datetime
import pandas as pd
import numpy as np
import time

from global_variables import *

def convertDatatimeToStr(opt_datatime):
    return "%04d-%02d-%02d" % (opt_datatime.year, opt_datatime.month, opt_datatime.day)


# 将date string 转成与checking date 之间相差的天数 
def convert_date_str_to_dayoffset(date_str_series, slide_window_size, checking_date):
    slide_window_date_dayoffset = {}
    for i in range(slide_window_size, 0, -1):
        date_in_window = checking_date + datetime.timedelta(days=-i)
        date_str = convertDatatimeToStr(date_in_window)
        slide_window_date_dayoffset[date_str] =  i

    dayoffset_series = pd.Series(list(map(lambda x : slide_window_date_dayoffset[x], date_str_series)))
    
    dayoffset_series.index = range(dayoffset_series.shape[0])

    return dayoffset_series
        
def SeriesDivision(divisor, dividend):
    quotient = divisor / dividend
    quotient.fillna(0, inplace=True)
    quotient[np.isinf(quotient)] = 0
    return quotient


rename_user_col_name = lambda col_name: "user_" + col_name if (col_name != 'item_id' and col_name != 'user_id' and col_name != 'item_category') else col_name
rename_item_col_name = lambda col_name: "item_" + col_name if (col_name != 'item_id' and col_name != 'user_id' and col_name != 'item_category') else col_name
rename_category_col_name = lambda col_name: "category_" + col_name if (col_name != 'item_id' and col_name != 'user_id' and col_name != 'item_category') else col_name


def remove_user_item_only_buy(slide_window_df):
    # 得到user在item上各个操作的次数,删除只有购买而没有其他操作的数据
    user_only_buyopt_df = slide_window_df[['user_id', 'item_id', 'behavior_type']]
    user_only_buyopt_df = user_only_buyopt_df.groupby(['user_id', 'item_id', 'behavior_type'], as_index=False, sort=False)
    user_only_buyopt_df = user_only_buyopt_df.size().unstack()
    user_only_buyopt_df.fillna(0, inplace=True)
    user_only_buyopt_df.reset_index(inplace=True)
    user_item_onlybuy_df = user_only_buyopt_df[['user_id', 'item_id']][(user_only_buyopt_df[1] == 0) &
                                                                       (user_only_buyopt_df[2] == 0) &
                                                                       (user_only_buyopt_df[3] == 0) & 
                                                                       ((user_only_buyopt_df[4] != 0))]
    # user只是购买了item，但没有其他操作，删除这些数据
    user_item_onlybuy_df['user_item_onlybuy'] = 1

    slide_window_df = pd.merge(slide_window_df, user_item_onlybuy_df, how='left', on=['user_id', 'item_id'])
    slide_window_df.drop(slide_window_df[slide_window_df['user_item_onlybuy'] == 1].index, inplace=True)
    del slide_window_df['user_item_onlybuy']

    return slide_window_df


def getCurrentTime():
    return time.strftime("%Y-%m-%d %X", time.localtime())

# 返回在之前有过操作并在checking day购买的记录，而不是返回所有在checking day的购买记录
def extracting_Y(UI, label_day_df):
    
    # 所有在checking day的购买记录
    Y = label_day_df[['user_id', 'item_id']][label_day_df['behavior_type'] == 4].drop_duplicates()
    Y['buy'] = 1
    Y.index = range(Y.shape[0])

    # merge 之后，则是在之前有过操作并在checking day购买的记录
    Y = pd.merge(UI, Y, how='left', on=['user_id', 'item_id'])
    Y.fillna(0, inplace=True)
    return Y


def get_output_filename(index, use_rule):
    if (use_rule):
        prob_output_filename = r"%s\..\output\single_window_forecast_with_prob_%s_%d_with_rule.csv" % (runningPath, datetime.date.today(), index)
        submit_output_filename = r"%s\..\output\single_window_forecast_submit_%s_%d_with_rule.csv" % (runningPath, datetime.date.today(), index)
    else:
        prob_output_filename = r"%s\..\output\single_window_forecast_with_prob_%s_%d_without_rule.csv" % (runningPath, datetime.date.today(), index)
        submit_output_filename = r"%s\..\output\single_window_forecast_submit_%s_%d_without_rule.csv" % (runningPath, datetime.date.today(), index)
        
    return prob_output_filename, submit_output_filename

def get_feature_name_for_model(features):
    s = set(features)
    r = set(['user_id', 'item_id', 'item_category', 'buy'])
    return list(s-r)

# 计算正例的 F1
def calculate_POS_F1(Y_true_UI, Y_fcsted_UI):
    UI_true = Y_true_UI.apply(lambda x : "%s,%s" % (str(x['user_id']), str(x['item_id'])), axis=1)
    UI_pred = Y_fcsted_UI.apply(lambda x : "%s,%s" % (str(np.int64(x['user_id'])), str(np.int64(x['item_id']))), axis=1)
    
    UI_true = set(UI_true.values)
    UI_pred = set(UI_pred.values)
    
    UI_hit = UI_pred.intersection(UI_true)
    
    hit_cnt = len(UI_hit)
    
    print("%s True count %d, Forecasted count %d, Hit count %d" % (getCurrentTime(), len(UI_true), len(UI_pred), hit_cnt))
    
    p = hit_cnt / len(UI_pred)
    r = hit_cnt / len(UI_true)
    
    f1 = 2 * p * r / (p + r)
    
#     r = fp/(2p-f)

    return p, r, f1

def PCA(feature_matrix_df):
    f_mean = feature_matrix_df.mean()
    feature_remove_mean = feature_matrix_df - f_mean
    feature_cov = np.cov(feature_remove_mean, rowvar=0)
    eigVals, eigVecs = np.linalg.eig(feature_cov)
    
    eigValTotalSum = np.sum(eigVals)
    sortedVals = np.argsort(-eigVals)
    eigValSum = 0
    i = 0
    for i in range(len(eigVals)):
        eigValSum += eigVals[sortedVals[i]]
        if (eigValSum >= 0.9 * eigValTotalSum):
            break
        
    return eigVals, eigVecs, i


def create_slide_window_df(raw_data_df, window_start_date, window_end_date, slide_window_size, fcsted_item_df):
    print("%s creating slide window [%s, %s]" % (getCurrentTime(), window_start_date, window_end_date))
    if (fcsted_item_df is not None):
        slide_window_df = raw_data_df[(raw_data_df['time'] >= convertDatatimeToStr(window_start_date))&
                                      (raw_data_df['time'] < convertDatatimeToStr(window_end_date)) &
                                      (np.in1d(raw_data_df['item_id'], fcsted_item_df['item_id']))]
    else:
        slide_window_df = raw_data_df[(raw_data_df['time'] >= convertDatatimeToStr(window_start_date))&
                                      (raw_data_df['time'] < convertDatatimeToStr(window_end_date))]

    slide_window_df = remove_user_item_only_buy(slide_window_df)
    slide_window_df.index = range(slide_window_df.shape[0])  # 重要！！

    slide_window_df['dayoffset'] = convert_date_str_to_dayoffset(slide_window_df['time'], slide_window_size, window_end_date)

    return slide_window_df