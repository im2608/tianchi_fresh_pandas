'''
Created on Aug 4, 2017

@author: Heng.Zhang
'''
import os
import sys
import numpy as np
import pandas as pd

import csv
import datetime
from global_variables import *
from user_features import *
from user_item_features import *
from feature_extraction import * 
from common import *

def calculate_slide_window(slide_window_df, slide_window_size, checking_date, ):    
    print("handling checking date ", checking_date)
    feature_matrix_df = pd.DataFrame()
    
    user_item_pair = slide_window_df[['user_id', 'item_id']].drop_duplicates()
    user_item_pair.index = range(np.shape(user_item_pair)[0])

    feature_matrix_df = pd.concat([feature_matrix_df, user_item_pair], axis = 1)

    # 用户属性
    # 用户在checking day 前一天对item是否有过cart/ favorite
    feature = feature_user_item_opt_before1day(slide_window_df, user_item_pair, 3)
    feature_matrix_df = pd.merge(feature_matrix_df, feature, how='lfet', on=['user_id', 'item_id'])
    
    feature = feature_user_item_opt_before1day(slide_window_df, user_item_pair, 2)
    feature_matrix_df = pd.merge(feature_matrix_df, feature, how='lfet', on=['user_id', 'item_id'])

    # 用户 - 商品 属性
    # 用户checking_date（不包括）之前 在item上操作（浏览， 收藏， 购物车， 购买）该商品的次数, 这些次数占该用户操作商品的总次数的比例,
    feature = feature_user_item_behavior_ratio(slide_window_df, user_item_pair)
    feature_matrix_df = pd.merge(feature_matrix_df, feature, how='lfet', on=['user_id', 'item_id'])
    
    # 用户第一次，最后一次操作 item 至 window_end_date(不包括) 的天数
    # 用户第一次，最后一次操作 item 之间的天数, 
    feature = feature_user_item_1stlast_opt(slide_window_df, user_item_pair)
    feature_matrix_df = pd.merge(feature_matrix_df, feature, how='lfet', on=['user_id', 'item_id'])

    #  用户第一次操作商品到购买之间的天数
    feature = feature_days_between_1stopt_and_buy(slide_window_df, user_item_pair)
    feature_matrix_df = pd.merge(feature_matrix_df, feature, how='lfet', on=['user_id', 'item_id'])
    
    #用户第一次购买 item 前， 在 item 上各个 behavior 的数量, 3个特征
    feature = feature_behavior_cnt_before_1st_buy(slide_window_df, user_item_pair)
    feature_matrix_df = pd.merge(feature_matrix_df, feature, how='lfet', on=['user_id', 'item_id'])

    checking_date += datetime.timedelta(days = 1)

    del slide_window_df
    return


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

def main():
    data_filename = r"%s\..\input\preprocessed_user_data_no_hour.csv" % (runningPath)
    
    print("reading csv ", data_filename)

    raw_data_df = pd.read_csv(data_filename)
    
    slide_window_size = 7
    start_date = datetime.datetime.strptime('2014-11-18', "%Y-%m-%d")
    end_date = datetime.datetime.strptime('2014-12-18', "%Y-%m-%d")

    print("slide window size %d, start date %s, end date %s" % 
          (slide_window_size, convertDatatimeToStr(start_date), convertDatatimeToStr(end_date)))
    
    checking_date = start_date + datetime.timedelta(days = slide_window_size)
    while (checking_date <= end_date):
        start_date_str = convertDatatimeToStr(start_date)
        checking_date_str =  convertDatatimeToStr(checking_date)
        slide_window_df = raw_data_df[(raw_data_df['time'] >= start_date_str ) & (raw_data_df['time'] < checking_date_str)]
        slide_window_df = remove_user_item_only_buy(slide_window_df)
        slide_window_df.index = range(np.shape(slide_window_df)[0])

        slide_window_df['dayoffset'] = convert_date_str_to_dayoffset(slide_window_df['time'], slide_window_size, checking_date)
        del slide_window_df['time']
        calculate_slide_window(slide_window_df, slide_window_size, checking_date)
        
        break

    return 0


if __name__ == '__main__':
    main()
    