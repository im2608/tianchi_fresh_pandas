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

    slide_window_df['dayoffset'] = convert_date_str_to_dayoffset(slide_window_df['time'], slide_window_size, checking_date)
    del slide_window_df['time']

    # 用户属性
    # 用户在checking day 前一天是否有过cart/ favorite
    feature_matrix_df['cart_before1day'] = feature_user_opt_before1day(slide_window_df, user_item_pair, 3)
    feature_matrix_df['favorite_before1day'] = feature_user_opt_before1day(slide_window_df, user_item_pair, 2)

    # 用户 - 商品 属性
    # 用户checking_date（不包括）之前 在item上操作（浏览， 收藏， 购物车， 购买）该商品的次数, 这些次数占该用户操作商品的总次数的比例,
    behavior_sum_ratio_df = feature_user_item_behavior_ratio(slide_window_df, user_item_pair)
    feature_matrix_df = pd.concat([feature_matrix_df, behavior_sum_ratio_df], axis= 1)
    
    # 用户第一次，最后一次操作 item 至 window_end_date(不包括) 的天数
    # 用户第一次，最后一次操作 item 之间的天数, 
    user_item_1stlast_opt = feature_user_item_1stlast_opt(slide_window_df, user_item_pair)
    feature_matrix_df = pd.concat([feature_matrix_df, user_item_1stlast_opt], axis= 1)

    #  用户第一次操作商品到购买之间的天数
    ays_between_1stopt_and_buy = feature_days_between_1stopt_and_buy(slide_window_df, user_item_pair)
    feature_matrix_df = pd.concat([feature_matrix_df, ays_between_1stopt_and_buy], axis=1)

    checking_date += datetime.timedelta(days = 1)

    del slide_window_df
    return

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
        slide_window_df.index = range(np.shape(slide_window_df)[0])
        calculate_slide_window(slide_window_df, slide_window_size, checking_date)
        
        break

    return 0


if __name__ == '__main__':
    main()
    