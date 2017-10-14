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
from common import *
from feature_extraction import *

from user_features import *
from user_item_features import * 
from user_category_features import *
from category_features import *
from item_features import *
from corss_features import *
from nltk.tbl import feature

def calculate_slide_window(slide_window_df, slide_window_size, checking_date, ):    
    print("handling checking date ", checking_date)
    
    UIC = slide_window_df[['user_id', 'item_id', 'item_category']].drop_duplicates()
    UIC.index = range(np.shape(UIC)[0])
    
    feature_matrix_df = pd.DataFrame()
    feature_matrix_df = pd.concat([feature_matrix_df, UIC], axis = 1)

    #############################################################################################
    #############################################################################################
    # user-item 特征
    #############################################################################################
    #############################################################################################    
    # 用户在checking day 前一天对item是否有过cart/ favorite
    feature_matrix_df = feature_user_item_opt_before1day(slide_window_df, UIC, 3, feature_matrix_df)   
    feature_matrix_df = feature_user_item_opt_before1day(slide_window_df, UIC, 2, feature_matrix_df)

    # 用户checking_date（不包括）之前 在item上操作（浏览， 收藏， 购物车， 购买）该商品的次数, 这些次数占该用户操作商品的总次数的比例,
    feature_matrix_df = feature_user_item_behavior_ratio(slide_window_df, UIC, feature_matrix_df)
    
    # 用户第一次，最后一次操作 item 至 window_end_date(不包括) 的天数
    # 用户第一次，最后一次操作 item 之间的天数, 
    feature_matrix_df = feature_user_item_1stlast_opt(slide_window_df, UIC, feature_matrix_df)
    
    #  用户第一次操作商品到购买之间的天数
    feature_matrix_df = feature_user_item_days_between_1stopt_and_buy(slide_window_df, UIC, feature_matrix_df)
    
    #用户第一次购买 item 前， 在 item 上各个 behavior 的数量, 3个特征
    feature_matrix_df = feature_user_item_behavior_cnt_before_1st_buy(slide_window_df, UIC, feature_matrix_df)
    
    # [begin date, end date) 期间，总共有多少用户购买了该  item
    feature_matrix_df = feature_how_many_users_bought_item(slide_window_df, UIC, feature_matrix_df)
    

    #############################################################################################
    #############################################################################################
    # user-category 特征
    #############################################################################################
    #############################################################################################
   # 用户在checking day 前一天对 category 是否有过cart/ favorite
    feature_matrix_df = feature_user_category_opt_before1day(slide_window_df, UIC, 3)
    feature_matrix_df = feature_user_category_opt_before1day(slide_window_df, UIC, 2)

    # 用户checking_date（不包括）之前 在 category 上操作（浏览， 收藏， 购物车， 购买）该商品的次数, 这些次数占该用户操作 category 的总次数的比例,
    feature_matrix_df = feature_user_category_behavior_ratio(slide_window_df, UIC, feature_matrix_df)
    
    # 用户第一次，最后一次操作 category 至 window_end_date(不包括) 的天数
    # 用户第一次，最后一次操作 category 之间的天数, 
    feature_matrix_df = feature_user_category_1stlast_opt(slide_window_df, UIC, feature_matrix_df)

    #  用户第一次操作 category 到购买之间的天数
    feature_matrix_df = feature_user_category_days_between_1stopt_and_buy(slide_window_df, UIC, feature_matrix_df)

    #用户第一次购买 category 前， 在 caetory 上各个 behavior 的数量, 3个特征
    feature_matrix_df = feature_user_category_behavior_cnt_before_1st_buy(slide_window_df, UIC, feature_matrix_df)
    
    # [begin date, end date) 期间，总共有多少用户购买了该 category
    feature_matrix_df = feature_how_many_users_bought_category(slide_window_df, UIC, feature_matrix_df)
    
    
    #############################################################################################
    #############################################################################################
    # category 特征
    #############################################################################################
    #############################################################################################
    
    # category 第一次, 最后一次 behavior 距离checking date 的天数, 以及第一次, 最后一次之间的天数， 返回 12 个特征
    feature_matrix_df = feature_category_days_from_1st_last_behavior(slide_window_df, UIC, feature_matrix_df)
    
    # category 上各个行为的次数,以及销量(即buy的次数)的排序
    feature_matrix_df = feature_item_behavior_cnt(slide_window_df, UIC, feature_matrix_df)
    
    # category 上各个行为用户的数量
    feature_matrix_df = feature_category_user_cnt_on_behavior(slide_window_df, UIC, feature_matrix_df)

    
    #############################################################################################
    #############################################################################################
    # item 特征
    #############################################################################################
    #############################################################################################
    # item 第一次, 最后一次 behavior 距离checking date 的天数, 以及第一次, 最后一次之间的天数， 返回 12 个特征
    feature_matrix_df = feature_item_days_from_1st_last_behavior(slide_window_df, UIC, feature_matrix_df)
    
    
    # item 上各个行为的次数,以及销量(即buy的次数)的排序
    feature_matrix_df = feature_item_behavior_cnt(slide_window_df, UIC, feature_matrix_df)
    
    # item 上各个行为用户的数量
    feature_matrix_df = feature_item_user_cnt_on_behavior(slide_window_df, UIC, feature_matrix_df)
 
 
 
    #############################################################################################
    #############################################################################################
    # cross 特征
    #############################################################################################
    #############################################################################################
    
    # item 的销量占 category 的销量的比例, 以及item 销量在category销量中的排序
    feature_matrix_df = feature_sales_ratio_itme_category(feature_matrix_df)
    
    # item 的1st, last behavior 与 category 的1st， last 相差的天数
    feature_1st_last_IC(feature_matrix_df)

    # item  在各个behavior上的次数占 category 上各个behavior次数的比例    
    feature_matrix_df = feature_behavior_cnt_itme_category(feature_matrix_df)
    
    
    
    
    
    #############################################################################################
    #############################################################################################
    # 特征结束
    #############################################################################################
    #############################################################################################
    checking_date += datetime.timedelta(days = 1)

    del slide_window_df
    return feature_matrix_df


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

        slide_window_df.index = range(np.shape(slide_window_df)[0])  # 重要！！

        slide_window_df['dayoffset'] = convert_date_str_to_dayoffset(slide_window_df, slide_window_size, checking_date)
        del slide_window_df['time']
        calculate_slide_window(slide_window_df, slide_window_size, checking_date)
        
        break

    return 0


if __name__ == '__main__':
    main()
    