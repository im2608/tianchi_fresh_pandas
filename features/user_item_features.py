'''
Created on Sep 29, 2017

@author: Heng.Zhang
'''

import pandas as pd
from global_variables import *
import numpy as np
from common import *
from feature_common import *

# 用户在checking day 前一天对商品是否有过某种操作 view/cart/favorite/buy
def feature_user_item_opt_before1day(slide_window_df, behavior_type, feature_matrix_df):
    
    opt_before1day_df = feature_user_opt_before1day(slide_window_df, behavior_type, 'item_id')
    column_name = {1:'item_view_opt_before1day', 2:'item_fav_opt_before1day', 3:'item_cart_opt_before1day', 4:'item_buy_opt_before1day'}    
    opt_before1day_df.rename(columns={'opt_before1day':column_name[behavior_type]}, inplace=True)
    feature_matrix_df = pd.merge(feature_matrix_df, opt_before1day_df, how='left', on=['user_id', 'item_id'])
    feature_matrix_df.fillna(0, inplace=True)
    return feature_matrix_df

# 用户checking_date（不包括）之前操作（浏览， 收藏， 购物车， 购买）的次数, 这些次数占该用户操作item总次数的比例,
# 购买/浏览
# 购买/收藏
# 购买/购物车
# 用户在过去 [1，2，3，4]天（浏览， 收藏， 购物车， 购买）的次数
# user 在前一天 最早，最晚操作item的hour
def feature_user_item_behavior_ratio(slide_window_df, slide_window_size, UIC, feature_matrix_df):

    user_behavior_df = feature_user_behavior_ratio(slide_window_df, slide_window_size, UIC, 'item_id')
    
    user_behavior_df.rename(columns=rename_item_col_name, inplace=True)
    user_behavior_df.name= "user_behavior_on_item"

    feature_matrix_df = pd.merge(feature_matrix_df, user_behavior_df, how='left', on=['user_id', 'item_id'])
    return feature_matrix_df


# 用户第一次，最后一次操作 item 至 window_end_date(不包括) 的天数
# 用户第一次，最后一次操作 item 之间的天数, 
def feature_user_item_1stlast_opt(slide_window_df, UIC, feature_matrix_df):
    dayoffset_1stlast = feature_user_1stlast_opt(slide_window_df, UIC, 'item_id')
    dayoffset_1stlast.rename(columns=rename_item_col_name, inplace=True)
    
    feature_matrix_df = pd.merge(feature_matrix_df, dayoffset_1stlast, how='left', on=['user_id', 'item_id'])
    return feature_matrix_df

#  用户第一次操作商品到购买之间的天数
def feature_user_item_days_between_1stopt_and_buy(slide_window_df, UIC, feature_matrix_df):
    user_buy_item_df = feature_days_between_1stopt_and_buy(slide_window_df, UIC, 'item_id')
    user_buy_item_df.rename(columns=rename_item_col_name, inplace=True)

    feature_matrix_df = pd.merge(feature_matrix_df, user_buy_item_df, how='left', on=['user_id', 'item_id'])
    return feature_matrix_df

#用户第一次购买 item 前， 在 item 上各个 behavior 的数量, 3个特征
def feature_user_item_behavior_cnt_before_1st_buy(slide_window_df, UIC, feature_matrix_df):
    user_opt_before_1st_buy_df = feature_behavior_cnt_before_1st_buy(slide_window_df, UIC, 'item_id')
    user_opt_before_1st_buy_df.rename(columns=rename_item_col_name, inplace=True)
    feature_matrix_df = pd.merge(feature_matrix_df, user_opt_before_1st_buy_df, how='left', on=['user_id', 'item_id'])
    feature_matrix_df.fillna(0, inplace=True)
    return feature_matrix_df











