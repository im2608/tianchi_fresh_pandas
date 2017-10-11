'''
Created on Sep 29, 2017

@author: Heng.Zhang
'''

import pandas as pd
from global_variables import *
import numpy as np
from common import *
from feature_common import *

# 用户在checking day 前一天对商品是否有过某种操作 cart/favorite
def feature_user_item_opt_before1day(slide_window_df, user_item_pair, behavior_type):
    
    opt_before1day_df = feature_user_opt_before1day(slide_window_df, user_item_pair, behavior_type, 'item_id')
    if (behavior_type == 2):
        opt_before1day_df.rename(columns={'opt_before1day':'item_fav_opt_before1day'}, inplace=True)
    else:
        opt_before1day_df.rename(columns={'opt_before1day':'item_cart_opt_before1day'}, inplace=True)
        
    return opt_before1day_df

# 用户checking_date（不包括）之前 在item上操作（浏览， 收藏， 购物车， 购买）的次数, 这些次数占该用户操作商品的总次数的比例,
def feature_user_item_behavior_ratio(slide_window_df, user_item_pair):

    user_behavior_df = feature_user_behavior_ratio(slide_window_df, user_item_pair, 'item_id')
    
    user_behavior_df.rename(columns=lambda col_name: "item_" + col_name if (col_name != 'item_id' and col_name != 'user_id') else col_name , inplace=True)
    user_behavior_df.name= "user_behavior_on_item"

    return user_behavior_df

# 用户第一次，最后一次操作 item 至 window_end_date(不包括) 的天数
# 用户第一次，最后一次操作 item 之间的天数, 
def feature_user_item_1stlast_opt(slide_window_df, user_item_pair):
    dayoffset_1stlast = feature_user_1stlast_opt(slide_window_df, user_item_pair, 'item_id')
    dayoffset_1stlast.rename(columns=lambda col_name: "item_" + col_name if (col_name != 'item_id' and col_name != 'user_id') else col_name , inplace=True)
    return dayoffset_1stlast


#  用户第一次操作商品到购买之间的天数
def feature_user_item_days_between_1stopt_and_buy(slide_window_df, user_item_pair):
    user_buy_item_df = feature_days_between_1stopt_and_buy(slide_window_df, user_item_pair, 'item_id')
    user_buy_item_df.rename(columns=lambda col_name: "item_" + col_name if (col_name != 'item_id' and col_name != 'user_id') else col_name , inplace=True)
    return user_buy_item_df 

#用户第一次购买 item 前， 在 item 上各个 behavior 的数量, 3个特征
def feature_user_item_behavior_cnt_before_1st_buy(slide_window_df, user_item_pair):
    user_opt_before_1st_buy_df = feature_behavior_cnt_before_1st_buy(slide_window_df, user_item_pair, 'item_id')
    user_opt_before_1st_buy_df.rename(columns=lambda col_name: "item_" + col_name if (col_name != 'item_id' and col_name != 'user_id') else col_name , inplace=True)    
    return user_opt_before_1st_buy_df











