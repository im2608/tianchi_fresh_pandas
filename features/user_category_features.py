'''
Created on Sep 29, 2017

@author: Heng.Zhang
'''

import pandas as pd
from global_variables import *
import numpy as np
from common import *
from feature_common import *

# 用户在checking day 前一天对category是否有过某种操作 cart/favorite
def feature_user_category_opt_before1day(slide_window_df, behavior_type, feature_matrix_df):

    opt_before1day_df = feature_user_opt_before1day(slide_window_df, behavior_type, 'item_category')
    if (behavior_type == 2):
        opt_before1day_df.rename(columns={'opt_before1day':'category_fav_opt_before1day'}, inplace=True)
    else:
        opt_before1day_df.rename(columns={'opt_before1day':'category_cart_opt_before1day'}, inplace=True)
        
    feature_matrix_df = pd.merge(feature_matrix_df, opt_before1day_df, how='left', on=['user_id', 'item_category'])
    feature_matrix_df.fillna(0, inplace=True)

    return feature_matrix_df

# 用户checking_date（不包括）之前操作（浏览， 收藏， 购物车， 购买）的次数, 这些次数占该用户操作 category 总次数的比例,
# 购买/浏览
# 购买/收藏
# 购买/购物车
# 用户在过去 [1，2，3，4]天（浏览， 收藏， 购物车， 购买）的次数
# user 在前一天 最早，最晚操作 category的hour
def feature_user_category_behavior_ratio(slide_window_df, slide_window_size, UIC, feature_matrix_df):

    user_behavior_df = feature_user_behavior_ratio(slide_window_df, slide_window_size, UIC, 'item_category')
    
    user_behavior_df.rename(columns=rename_category_col_name, inplace=True)
    user_behavior_df.name= "user_behavior_on_category"
    
    feature_matrix_df = pd.merge(feature_matrix_df, user_behavior_df, how='left', on=['user_id', 'item_category'])
    feature_matrix_df.fillna(0, inplace=True)

    return feature_matrix_df

# 用户第一次，最后一次操作 category 至 window_end_date(不包括) 的天数
# 用户第一次，最后一次操作 category 之间的天数, 
def feature_user_category_1stlast_opt(slide_window_df, UIC, feature_matrix_df):
    dayoffset_1stlast = feature_user_1stlast_opt(slide_window_df, UIC, 'item_category')
    dayoffset_1stlast.rename(columns=rename_category_col_name, inplace=True)

    feature_matrix_df = pd.merge(feature_matrix_df, dayoffset_1stlast, how='left', on=['user_id', 'item_category'])
    feature_matrix_df.fillna(0, inplace=True)

    return feature_matrix_df


#  用户第一次操作 category 到购买之间的天数
def feature_user_category_days_between_1stopt_and_buy(slide_window_df, UIC, feature_matrix_df):
    user_buy_item_df = feature_days_between_1stopt_and_buy(slide_window_df, UIC, 'item_category')
    user_buy_item_df.rename(columns=rename_category_col_name, inplace=True)
    feature_matrix_df = pd.merge(feature_matrix_df, user_buy_item_df, how='left', on=['user_id', 'item_category'])
    feature_matrix_df.fillna(0, inplace=True)

    return feature_matrix_df 

#用户第一次购买 category 前， 在 category 上各个 behavior 的数量, 3个特征
def feature_user_category_behavior_cnt_before_1st_buy(slide_window_df, UIC, feature_matrix_df):
    user_opt_before_1st_buy_df = feature_behavior_cnt_before_1st_buy(slide_window_df, UIC, 'item_category')
    user_opt_before_1st_buy_df.rename(columns=rename_category_col_name, inplace=True)
    feature_matrix_df = pd.merge(feature_matrix_df, user_opt_before_1st_buy_df, how='left', on=['user_id', 'item_category'])
    feature_matrix_df.fillna(0, inplace=True)

    return feature_matrix_df

# 在 [window_start_date, window_end_dat) 范围内 ， 用户一共购买过多少同类型的商品
def feature_how_many_buy_category(slide_window_df, UIC, feature_matrix_df):
    # 得到user购买过category的次数
    user_buy_category_df = slide_window_df[['user_id', 'item_category']][slide_window_df['behavior_type'] == 4]
    # 此处得到的是一个Series，需要转成DataFrame
    user_buy_category_df = user_buy_category_df.groupby(['user_id', 'item_category'], sort=False, as_index=False).size()    
    user_buy_category_df = user_buy_category_df.reset_index()
    user_buy_category_df.rename(columns={0:'user_buy_category_cnt'}, inplace=True)

    feature_matrix_df = pd.merge(feature_matrix_df, user_buy_category_df, how='left', on=['user_id', 'item_category'])
    feature_matrix_df.fillna(0, inplace=True)

    return user_buy_category_df











