'''
Created on Sep 29, 2017

@author: Heng.Zhang
'''

import pandas as pd
from global_variables import *
import numpy as np
from unittest.mock import inplace


# 用户在checking day 前一天对商品是否有过某种操作 cart/favorite
def feature_user_opt_before1day(slide_window_df, user_item_pair, behavior_type):
    opt_before1day_df = slide_window_df[['user_id', 'item_id']][(slide_window_df['dayoffset'] == 1) & (slide_window_df['behavior_type'] == behavior_type)]
    opt_before1day_df.drop_duplicates(inplace=True)
    opt_before1day_df['opt_before1day' ] = 1
    
    merged = pd.merge(user_item_pair, opt_before1day_df, how='left')
    merged['opt_before1day'][merged['opt_before1day'] != 1] = 0

    return merged['opt_before1day']

# 用户checking_date（不包括）之前 在item上操作（浏览， 收藏， 购物车， 购买）的次数, 这些次数占该用户操作商品的总次数的比例,
def feature_user_item_behavior_ratio(slide_window_df, user_item_pair):
    user_behavior_cnt_df = slide_window_df[['user_id', 'item_id', 'behavior_type']].groupby(['user_id', 'item_id', 'behavior_type'])
    user_behavior_cnt_df = user_behavior_cnt_df.size().unstack().fillna(0)    
    user_behavior_sum_df = user_behavior_cnt_df.groupby(level='user_id').sum()
    user_behavior_raito_df = user_behavior_cnt_df / user_behavior_sum_df
    user_behavior_raito_df.fillna(0, inplace=True)
    
    user_behavior_sum_df.rename(columns={1:'view_sum', 2:'favorite_sum', 3:'cart_sum', 4:'buy_sum'}, inplace=True)
    user_behavior_sum_df.reset_index(inplace=True)
    user_behavior_sum_df.index = range(np.shape(user_behavior_sum_df)[0])
    user_behavior_sum_df['buy_divides_view'] = user_behavior_sum_df['buy_sum'] / user_behavior_sum_df['view_sum']
    user_behavior_sum_df['buy_divides_favorite'] = user_behavior_sum_df['buy_sum'] / user_behavior_sum_df['favorite_sum']
    user_behavior_sum_df['buy_divides_cart'] = user_behavior_sum_df['buy_sum'] / user_behavior_sum_df['cart_sum']
    user_behavior_sum_df.fillna(0, inplace=True)
    
    user_behavior_cnt_df.reset_index(inplace=True)
    user_behavior_cnt_df.index = range(np.shape(user_behavior_cnt_df)[0])
    user_behavior_cnt_df.rename(columns={1:'view_on_item_cnt', 2:'favorite_on_item_cnt', 3:'cart_on_item_cnt', 4:'buy_on_item_cnt'}, inplace=True)

    user_behavior_raito_df.reset_index(inplace=True)
    user_behavior_raito_df.index = range(np.shape(user_behavior_raito_df)[0])
    user_behavior_raito_df.rename(columns={1:'view_raiot', 2:'favorite_ratio', 3:'cart_ratio', 4:'buy_ratio'}, inplace=True)
    del user_behavior_raito_df['user_id']
    del user_behavior_raito_df['item_id']

    user_behavior_df = pd.merge(user_behavior_sum_df, user_behavior_cnt_df, how='left', on='user_id')
    
    user_behavior_df = pd.concat([user_behavior_df, user_behavior_raito_df], axis=1)
    
    del user_behavior_df['user_id']
    del user_behavior_df['item_id']
    user_behavior_df.index = range(np.shape(user_behavior_df)[0])

    return user_behavior_df

# 用户第一次，最后一次操作 item 至 window_end_date(不包括) 的天数
# 用户第一次，最后一次操作 item 之间的天数, 
def feature_user_item_1stlast_opt(slide_window_df, user_item_pair):
    dayoffset_df = slide_window_df[['user_id', 'item_id', 'dayoffset']].groupby(['user_id', 'item_id'], as_index=False)
    dayoffset_1stopt = dayoffset_df.max()
    del dayoffset_1stopt['user_id']
    del dayoffset_1stopt['item_id']
    dayoffset_1stopt.columns = ['1stopt_dayoffset']
    
    dayoffset_lastopt = dayoffset_1stopt.min()
    del dayoffset_lastopt['user_id']
    del dayoffset_lastopt['item_id']
    dayoffset_lastopt.columns = ['lastopt_dayoffset']
    
    dayoffset_1stlast = dayoffset_1stopt - dayoffset_lastopt
    dayoffset_1stlast.columns = ['1st_last_dayoffset']

    return pd.concat([dayoffset_1stopt, dayoffset_lastopt, dayoffset_1stlast], axis=1)



#  用户第一次操作商品到购买之间的天数
def feature_days_between_1stopt_and_buy(slide_window_df, user_item_pair):
    # 购买商品的用户信息
    user_buy_df = slide_window_df[['user_id', 'item_id', 'dayoffset']][slide_window_df['behavior_type'] == 4].drop_duplicates()
    user_buy_df.rename(columns={'dayoffset':'dayoffset_buy'}, inplace=True)
    
    user_dayoffset_1st = slide_window_df[['user_id', 'item_id', 'dayoffset']].groupby(['user_id', 'item_id'], as_index=False, sort=False).max()
    user_dayoffset_1st.rename(columns={'dayoffset':'dayoffset_1stopt'}, inplace=True)

    user_buy_item_df = pd.merge(user_dayoffset_1st, user_buy_df, on=['user_id', 'item_id'], how='left')
    user_buy_item_df['dayoffset_1stopt_buy'] = user_buy_item_df['dayoffset_1stopt'] - user_buy_item_df['dayoffset_buy']
    user_buy_item_df['dayoffset_1stopt_buy'][user_buy_item_df['dayoffset_1stopt_buy'] == 0] = 0.5
    user_buy_item_df['dayoffset_1stopt_buy'].fillna(0, inplace=True)

    pd.merge(user_item_pair, user_buy_item_df, how='left', on=['user_id', 'item_id'])
    return user_buy_item_df['dayoffset_1stopt_buy']

#用户第一次购买 item 前， 在 item 上的的各个 behavior 的数量, 3个特征
def feature_behavior_cnt_before_1st_buy(slide_window_df, user_item_pair):
    user_buy_df = slide_window_df[['user_id', 'item_id', 'dayoffset']][slide_window_df['behavior_type'] == 4].drop_duplicates()
    user_1st_buy = user_buy_df.groupby(['user_id', 'item_id'], as_index=False, sort=False).max()
    user_1st_buy = pd.merge(user_1st_buy, user_buy_df, how='left', on=['user_id', 'item_id'] )
    return
















