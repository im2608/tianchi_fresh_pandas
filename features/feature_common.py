'''
Created on Oct 11, 2017

@author: Heng.Zhang
'''

import pandas as pd
from global_variables import *
import numpy as np
from common import *
from unittest.mock import inplace

# 用户在checking day 前一天对商品是否有过某种操作 cart/favorite
def feature_user_opt_before1day(slide_window_df, UIC, behavior_type, item_or_category):
    opt_before1day_df = slide_window_df[(slide_window_df['dayoffset'] == 1)&(slide_window_df['behavior_type'] == behavior_type)][['user_id', item_or_category]]
    opt_before1day_df.drop_duplicates(inplace=True)
    opt_before1day_df['opt_before1day' ] = 1

    return opt_before1day_df


# 用户checking_date（不包括）之前操作（浏览， 收藏， 购物车， 购买）的次数, 这些次数占该用户操作item/category总次数的比例,
# 购买/浏览
# 购买/收藏
# 购买/购物车
def feature_user_behavior_ratio(slide_window_df, UIC, item_or_category):
    # user在 item/category 上各个操作的次数
    user_behavior_cnt_df = slide_window_df[['user_id', item_or_category, 'behavior_type']]
    user_behavior_cnt_df = user_behavior_cnt_df.groupby(['user_id', item_or_category, 'behavior_type'], sort=False, as_index=False)
    user_behavior_cnt_df = user_behavior_cnt_df.size().unstack().fillna(0)

    # user在 item/category 各种操作总的次数
    user_behavior_sum_df = user_behavior_cnt_df.groupby(level='user_id').sum()

    user_behavior_raito_df = user_behavior_cnt_df / user_behavior_sum_df
    user_behavior_raito_df.fillna(0, inplace=True)

    user_behavior_sum_df.rename(columns={1:'user_view_sum', 2:'user_fav_sum', 3:'user_cart_sum', 4:'user_buy_sum'}, inplace=True)
    user_behavior_sum_df.reset_index(inplace=True)
    user_behavior_sum_df.index = range(np.shape(user_behavior_sum_df)[0])
    user_behavior_sum_df['user_buy_divides_view'] = SeriesDivision(user_behavior_sum_df['user_buy_sum'], user_behavior_sum_df['user_view_sum'])
    user_behavior_sum_df['user_buy_divides_fav'] = SeriesDivision(user_behavior_sum_df['user_buy_sum'], user_behavior_sum_df['user_fav_sum'])
    user_behavior_sum_df['user_buy_divides_cart'] = SeriesDivision(user_behavior_sum_df['user_buy_sum'], user_behavior_sum_df['user_cart_sum'])

    # user在某个 item/category 上各种操作次数
    user_behavior_cnt_df.reset_index(inplace=True)
    user_behavior_cnt_df.index = range(np.shape(user_behavior_cnt_df)[0])
    user_behavior_cnt_df.rename(columns={1:'user_view_cnt', 2:'user_fav_cnt', 3:'user_cart_cnt', 4:'user_buy_cnt'}, inplace=True)

    user_behavior_raito_df.reset_index(inplace=True)
    user_behavior_raito_df.index = range(np.shape(user_behavior_raito_df)[0])
    user_behavior_raito_df.rename(columns={1:'user_view_ratio', 2:'user_fav_ratio', 3:'user_cart_ratio', 4:'user_buy_ratio'}, inplace=True)
    del user_behavior_raito_df['user_id']
    del user_behavior_raito_df[item_or_category]

    user_behavior_df = pd.merge(user_behavior_sum_df, user_behavior_cnt_df, how='left', on='user_id')

    user_behavior_df = pd.concat([user_behavior_df, user_behavior_raito_df], axis=1)
    user_behavior_df.index = range(np.shape(user_behavior_df)[0])

    return user_behavior_df


# 用户第一次，最后一次操作  至 window_end_date(不包括) 的天数
# 用户第一次，最后一次操作 之间的天数, 
def feature_user_1stlast_opt(slide_window_df, UIC, item_or_category):
    dayoffset_df = slide_window_df[['user_id', item_or_category, 'dayoffset']].groupby(['user_id', item_or_category], as_index=False, sort=False)

    # 第一次操作的dayoffset
    dayoffset_1stopt = dayoffset_df.max()
    dayoffset_1stopt.rename(columns={"dayoffset":"dayoffset_1st"}, inplace=True)

    # 最后一次操作的dayoffset
    dayoffset_lastopt = dayoffset_df.min()
    dayoffset_lastopt.rename(columns={"dayoffset":"dayoffset_last"}, inplace=True)
    
    dayoffset_1stlast = pd.merge(dayoffset_1stopt, dayoffset_lastopt, on=['user_id', item_or_category])
    
    # 第一次最后一次操作之间的dayoffset
    dayoffset_1stlast['1st_last_dayoffset'] = dayoffset_1stlast['dayoffset_1st'] - dayoffset_1stlast['dayoffset_last']
    
    dayoffset_1stlast.index = range(np.shape(dayoffset_1stlast)[0])
    
    return dayoffset_1stlast

#  用户第一次操作到购买之间的天数
def feature_days_between_1stopt_and_buy(slide_window_df, UIC, item_or_category):
    # 用户第一次购买的dayoffset
    user_buy_df = slide_window_df[slide_window_df['behavior_type'] == 4][['user_id', item_or_category, 'dayoffset']]
    user_buy_df  = user_buy_df.groupby(['user_id', item_or_category], as_index=False, sort=False).max()
    user_buy_df.rename(columns={'dayoffset':'dayoffset_buy'}, inplace=True)

    # 用户第一次操作的dayoffset
    user_dayoffset_1st = slide_window_df[['user_id', item_or_category, 'dayoffset']]
    user_dayoffset_1st = user_dayoffset_1st.groupby(['user_id', item_or_category], as_index=False, sort=False).max()
    user_dayoffset_1st.rename(columns={'dayoffset':'dayoffset_1stopt'}, inplace=True)

    user_dayofset_1stbuy_df = pd.merge(user_dayoffset_1st, user_buy_df, on=['user_id', item_or_category], how='left')
    user_dayofset_1stbuy_df['dayoffset_1stopt_buy'] = user_dayofset_1stbuy_df['dayoffset_1stopt'] - user_dayofset_1stbuy_df['dayoffset_buy']
    user_dayofset_1stbuy_df['dayoffset_1stopt_buy'][user_dayofset_1stbuy_df['dayoffset_1stopt_buy'] == 0] = 0.5
    user_dayofset_1stbuy_df['dayoffset_1stopt_buy'].fillna(0, inplace=True)
    
    del user_dayofset_1stbuy_df['dayoffset_1stopt']
    del user_dayofset_1stbuy_df['dayoffset_buy']
    
    user_dayofset_1stbuy_df.index = range(np.shape(user_dayofset_1stbuy_df)[0])

    return user_dayofset_1stbuy_df


#用户第一次购买 item/cat 前， 在 item/cat 上各个 behavior 的数量, 3个特征
def feature_behavior_cnt_before_1st_buy(slide_window_df, UIC, item_or_category):
    
    # 用户第一次购买的dayoffset
    user_buy_df = slide_window_df[slide_window_df['behavior_type'] == 4][['user_id', item_or_category, 'dayoffset']]
    user_1stbuy_df = user_buy_df.groupby(['user_id', item_or_category], as_index=False, sort=False).max()
    user_1stbuy_df.rename(columns={'dayoffset':'dayoffset_1stbuy'}, inplace=True)

    # 将dayoffset_1stbuy与slide window 合并，添加一个新列，没有购买的记录为NaN
    user_1st_buy_df = pd.merge(slide_window_df, user_1stbuy_df, how='left', on=['user_id', item_or_category])
    
    # 取得user第一次购买前 在 item 上各个 behavior 的数量
    user_opt_before_1st_buy_df = user_1st_buy_df[user_1st_buy_df['dayoffset'] > user_1st_buy_df['dayoffset_1stbuy']][['user_id', item_or_category, 'behavior_type']]
    user_opt_before_1st_buy_df = user_opt_before_1st_buy_df.groupby(['user_id', item_or_category, 'behavior_type'], as_index=False, sort=False).size().unstack()
    user_opt_before_1st_buy_df.fillna(0, inplace=True)
    user_opt_before_1st_buy_df.reset_index(inplace=True)
    user_opt_before_1st_buy_df.index = range(np.shape(user_opt_before_1st_buy_df)[0])
    user_opt_before_1st_buy_df.rename(columns={1:'viewcnt_bef_1stbuy', 2:'favcnt_bef_1stbuy', 3:'cartcnt_bef_1stbuy'}, inplace=True)

    user_opt_before_1st_buy_df.index = range(np.shape(user_opt_before_1st_buy_df)[0])

    return user_opt_before_1st_buy_df


# item/category 第一次, 最后一次 behavior 距离checking date 的天数, 以及第一次, 最后一次之间的天数， 返回 12 个特征
def feature_days_from_1st_last_behavior(slide_window_df, UIC, item_or_category):
    behavior_dayoffset = slide_window_df[[item_or_category, 'behavior_type', 'dayoffset']]
    behavior_dayoffset = behavior_dayoffset.groupby([item_or_category, 'behavior_type'], sort=False, as_index=False)
    
    # item/category 第一次的dayoffset
    behavior_dayoffset_1st = behavior_dayoffset.max()
    behavior_dayoffset_1st = behavior_dayoffset_1st.pivot(index=item_or_category, columns='behavior_type', values='dayoffset')
    behavior_dayoffset_1st.reset_index(inplace=True)
    behavior_dayoffset_1st.fillna(0, inplace=True)    

    # item/category  最后一次的dayoffset
    behavior_dayoffset_last = behavior_dayoffset.min()
    behavior_dayoffset_last = behavior_dayoffset_last.pivot(index=item_or_category, columns='behavior_type', values='dayoffset')
    behavior_dayoffset_last.reset_index(inplace=True)
    behavior_dayoffset_last.fillna(0, inplace=True)
    dayoffset_1st_last = behavior_dayoffset_1st[[1,2,3,4]] - behavior_dayoffset_last[[1,2,3,4]]
    
    behavior_dayoffset_1st.rename(columns={1:'view_1st_dayoffset', 2:'fav_1st_dayoffset', 
                                           3:'cart_1st_dayoffset', 4:'buy_1st_dayoffset'}, inplace=True)    
    behavior_dayoffset_last.rename(columns={1:'view_last_dayoffset', 2:'fav_last_dayoffset', 
                                            3:'cart_last_dayoffset', 4:'buy_last_dayoffset'}, inplace=True)
    dayoffset_1st_last.rename(columns={1:'view_1st_last_dayoffset', 2:'fav_1st_last_dayoffset', 
                                       3:'cart_1st_last_dayoffset', 4:'buy_1st_last_dayoffset'}, inplace=True)

    behavior_dayoffset = pd.merge(behavior_dayoffset_1st, behavior_dayoffset_last, how='left', on=item_or_category)
    behavior_dayoffset = pd.concat([behavior_dayoffset, dayoffset_1st_last], axis=1)

    return behavior_dayoffset

# item/category 上各个行为的次数,以及销量(即buy的次数)的排序
def feature_behavior_cnt(slide_window_df, UIC, item_or_category):
    behavior_cnt_df = slide_window_df[[item_or_category, 'behavior_type']]
    behavior_cnt_df = behavior_cnt_df.groupby([item_or_category, 'behavior_type'], sort=False, as_index=False)
    behavior_cnt_df = behavior_cnt_df.size().unstack()
    behavior_cnt_df.reset_index(inplace=True)
    behavior_cnt_df.fillna(0, inplace=True)
    behavior_cnt_df.rename(columns={1:'view_cnt', 2:'fav_cnt', 3:'cart_cnt', 4:'sale_volume'}, inplace=True)
    
    behavior_cnt_df['sale_volume_rank'] = behavior_cnt_df['sale_volume'].rank(method='dense', ascending=False)
    
    return behavior_cnt_df

# item / category 上各个行为用户的数量
def feature_user_cnt_on_behavior(slide_window_df, UIC, item_or_category):
    user_cnt_on_behavior_df = slide_window_df[[item_or_category, 'behavior_type', 'user_id']]
    user_cnt_on_behavior_df = user_cnt_on_behavior_df.groupby([item_or_category, 'behavior_type', 'user_id'], sort=False, as_index=False)
    user_cnt_on_behavior_df = user_cnt_on_behavior_df.size()
    user_cnt_on_behavior_df = user_cnt_on_behavior_df.reset_index() #不能用 inplace=Ture, 因为没有多重索引
    
    del user_cnt_on_behavior_df[0] # 此时得到item的各个行为上有哪些user操作过，列 0 表示操作过多少次，不需要，把它删掉
    
    user_cnt_on_behavior_df = user_cnt_on_behavior_df.groupby([item_or_category, 'behavior_type'], sort=False, as_index=False)
    user_cnt_on_behavior_df = user_cnt_on_behavior_df.size().unstack() 
    user_cnt_on_behavior_df.fillna(0, inplace=True)
    user_cnt_on_behavior_df.reset_index(inplace=True)
    user_cnt_on_behavior_df.rename(columns={1:'user_cnt_on_view', 2:'user_cnt_on_fav', 3:'user_cnt_on_cart', 4:'user_cnt_on_buy'}, inplace=True)
    
    return user_cnt_on_behavior_df







