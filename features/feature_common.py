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
def feature_user_opt_before1day(slide_window_df, behavior_type, item_or_category):
    opt_before1day_df = slide_window_df[(slide_window_df['dayoffset'] == 1)&(slide_window_df['behavior_type'] == behavior_type)][['user_id', item_or_category]]
    opt_before1day_df.drop_duplicates(inplace=True)
    opt_before1day_df['opt_before1day' ] = 1

    return opt_before1day_df


# 用户checking_date（不包括）之前操作（浏览， 收藏， 购物车， 购买）的次数, 这些次数占该用户操作item/category总次数的比例,
# 购买/浏览
# 购买/收藏
# 购买/购物车
# 用户在过去 [1，2，3，4]天在 item/category上（浏览， 收藏， 购物车， 购买）的次数
# user 在前一天 最早，最晚操作item/category的hour
def get_user_behavior_cnt(slide_window_df, dayoffset, item_or_category):
    user_behavior_cnt_df = slide_window_df[['user_id', item_or_category, 'behavior_type']][slide_window_df['dayoffset'] <= dayoffset].drop_duplicates()
    user_behavior_cnt_df = user_behavior_cnt_df.groupby(['user_id', item_or_category, 'behavior_type'], sort=False, as_index=False)
    user_behavior_cnt_df = user_behavior_cnt_df.size().unstack().fillna(0)
    user_behavior_cnt_df.reset_index(inplace=True)
    user_behavior_cnt_df.rename(columns={1:"%d_day_user_view_cnt" % dayoffset, 
                                         2:"%d_day_user_fav_cnt" % dayoffset,
                                         3:"%d_day_user_cart_cnt" % dayoffset,
                                         4:"%d_day_user_buy_cnt" % dayoffset}, inplace=True)
    return user_behavior_cnt_df

def feature_user_behavior_ratio(slide_window_df, slide_window_size, UIC, item_or_category):
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
    user_behavior_cnt_df = pd.merge(user_behavior_cnt_df, user_behavior_raito_df, on=['user_id', item_or_category], how='left')

    # 用户在过去 [1，2，3，4,7]天在item/category （浏览， 收藏， 购物车， 购买）的次数
    user_behavior_cnt_n_day_arr = [user_behavior_cnt_df]
    for dayoffset in [1,2,3,4,7]:
        user_behavior_cnt_n_day_arr = get_user_behavior_cnt(slide_window_df, dayoffset, item_or_category)
        user_behavior_cnt_df = pd.merge(user_behavior_cnt_df, user_behavior_cnt_n_day_arr, on=['user_id', item_or_category], how='left')
        user_behavior_cnt_df.fillna(0, inplace=True)

    # user 在前一天 最早，最晚操作item/category的hour
    user_1st_last_opt_hour = slide_window_df[['user_id', item_or_category, 'hour']][slide_window_df['dayoffset'] == 1].drop_duplicates()
    user_1st_last_opt_hour = user_1st_last_opt_hour.groupby(['user_id', item_or_category], sort=False, as_index=False)
    
    user_1st_opt_hour = user_1st_last_opt_hour.min()
    user_1st_opt_hour.rename(columns={'hour':'user_1st_opt_hour'}, inplace=True)
    
    user_last_opt_hour = user_1st_last_opt_hour.max()
    user_last_opt_hour.rename(columns={'hour':'user_last_opt_hour'}, inplace=True)

    user_1st_last_opt_hour = pd.merge(user_1st_opt_hour, user_last_opt_hour, on=['user_id', item_or_category], how='inner')
    user_behavior_cnt_df = pd.merge(user_behavior_cnt_df, user_1st_last_opt_hour, on=['user_id', item_or_category], how='left')

    user_behavior_df = pd.merge(user_behavior_sum_df, user_behavior_cnt_df, how='left', on='user_id')
    user_behavior_df.fillna(0, inplace=True)
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
    user_dayofset_1stbuy_df.loc[user_dayofset_1stbuy_df['dayoffset_1stopt_buy'] == 0, 'dayoffset_1stopt_buy'] = 0.5
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


# item/category 上，前 dayoffset 天各个行为的总次数
def behavior_cnt_in_days(slide_window_df, dayoffset, item_or_category):
    behavior_cnt_df = slide_window_df[[item_or_category, 'behavior_type']][slide_window_df['dayoffset'] <= dayoffset]
    behavior_cnt_df = behavior_cnt_df.groupby([item_or_category, 'behavior_type'], sort=False, as_index=False)
    behavior_cnt_df = behavior_cnt_df.size().unstack()
    behavior_cnt_df.reset_index(inplace=True)
    behavior_cnt_df.fillna(0, inplace=True)
    behavior_cnt_df.rename(columns={1:'%d_day_view_cnt' % dayoffset , 
                                    2:'%d_day_fav_cnt' % dayoffset ,
                                    3:'%d_day_cart_cnt' % dayoffset ,
                                    4:'%d_day_sale_volume' % dayoffset}, inplace=True)
    return behavior_cnt_df


# item/category 上各个行为的次数, 
# 每日的平均次数，
# 前[1,2,3,4]天的次数/ 每日的平均次数，
# 以及销量(即buy的次数)的排序
def feature_behavior_cnt(slide_window_df, slide_window_size, UIC, item_or_category):
    behavior_cnt_df = behavior_cnt_in_days(slide_window_df, slide_window_size, item_or_category)
    
    # 销量(即buy的次数)的排序
    behavior_cnt_df['sale_volume_rank'] = behavior_cnt_df['%d_day_sale_volume' % slide_window_size].rank(method='dense', ascending=False)

    behavior_cnt_mean_df = behavior_cnt_df[['%d_day_view_cnt' % slide_window_size, 
                                            '%d_day_fav_cnt' % slide_window_size,
                                            '%d_day_cart_cnt' % slide_window_size, 
                                            '%d_day_sale_volume' % slide_window_size]] / slide_window_size
    behavior_cnt_mean_df.rename(columns={'%d_day_view_cnt' % slide_window_size:'view_cnt_mean', 
                                         '%d_day_fav_cnt' % slide_window_size:'fav_cnt_mean',
                                         '%d_day_cart_cnt' % slide_window_size:'cart_cnt_mean',
                                         '%d_day_sale_volume' % slide_window_size:'sale_volume_mean'}, inplace=True)

    behavior_cnt_df = pd.concat([behavior_cnt_df, behavior_cnt_mean_df], axis=1)
    for i in range(1, 5):
        # item / category 在某天可能没有任何操作，所以不能concat，应该merge
        behavior_cnt_n_day_df = behavior_cnt_in_days(slide_window_df, i, item_or_category)
        behavior_cnt_df = pd.merge(behavior_cnt_df, behavior_cnt_n_day_df, on=item_or_category, how='left')
        behavior_cnt_df.fillna(0, inplace=True)

        behavior_cnt_df["%d_day_view_cnt_divides_mean" % i] = SeriesDivision(behavior_cnt_n_day_df['%d_day_view_cnt' % i], behavior_cnt_df['view_cnt_mean'])
        behavior_cnt_df["%d_day_fav_cnt_divides_mean" % i] = SeriesDivision(behavior_cnt_n_day_df['%d_day_fav_cnt' % i], behavior_cnt_df['fav_cnt_mean'])
        behavior_cnt_df["%d_day_cart_cnt_divides_mean" % i] = SeriesDivision(behavior_cnt_n_day_df['%d_day_cart_cnt' % i], behavior_cnt_df['cart_cnt_mean'])
        behavior_cnt_df["%d_day_sale_volume_divides_mean" % i] = SeriesDivision(behavior_cnt_n_day_df['%d_day_sale_volume' % i], behavior_cnt_df['sale_volume_mean'])
    
    behavior_cnt_df.fillna(0, inplace=True)

    return behavior_cnt_df

# item / category 上[1, 2, 3, 4, slide_window_size] 天各个行为用户的数量
# 转化率：  [fav, cart, buy]的user数量/view 过的 user数量
def get_user_cnt_on_behavior(slide_window_df, item_or_category, dayoffset):
    user_cnt_on_behavior_df = slide_window_df[[item_or_category, 'behavior_type', 'user_id']][slide_window_df['dayoffset'] <= dayoffset].drop_duplicates()
    user_cnt_on_behavior_df = user_cnt_on_behavior_df.groupby([item_or_category, 'behavior_type'], sort=False, as_index=False)
    user_cnt_on_behavior_df = user_cnt_on_behavior_df.size().unstack()
    user_cnt_on_behavior_df.reset_index(inplace=True)
    user_cnt_on_behavior_df.fillna(0, inplace=True)

    user_cnt_on_behavior_df.rename(columns={1:"%d_day_user_cnt_on_view" % dayoffset, 
                                            2:"%d_day_user_cnt_on_fav" % dayoffset, 
                                            3:"%d_day_user_cnt_on_cart" % dayoffset,
                                            4:"%d_day_user_cnt_on_sale_volume" % dayoffset}, inplace=True)
    return user_cnt_on_behavior_df


def feature_user_cnt_on_behavior(slide_window_df, slide_window_size, UIC, item_or_category):
    user_cnt_on_behavior_df = get_user_cnt_on_behavior(slide_window_df, item_or_category, slide_window_size)
    user_cnt_on_behavior_df['fav_conversion'] =  SeriesDivision(user_cnt_on_behavior_df['%d_day_user_cnt_on_fav' % slide_window_size], 
                                                                user_cnt_on_behavior_df['%d_day_user_cnt_on_view' % slide_window_size])
    
    user_cnt_on_behavior_df['cart_conversion'] =  SeriesDivision(user_cnt_on_behavior_df['%d_day_user_cnt_on_cart' % slide_window_size], 
                                                                 user_cnt_on_behavior_df['%d_day_user_cnt_on_view' % slide_window_size])
    
    user_cnt_on_behavior_df['buy_conversion'] =  SeriesDivision(user_cnt_on_behavior_df['%d_day_user_cnt_on_sale_volume' % slide_window_size], 
                                                                user_cnt_on_behavior_df['%d_day_user_cnt_on_view' % slide_window_size])

    for dayoffset in [1,2,3,4]:
        user_cnt_on_behavior_of_n_days = get_user_cnt_on_behavior(slide_window_df, item_or_category, dayoffset)
        user_cnt_on_behavior_df = pd.merge(user_cnt_on_behavior_df, user_cnt_on_behavior_of_n_days, on=item_or_category, how='left')
        user_cnt_on_behavior_df.fillna(0, inplace=True)
    
    return user_cnt_on_behavior_df


# user/item/category 在一周内每天各个操作的次数
def feature_behavior_cnt_on_weekday(slide_window_df, user_or_item_or_category):
    behavior_cnt_on_weekday_df = slide_window_df[[user_or_item_or_category, 'behavior_type', 'weekday']]
    behavior_cnt_on_weekday_df = behavior_cnt_on_weekday_df.groupby([user_or_item_or_category, 'weekday', 'behavior_type'], sort=False, as_index=False)
    behavior_cnt_on_weekday_df = behavior_cnt_on_weekday_df.size().unstack() # 展开 behavior_type
    behavior_cnt_on_weekday_df.fillna(0, inplace=True)
    behavior_cnt_on_weekday_df = behavior_cnt_on_weekday_df.unstack() # 展开 weekday
    behavior_cnt_on_weekday_df.fillna(0, inplace=True)
    behavior_cnt_on_weekday_df.reset_index(inplace=True)
    
    behavior_cnt_name = {1:'view_cnt_at_weekday_', 2:'fav_cnt_at_weekday_', 3:'cart_cnt_at_weekday_', 4:'buy_cnt_at_weekday_'}  
    arr = [behavior_cnt_on_weekday_df[user_or_item_or_category]]
    for behavior in [1,2,3,4]:
        arr.append(behavior_cnt_on_weekday_df[behavior].rename(columns=lambda weekday: "%s%d" % (behavior_cnt_name[behavior], weekday)))
    
    behavior_cnt_on_weekday_df = pd.concat(arr, axis=1)    
    return behavior_cnt_on_weekday_df

