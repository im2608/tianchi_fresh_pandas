'''
Created on Oct 12, 2017

@author: Heng.Zhang
'''


import pandas as pd
from global_variables import *
import numpy as np
from common import *
from feature_common import *

# item 第一次, 最后一次 behavior 距离checking date 的天数, 以及第一次, 最后一次之间的天数， 返回 12 个特征
def feature_item_days_from_1st_last_behavior(slide_window_df, UIC, feature_matrix):
    behavior_dayoffset = feature_days_from_1st_last_behavior(slide_window_df, UIC, 'item_id')
    behavior_dayoffset.rename(columns=rename_item_col_name, inplace=True)
    feature_matrix = pd.merge(feature_matrix, behavior_dayoffset, how='left', on=['item_id'])

    return feature_matrix

# item 上各个行为的次数, 
# 每日的平均次数，
# 前[1,2,3,4]天的次数/ 每日的平均次数，
# 以及销量(即buy的次数)的排序
def feature_item_behavior_cnt(slide_window_df, slide_window_size, UIC, feature_matrix):
    item_behavior_cnt = feature_behavior_cnt(slide_window_df, slide_window_size, UIC, 'item_id')
    item_behavior_cnt.rename(columns=rename_item_col_name, inplace=True)
    feature_matrix = pd.merge(feature_matrix, item_behavior_cnt, how='left', on=['item_id'])

    return feature_matrix

# item  上[1, 2, 3, slide_window_size] 天各个行为用户的数量
# 转化率：  [fav, cart, buy]的user数量/view 过的 user数量
def feature_item_user_cnt_on_behavior(slide_window_df, slide_window_size, UIC, feature_matrix):
    item_user_cnt_on_behavior = feature_user_cnt_on_behavior(slide_window_df, slide_window_size, UIC, 'item_id')
    item_user_cnt_on_behavior.rename(columns=rename_item_col_name, inplace=True)
    
    feature_matrix = pd.merge(feature_matrix, item_user_cnt_on_behavior, how='left', on=['item_id'])
    return feature_matrix

# item 上每个平均每个 user 点击、收藏、购物车、购买多少次 
def featur_user_mean_of_behvior_cnt_on_item(slide_window_df, UIC, feature_matrix):
    feature_matrix['user_mean_of_view_cnt_on_item'] = SeriesDivision(feature_matrix['item_user_cnt_on_view'], feature_matrix['item_view_cnt'])
    feature_matrix['user_mean_of_fav_cnt_on_item'] = SeriesDivision(feature_matrix['item_user_cnt_on_fav'], feature_matrix['item_fav_cnt'])
    feature_matrix['user_mean_of_cart_cnt_on_item'] = SeriesDivision(feature_matrix['item_user_cnt_on_cart'], feature_matrix['item_cart_cnt'])
    feature_matrix['user_mean_of_buy_cnt_on_item'] = SeriesDivision(feature_matrix['item_user_cnt_on_buy'], feature_matrix['item_buy_cnt'])

    return feature_matrix


