'''
Created on Sep 29, 2017

@author: Heng.Zhang
'''

import pandas as pd
from global_variables import *
import numpy as np
from common import *
from feature_common import *


# 购买/浏览， 购买/收藏， 购买/购物车, 购物车/收藏， 购物车/浏览
# 浏览/行为总数， 收藏/行为总数， 购物车/行为总数， 购买/行为总数
# 返回 14 个特征
def feature_how_many_behavior_user(slide_window_df, UIC, feature_matrix):

    # 用户总共有过多少次浏览，收藏，购物车，购买的行为已经在 feature_user_item_behavior_ratio 得到
    # 购买/浏览， 购买/收藏， 购买/购物车    
    feature_matrix['user_buy_cnt_divides_view'] = SeriesDivision(feature_matrix['item_user_buy_sum'], feature_matrix['item_user_view_sum'])
    feature_matrix['user_buy_cnt_divides_fav'] = SeriesDivision(feature_matrix['item_user_buy_sum'], feature_matrix['item_user_fav_sum'])
    feature_matrix['user_buy_cnt_divides_cart'] = SeriesDivision(feature_matrix['item_user_buy_sum'], feature_matrix['item_user_cart_sum'])

    # 购物车/收藏， 购物车/浏览
    feature_matrix['user_cart_cn_divides_view'] = SeriesDivision(feature_matrix['item_user_cart_sum'], feature_matrix['item_user_view_sum'])
    feature_matrix['user_cart_cn_divides_fav'] = SeriesDivision(feature_matrix['item_user_cart_sum'], feature_matrix['item_user_fav_sum'])

    # 浏览/行为总数， 收藏/行为总数， 购物车/行为总数， 购买/行为总数
    feature_matrix['user_total_behavior_cnt'] = feature_matrix['item_user_view_sum'] + feature_matrix['item_user_fav_sum'] +\
                                                feature_matrix['item_user_cart_sum'] + feature_matrix['item_user_buy_sum']

    feature_matrix['user_veiw_cnt_ratio'] = SeriesDivision(feature_matrix['item_user_view_sum'], feature_matrix['user_total_behavior_cnt'])
    feature_matrix['user_fav_cnt_ratio'] = SeriesDivision(feature_matrix['item_user_fav_sum'], feature_matrix['user_total_behavior_cnt'])
    feature_matrix['user_cart_cnt_ratio'] = SeriesDivision(feature_matrix['item_user_cart_sum'], feature_matrix['user_total_behavior_cnt'])
    feature_matrix['user_buy_cnt_ratio'] = SeriesDivision(feature_matrix['item_user_buy_sum'], feature_matrix['user_total_behavior_cnt'])

    return feature_matrix

# user 浏览，收藏，购物车，购买了多少不同的item
# user 浏览，收藏，购物车，购买的不同 item 数量 / user 所有操作过的 item 数量
def feature_how_many_itme_user_opted(slide_window_df, UIC, feature_matrix):

    # user 浏览，收藏，购物车，购买了多少不同的 item
    item_cnt_user_opted_df = slide_window_df[['user_id', 'behavior_type', 'item_id']].drop_duplicates()
    item_cnt_user_opted_df.index = range(np.shape(item_cnt_user_opted_df)[0])
    item_cnt_user_opted_df = item_cnt_user_opted_df.groupby(['user_id', 'behavior_type'], sort=False, as_index=False)
    item_cnt_user_opted_df = item_cnt_user_opted_df.size().unstack().fillna(0)
    item_cnt_user_opted_df.reset_index(inplace=True)

    # user 操作过的item数量
    total_item_cnt_user_opted_df = UIC[['user_id', 'item_id']].groupby('user_id', sort=False, as_index=False)
    total_item_cnt_user_opted_df = total_item_cnt_user_opted_df.size().reset_index()
    total_item_cnt_user_opted_df.rename(columns={0:'user_opted_item_cnt'}, inplace=True)
    
    item_cnt_user_opted_df = pd.merge(item_cnt_user_opted_df, total_item_cnt_user_opted_df, how='inner', sort=False)
    
    # 浏览过的不同 item 数量/操作过的 item 总数，  收藏/总数， 购物车/总数， 购买/总数
    item_cnt_user_opted_df['user_veiw_item_ratio'] = SeriesDivision(item_cnt_user_opted_df[1], item_cnt_user_opted_df['user_opted_item_cnt'])
    item_cnt_user_opted_df['user_fav_item_ratio'] = SeriesDivision(item_cnt_user_opted_df[2], item_cnt_user_opted_df['user_opted_item_cnt'])
    item_cnt_user_opted_df['user_cart_item_ratio'] = SeriesDivision(item_cnt_user_opted_df[3], item_cnt_user_opted_df['user_opted_item_cnt'])
    item_cnt_user_opted_df['user_buy_item_ratio'] = SeriesDivision(item_cnt_user_opted_df[4], item_cnt_user_opted_df['user_opted_item_cnt'])

    item_cnt_user_opted_df.rename(columns={1:'user_view_item_cnt', 
                                           2:'user_fav_item_cnt', 
                                           3:'user_cart_item_cnt', 
                                           4:'user_buy_item_cnt'}, inplace=True)

    # 用户购买率：购买的 item 数量 / 操作过的item 数量
    item_cnt_user_opted_df['user_buy_ratio'] = SeriesDivision(item_cnt_user_opted_df['user_buy_item_cnt'], item_cnt_user_opted_df['user_opted_item_cnt'])

    feature_matrix = pd.merge(feature_matrix, item_cnt_user_opted_df, how='left', on='user_id', sort=False)

    return feature_matrix




