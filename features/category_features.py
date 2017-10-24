'''
Created on Oct 12, 2017

@author: Heng.Zhang
'''

import pandas as pd
from global_variables import *
import numpy as np
from common import *
from feature_common import *
from dask.dataframe.core import Series


# category 第一次, 最后一次 behavior 距离checking date 的天数, 以及第一次, 最后一次之间的天数， 返回 12 个特征
def feature_category_days_from_1st_last_behavior(slide_window_df, UIC, feature_matrix):
    behavior_dayoffset = feature_days_from_1st_last_behavior(slide_window_df, UIC, 'item_category')
    behavior_dayoffset.rename(columns=rename_category_col_name, inplace=True)
    feature_matrix = pd.merge(feature_matrix, behavior_dayoffset, how='left', on=['item_category'])
    
    return feature_matrix

# category 上各个行为的次数, 
# 每日的平均次数，
# 前[1,2,3,4]天的次数/ 每日的平均次数，
# 以及销量(即buy的次数)的排序
# category 上各个行为的次数平均在每个item上的次数
def feature_category_behavior_cnt(slide_window_df, slide_window_size, UIC, feature_matrix):
    category_behavior_cnt = feature_behavior_cnt(slide_window_df, slide_window_size, UIC, 'item_category')
    category_behavior_cnt.rename(columns=rename_category_col_name, inplace=True)

    items_in_cat_df = slide_window_df[['item_id', 'item_category']].drop_duplicates()
    items_in_cat_df = items_in_cat_df.groupby('item_category', sort=False, as_index=False).size()
    items_in_cat_df = items_in_cat_df.reset_index()
    items_in_cat_df.rename(columns={0:"item_cnt_in_cat"}, inplace=True) # 每个category内有多少item
    
    # category 上各个行为的次数平均到每个item上的次数
    category_behavior_cnt = pd.merge(category_behavior_cnt, items_in_cat_df, how='left', on='item_category')
    category_behavior_cnt['category_view_cnt_mean_on_item'] = SeriesDivision(category_behavior_cnt['category_view_cnt'], category_behavior_cnt['item_cnt_in_cat'])
    category_behavior_cnt['category_fav_cnt_mean_on_item'] = SeriesDivision(category_behavior_cnt['category_fav_cnt'], category_behavior_cnt['item_cnt_in_cat'])
    category_behavior_cnt['category_cart_cnt_mean_on_item'] = SeriesDivision(category_behavior_cnt['category_cart_cnt'], category_behavior_cnt['item_cnt_in_cat'])
    category_behavior_cnt['category_sale_vol_mean_on_item'] = SeriesDivision(category_behavior_cnt['category_sale_volume'], category_behavior_cnt['item_cnt_in_cat'])
    del category_behavior_cnt['item_cnt_in_cat']
    
    feature_matrix = pd.merge(feature_matrix, category_behavior_cnt, how='left', on=['item_category'])

    return feature_matrix


# category  上[1, 2, 3, slide_window_size] 天各个行为用户的数量
# 转化率：  [fav, cart, buy]的user数量/view 过的 user数量
def feature_category_user_cnt_on_behavior(slide_window_df, slide_window_size, UIC, feature_matrix):
    category_user_cnt_on_behavior = feature_user_cnt_on_behavior(slide_window_df, slide_window_size, UIC, 'item_category')
    category_user_cnt_on_behavior.rename(columns=rename_category_col_name, inplace=True)
    
    feature_matrix = pd.merge(feature_matrix, category_user_cnt_on_behavior, how='left', on=['item_category'])
    return feature_matrix



