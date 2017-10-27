'''
Created on Oct 12, 2017

@author: Heng.Zhang
'''

import pandas as pd
from global_variables import *
import numpy as np
from common import *
from feature_common import *


# user 在 item上各个行为的次数占user在 category 上各个行为次数的比例
# item 上各个行为的次数 / category 上各个行为的次数平均在每个item上的次数
def feature_user_item_cnt_ratio_on_category(feature_matrix_df):
    feature_matrix_df['user_item_view_cnt_ratio_on_cat'] = SeriesDivision(feature_matrix_df['item_user_view_cnt'], feature_matrix_df['category_user_view_cnt'])
    feature_matrix_df['user_item_favorite_cnt_ratio_on_cat'] = SeriesDivision(feature_matrix_df['item_user_favorite_cnt'], feature_matrix_df['category_favorite_cnt'])
    feature_matrix_df['user_item_cart_cnt_ratio_on_cat'] = SeriesDivision(feature_matrix_df['item_user_cart_cnt'], feature_matrix_df['category_user_cart_cnt'])
    feature_matrix_df['user_item_buy_cnt_ratio_on_cat'] = SeriesDivision(feature_matrix_df['item_user_buy_cnt'], feature_matrix_df['category_user_buy_cnt'])
    
    return feature_matrix_df

# item  在各个behavior上的次数占 category 上各个behavior次数的比例
# item 的销量占 category 的销量的比例, 以及 item 销量在 category 销量中的排序
def feature_sales_ratio_itme_category(feature_matrix_df, slide_window_size):
    feature_matrix_df['IC_view_cnt_ratio'] = SeriesDivision(feature_matrix_df['item_%d_day_view_cnt' % slide_window_size], 
                                                            feature_matrix_df['category_%d_day_view_cnt' % slide_window_size])
    feature_matrix_df['IC_fav_cnt_ratio'] = SeriesDivision(feature_matrix_df['item_%d_day_fav_cnt' % slide_window_size], 
                                                           feature_matrix_df['category_%d_day_fav_cnt' % slide_window_size])
    feature_matrix_df['IC_cart_cnt_ratio'] = SeriesDivision(feature_matrix_df['item_%d_day_cart_cnt' % slide_window_size], 
                                                            feature_matrix_df['category_%d_day_cart_cnt' % slide_window_size])
    feature_matrix_df['item_sale_vol_ratio_in_category'] = SeriesDivision(feature_matrix_df['item_%d_day_sale_volume' % slide_window_size], 
                                                                          feature_matrix_df['category_%d_day_sale_volume' % slide_window_size])

#     item_sale_vol_rank_in_category = feature_matrix_df[['item_id', 'item_category', 'item_%d_day_sale_volume' % slide_window_size]]
#     grouped_item_sale_val = item_sale_vol_rank_in_category.groupby(['item_category'], sort=False, as_index=False)
# 
#     item_sale_vol_rank_in_category = []    
#     # 循环中计算 item 销量在各个 category 销量中的排序
#     i = 0
#     for item_category, item_sale_vol_group in grouped_item_sale_val:
#         i += 1
#         if (i % 100 == 0):
#             print("%s, %d groups ranked\r" % (getCurrentTime(), i), end="")
#         item_sale_vol_group.drop_duplicates(inplace=True)
#         item_sale_vol_group['item_sale_volume_rank_in_cat'] = item_sale_vol_group['item_%d_day_sale_volume' % slide_window_size].rank(method='dense', ascending=False)
#         item_sale_vol_rank_in_category.append(item_sale_vol_group)
#  
#     item_sale_vol_rank_in_category = pd.concat(item_sale_vol_rank_in_category, axis=0)
#     del item_sale_vol_rank_in_category['item_%d_day_sale_volume' % slide_window_size]
#      
#     feature_matrix_df = pd.merge(feature_matrix_df, item_sale_vol_rank_in_category, how='left', on=['item_category', 'item_id'] )
         
    return feature_matrix_df


# item 的1st, last behavior 与 category 的1st， last 相差的天数
def feature_1st_last_IC(feature_matrix_df):
    feature_matrix_df['IC_view_1st_dayoffset'] = np.abs(feature_matrix_df['item_view_1st_dayoffset'] - feature_matrix_df['category_view_1st_dayoffset'])
    feature_matrix_df['IC_fav_1st_dayoffset']  = np.abs(feature_matrix_df['item_fav_1st_dayoffset'] - feature_matrix_df['category_fav_1st_dayoffset'])
    feature_matrix_df['IC_cart_1st_dayoffset'] = np.abs(feature_matrix_df['item_cart_1st_dayoffset'] - feature_matrix_df['category_cart_1st_dayoffset'])
    feature_matrix_df['IC_buy_1st_dayoffset']  = np.abs(feature_matrix_df['item_buy_1st_dayoffset'] - feature_matrix_df['category_buy_1st_dayoffset'])

    feature_matrix_df['IC_view_last_dayoffset'] = np.abs(feature_matrix_df['item_view_last_dayoffset'] - feature_matrix_df['category_view_last_dayoffset'])
    feature_matrix_df['IC_fav_last_dayoffset'] = np.abs(feature_matrix_df['item_fav_last_dayoffset'] - feature_matrix_df['category_fav_last_dayoffset'])
    feature_matrix_df['IC_cart_last_dayoffset'] = np.abs(feature_matrix_df['item_cart_last_dayoffset'] - feature_matrix_df['category_cart_last_dayoffset'])
    feature_matrix_df['IC_buy_last_dayoffset'] = np.abs(feature_matrix_df['item_buy_last_dayoffset'] - feature_matrix_df['category_buy_last_dayoffset'])

    return feature_matrix_df

# 商品热度 浏览，收藏，购物车，购买该商品的用户数/浏览，收藏，购物车，购买同类型商品的总用户数
def feature_item_popularity(feature_matrix_df, slide_window_size):
    feature_matrix_df['item_view_pop'] = SeriesDivision(feature_matrix_df['item_%d_day_user_cnt_on_view' % slide_window_size], 
                                                        feature_matrix_df['category_%d_day_user_cnt_on_view' % slide_window_size])
    feature_matrix_df['item_fav_pop'] = SeriesDivision(feature_matrix_df['item_%d_day_user_cnt_on_fav' % slide_window_size],
                                                       feature_matrix_df['category_%d_day_user_cnt_on_fav' % slide_window_size])
    feature_matrix_df['item_cart_pop'] = SeriesDivision(feature_matrix_df['item_%d_day_user_cnt_on_cart' % slide_window_size], 
                                                        feature_matrix_df['category_%d_day_user_cnt_on_cart' % slide_window_size])
    feature_matrix_df['item_buy_pop'] = SeriesDivision(feature_matrix_df['item_%d_day_user_cnt_on_sale_volume' % slide_window_size], 
                                                       feature_matrix_df['category_%d_day_user_cnt_on_sale_volume' % slide_window_size])
    return feature_matrix_df

# item的[fav, cart, buy]转化率/category的购买转化率
def feature_item_conversion(feature_matrix_df):
    feature_matrix_df['item_fav_conversion_divids_cat'] = SeriesDivision(feature_matrix_df['item_fav_conversion'], feature_matrix_df['category_buy_conversion'])
    feature_matrix_df['item_cart_conversion_divids_cat'] = SeriesDivision(feature_matrix_df['item_cart_conversion'], feature_matrix_df['category_buy_conversion'])
    feature_matrix_df['item_buy_conversion_divids_cat'] = SeriesDivision(feature_matrix_df['item_buy_conversion'], feature_matrix_df['category_buy_conversion'])

    return feature_matrix_df


# item 上各个行为的次数 / category 上各个行为的次数平均在每个item上的次数
def feature_item_cnt_ratio_on_category(feature_matrix_df, slide_window_size):
    feature_matrix_df['item_view_ratio_in_cat'] = SeriesDivision(feature_matrix_df['item_%d_day_view_cnt' % slide_window_size], 
                                                                 feature_matrix_df['category_view_cnt_mean_on_item'])
    feature_matrix_df['item_fav_ratio_in_cat'] = SeriesDivision(feature_matrix_df['item_%d_day_fav_cnt' % slide_window_size], 
                                                                feature_matrix_df['category_fav_cnt_mean_on_item'])
    feature_matrix_df['item_cart_ratio_in_cat'] = SeriesDivision(feature_matrix_df['item_%d_day_cart_cnt' % slide_window_size], 
                                                                 feature_matrix_df['category_cart_cnt_mean_on_item'])
    feature_matrix_df['item_buy_ratio_in_cat'] = SeriesDivision(feature_matrix_df['item_%d_day_sale_volume' % slide_window_size], 
                                                                feature_matrix_df['category_sale_vol_mean_on_item'])
    return feature_matrix_df

