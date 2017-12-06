'''
Created on Sep 26, 2017

@author: Heng.Zhang
'''

import pandas as pd
import numpy as np
from common import *
from feature_common import *
from user_features import *
from user_item_features import * 
from user_category_features import *
from category_features import *
from item_features import *
from corss_features import *
from sklearn import preprocessing


def extracting_features(slide_window_df, slide_window_size, fcsted_item_df):
    # 只对在label day 前一天有过交互的user-item 生成特征矩阵 
    last_day_of_slide_window = slide_window_df['time'].max()

    if (fcsted_item_df is not None):
        UIC = slide_window_df[['user_id', 'item_id', 'item_category']]\
                              [(np.in1d(slide_window_df['item_id'], fcsted_item_df['item_id']))&((slide_window_df['time'] == last_day_of_slide_window))].drop_duplicates()
    else:
        UIC = slide_window_df[['user_id', 'item_id', 'item_category']][slide_window_df['time'] == last_day_of_slide_window].drop_duplicates()

    UIC.index = range(np.shape(UIC)[0])

    feature_matrix_df = pd.DataFrame()
    feature_matrix_df = pd.concat([feature_matrix_df, UIC], axis = 1)

    #############################################################################################
    #############################################################################################
    # user-item 特征
    #############################################################################################
    #############################################################################################
    # 用户在checking day 前一天对item是否有过cart/ favorite
    feature_matrix_df = feature_user_item_opt_before1day(slide_window_df, 3, feature_matrix_df)   
    feature_matrix_df = feature_user_item_opt_before1day(slide_window_df, 2, feature_matrix_df)
    
    # 用户checking_date（不包括）之前 在item上操作（浏览， 收藏， 购物车， 购买）该商品的次数, 这些次数占该用户操作商品的总次数的比例,
    feature_matrix_df = feature_user_item_behavior_ratio(slide_window_df, slide_window_size, UIC, feature_matrix_df)
    
    # 用户第一次，最后一次操作 item 至 window_end_date(不包括) 的天数
    # 用户第一次，最后一次操作 item 之间的天数, 
    feature_matrix_df = feature_user_item_1stlast_opt(slide_window_df, UIC, feature_matrix_df)
    
    #  用户第一次操作商品到购买之间的天数
    feature_matrix_df = feature_user_item_days_between_1stopt_and_buy(slide_window_df, UIC, feature_matrix_df)
    
    #用户第一次购买 item 前， 在 item 上各个 behavior 的数量, 3个特征
    feature_matrix_df = feature_user_item_behavior_cnt_before_1st_buy(slide_window_df, UIC, feature_matrix_df)
    
#     print("%s After extracting user-item features, shape is " % (getCurrentTime()), feature_matrix_df.shape)
    

    #############################################################################################
    #############################################################################################
    # user-category 特征
    #############################################################################################
    #############################################################################################
   # 用户在checking day 前一天对 category 是否有过cart/ favorite
    feature_matrix_df = feature_user_category_opt_before1day(slide_window_df, 3, feature_matrix_df)
    feature_matrix_df = feature_user_category_opt_before1day(slide_window_df, 2, feature_matrix_df)

    # 用户checking_date（不包括）之前 在 category 上操作（浏览， 收藏， 购物车， 购买）该商品的次数, 这些次数占该用户操作 category 的总次数的比例,
    feature_matrix_df = feature_user_category_behavior_ratio(slide_window_df, slide_window_size, UIC, feature_matrix_df)
    
    # 用户第一次，最后一次操作 category 至 window_end_date(不包括) 的天数
    # 用户第一次，最后一次操作 category 之间的天数, 
    feature_matrix_df = feature_user_category_1stlast_opt(slide_window_df, UIC, feature_matrix_df)

    #  用户第一次操作 category 到购买之间的天数
    feature_matrix_df = feature_user_category_days_between_1stopt_and_buy(slide_window_df, UIC, feature_matrix_df)
    
    #用户第一次购买 category 前， 在 caetory 上各个 behavior 的数量, 3个特征
    feature_matrix_df = feature_user_category_behavior_cnt_before_1st_buy(slide_window_df, UIC, feature_matrix_df)
    
#     print("%s After extracting user-category features, shape is " % (getCurrentTime()), feature_matrix_df.shape)

    #############################################################################################
    #############################################################################################
    # category 特征
    #############################################################################################
    #############################################################################################
    # category 第一次, 最后一次 behavior 距离checking date 的天数, 以及第一次, 最后一次之间的天数， 返回 12 个特征
    feature_matrix_df = feature_category_days_from_1st_last_behavior(slide_window_df, UIC, feature_matrix_df)
    # category 上各个行为的次数,以及销量(即buy的次数)的排序
    feature_matrix_df = feature_category_behavior_cnt(slide_window_df, slide_window_size, UIC, feature_matrix_df)

    # category 上各个行为用户的数量    
    feature_matrix_df = feature_category_user_cnt_on_behavior(slide_window_df, slide_window_size, UIC, feature_matrix_df)

    # category 在一周内每天各个操作的次数
#     feature_matrix_df = feature_category_behavior_cnt_on_weekday(slide_window_df, UIC, feature_matrix_df)
    
#     print("%s After extracting category features, shape is " % (getCurrentTime()), feature_matrix_df.shape)
    #############################################################################################
    #############################################################################################
    # item 特征
    #############################################################################################
    #############################################################################################
    # item 第一次, 最后一次 behavior 距离checking date 的天数, 以及第一次, 最后一次之间的天数， 返回 12 个特征
    feature_matrix_df = feature_item_days_from_1st_last_behavior(slide_window_df, UIC, feature_matrix_df)
    
    # item 上各个行为的次数,以及销量(即buy的次数)的排序
    feature_matrix_df = feature_item_behavior_cnt(slide_window_df, slide_window_size, UIC, feature_matrix_df)
    
    # item 上各个行为用户的数量
    feature_matrix_df = feature_item_user_cnt_on_behavior(slide_window_df, slide_window_size, UIC, feature_matrix_df)
    
    # item 在一周内每天各个操作的次数
#     feature_matrix_df = feature_item_behavior_cnt_on_weekday(slide_window_df, UIC, feature_matrix_df)
    
#     print("%s After extracting item features, shape is " % (getCurrentTime()), feature_matrix_df.shape)
    
    #############################################################################################
    #############################################################################################
    # user 特征
    #############################################################################################
    #############################################################################################
    # 用户总共有过多少次浏览，收藏，购物车，购买的行为, 购买/浏览， 购买/收藏， 购买/购物车, 购物车/收藏， 购物车/浏览
    # 浏览/行为总数， 收藏/行为总数， 购物车/行为总数， 购买/行为总数
    # 用户购买率：购买的item/操作过的item    
    feature_matrix_df = feature_how_many_behavior_user(slide_window_df, UIC, feature_matrix_df)
    
    # user 浏览，收藏，购物车，购买了多少不同的item
    # user 浏览，收藏，购物车，购买的不同 item 数量 / user 所有操作过的 item 数量
    feature_matrix_df = feature_how_many_itme_user_opted(slide_window_df, UIC, feature_matrix_df)
    
    # user 在前[1,2,3,4, slide_window_size]天24小时上各个操作的次数
    for dayoffset in [1,2,3,4, slide_window_size]:
        feature_matrix_df = feature_user_behavior_before_1day_24hour(slide_window_df, dayoffset, UIC, feature_matrix_df)
    
    # user 在一周内每天各个操作的次数
#     feature_matrix_df = feature_user_behavior_cnt_on_weekday(slide_window_df, UIC, feature_matrix_df)
    
#     print("%s After extracting user features, shape is " % (getCurrentTime()), feature_matrix_df.shape)


    #############################################################################################
    #############################################################################################
    # cross 特征
    #############################################################################################
    #############################################################################################
    
    # item 的销量占 category 的销量的比例, 以及item 销量在category销量中的排序
    feature_matrix_df = feature_sales_ratio_itme_category(feature_matrix_df, slide_window_size)
    
    # item 的1st, last behavior 与 category 的1st， last 相差的天数
    feature_1st_last_IC(feature_matrix_df)
    

    # 商品热度 浏览，收藏，购物车，购买该商品的用户数/浏览，收藏，购物车，购买同类型商品的总用户数
    feature_matrix_df = feature_item_popularity(feature_matrix_df, slide_window_size)

    # item的[fav, cart, buy]转化率/category的购买转化率
    feature_matrix_df = feature_item_conversion(feature_matrix_df)
    
    feature_matrix_df = feature_item_cnt_ratio_on_category(feature_matrix_df, slide_window_size)

    feature_matrix_df.fillna(0, inplace=True)
    feature_matrix_df.replace([np.inf, -np.inf], 0, inplace=True)

    print("%s After extracting cross features, shape is " % (getCurrentTime()), feature_matrix_df.shape)
    #############################################################################################
    #############################################################################################
    # 特征结束
    #############################################################################################
    #############################################################################################
    
#     del feature_matrix_df['user_id']
#     del feature_matrix_df['item_id']
#     del feature_matrix_df['item_category']
#     feature_matrix_df = pd.DataFrame(preprocessing.scale(feature_matrix_df))

    return feature_matrix_df, UIC


