'''
Created on Aug 4, 2017

@author: Heng.Zhang
'''

import datetime
import pandas as pd
import numpy as np
import time


def convertDatatimeToStr(opt_datatime):
    return "%04d-%02d-%02d" % (opt_datatime.year, opt_datatime.month, opt_datatime.day)


# 将date string 转成与checking date 之间相差的天数 
def convert_date_str_to_dayoffset(date_str_series, slide_window_size, checking_date):
    slide_window_date_dayoffset = {}
    for i in range(slide_window_size, 0, -1):
        date_in_window = checking_date + datetime.timedelta(days=-i)
        date_str = convertDatatimeToStr(date_in_window)
        slide_window_date_dayoffset[date_str] =  i

    dayoffset_series = pd.Series(list(map(lambda x : slide_window_date_dayoffset[x], date_str_series)))
    
    dayoffset_series.index = range(dayoffset_series.shape[0])

    return dayoffset_series
        
def SeriesDivision(divisor, dividend):
    quotient = divisor / dividend
    quotient.fillna(0, inplace=True)
    quotient[np.isinf(quotient)] = 0
    return quotient


rename_item_col_name = lambda col_name: "item_" + col_name if (col_name != 'item_id' and col_name != 'user_id' and col_name != 'item_category') else col_name
rename_category_col_name = lambda col_name: "category_" + col_name if (col_name != 'item_id' and col_name != 'user_id' and col_name != 'item_category') else col_name


def remove_user_item_only_buy(slide_window_df):
    # 得到user在item上各个操作的次数,删除只有购买而没有其他操作的数据
    user_only_buyopt_df = slide_window_df[['user_id', 'item_id', 'behavior_type']]
    user_only_buyopt_df = user_only_buyopt_df.groupby(['user_id', 'item_id', 'behavior_type'], as_index=False, sort=False)
    user_only_buyopt_df = user_only_buyopt_df.size().unstack()
    user_only_buyopt_df.fillna(0, inplace=True)
    user_only_buyopt_df.reset_index(inplace=True)
    user_item_onlybuy_df = user_only_buyopt_df[['user_id', 'item_id']][(user_only_buyopt_df[1] == 0) &
                                                                       (user_only_buyopt_df[2] == 0) &
                                                                       (user_only_buyopt_df[3] == 0) & 
                                                                       ((user_only_buyopt_df[4] != 0))]
    # user只是购买了item，但没有其他操作，删除这些数据
    user_item_onlybuy_df['user_item_onlybuy'] = 1

    slide_window_df = pd.merge(slide_window_df, user_item_onlybuy_df, how='left', on=['user_id', 'item_id'])
    slide_window_df.drop(slide_window_df[slide_window_df['user_item_onlybuy'] == 1].index, inplace=True)
    del slide_window_df['user_item_onlybuy']

    return slide_window_df


def getCurrentTime():
    return time.strftime("%Y-%m-%d %X", time.localtime())


def extracting_Y(UI, label_day_df):
    Y = label_day_df[['user_id', 'item_id']][label_day_df['behavior_type'] == 4].drop_duplicates()
    Y['buy'] = 1
    Y.index = range(Y.shape[0])

    Y = pd.merge(UI, Y, how='left', on=['user_id', 'item_id'])
    Y.fillna(0, inplace=True)
    return Y
