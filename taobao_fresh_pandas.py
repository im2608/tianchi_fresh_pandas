'''
Created on Aug 4, 2017

@author: Heng.Zhang
'''
import os
import sys
import numpy as np
import pandas as pd

import csv
import datetime
from global_variables import *
from common import *
from feature_extraction import *
from greenlet import getcurrent

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold 
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle


def extracting_Y(UI, label_day_df):
    Y = label_day_df[['user_id', 'item_id']][label_day_df['behavior_type'] == 4].drop_duplicates()
    Y['buy'] = 1
    Y.index = range(Y.shape[0])

    Y = pd.merge(UI, Y, how='left', on=['user_id', 'item_id'])
    Y.fillna(0, inplace=True)
    return Y


def calculate_slide_window(slide_window_df, slide_window_size, checking_date, Y_label):    
    print("handling checking date ", checking_date)
    
    feature_matrix_df = extracting_features(slide_window_df)
    
    gbcf = GradientBoostingClassifier(n_estimators=500, max_depth=7, learning_rate=0.1, loss="exponential")
    gbcf.fit(feature_matrix_df, Y_label['buy'])

    del slide_window_df
    return gbcf


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


def single_window():
    data_filename = r"%s\..\input\preprocessed_user_data_no_hour.csv" % (runningPath)
    
    print(getCurrentTime(), "reading csv ", data_filename)

    raw_data_df = pd.read_csv(data_filename)
    
    training_date = '2014-12-18'
    forecasting_date = '2014-12-19'
   
    training_window_df = raw_data_df[raw_data_df['time'] < training_date]
    
    training_window_df = remove_user_item_only_buy(training_window_df)

    training_window_df.index = range(training_window_df.shape[0])  # 重要！！

    slide_window_size = 30
    training_window_df['dayoffset'] = convert_date_str_to_dayoffset(training_window_df['time'], slide_window_size, 
                                                                    datetime.datetime.strptime(training_date, "%Y-%m-%d"))

    training_UI = training_window_df[['user_id', 'item_id']].drop_duplicates()

    Y_label = extracting_Y(training_UI, raw_data_df[raw_data_df['time'] == training_date][['user_id', 'item_id', 'behavior_type']])

    gbcf = calculate_slide_window(training_window_df, slide_window_size, training_date, Y_label)

    training_window_df = raw_data_df
    training_window_df = remove_user_item_only_buy(training_window_df)
    training_window_df.index = range(training_window_df.shape[0])  # 重要！！

    slide_window_size = 30
    training_window_df['dayoffset'] = convert_date_str_to_dayoffset(training_window_df['time'], slide_window_size, training_date)
    del training_window_df['time']

    feature_matrix_df = extracting_features(training_window_df)
    
    UI = training_window_df[['user_id', 'item_id']].drop_duplicates()
    UI.index = range(np.shape(UI)[0])

    Y_fcsted = gbcf.predict(feature_matrix_df)
    
    print('Y_fcsted.shape', Y_fcsted.shape)
    print('UI.shape', UI.shape)
    np.savetxt(r"%s\..\output\fcst.csv" % (runningPath))
    UI = pd.concat([UI, Y_fcsted], axis=1)
    
    outputFile = open(r"%s\..\output\fcst.csv" % (runningPath), encoding="utf-8", mode='w')
    for i in range(UI.shape[0]):
        outputFile.write("%s,%s,%d" % (UI.ix[i]['user_id'], UI.ix[i]['item_id'], Y_fcsted[i]))

    return 0

def slide_window():
    data_filename = r"%s\..\input\preprocessed_user_data_no_hour.csv" % (runningPath)

    print(getCurrentTime(), "reading csv ", data_filename)

    raw_data_df = pd.read_csv(data_filename)
    
    slide_window_size = 7
    start_date = datetime.datetime.strptime('2014-11-18', "%Y-%m-%d")
    end_date = datetime.datetime.strptime('2014-12-18', "%Y-%m-%d")

    print("%s slide window size %d, start date %s, end date %s" % 
          (getCurrentTime(), slide_window_size, convertDatatimeToStr(start_date), convertDatatimeToStr(end_date)))
    
    checking_date = start_date + datetime.timedelta(days = slide_window_size)
    models = []
    while (checking_date < end_date):

        start_date_str = convertDatatimeToStr(start_date)
        checking_date_str = convertDatatimeToStr(checking_date)

        slide_window_df = raw_data_df[(raw_data_df['time'] >= start_date_str ) & (raw_data_df['time'] < checking_date_str)]
        slide_window_df = remove_user_item_only_buy(slide_window_df)

        slide_window_df.index = range(np.shape(slide_window_df)[0])  # 重要！！

        slide_window_df['dayoffset'] = convert_date_str_to_dayoffset(slide_window_df['time'], slide_window_size, checking_date)
        del slide_window_df['time']
        
        training_UI = slide_window_df[['user_id', 'item_id']].drop_duplicates()

        Y_label = extracting_Y(training_UI, raw_data_df[raw_data_df['time'] == checking_date_str][['user_id', 'item_id', 'behavior_type']])

        gbcf = calculate_slide_window(slide_window_df, slide_window_size, checking_date, Y_label)

        models.append(gbcf)

        start_date = start_date + datetime.timedelta(days=1)
        checking_date = start_date + datetime.timedelta(days = slide_window_size)

        break

    return 0




def run_in_ipython():
    df = pd.read_csv(r'F:\doc\ML\taobao\fresh_comp_offline\taobao_fresh_pandas\input\preprocessed_user_data_no_hour.csv')
    slide_window_df = df[df['time'] < '2014-11-25']
    start_date = datetime.datetime.strptime('2014-11-25', "%Y-%m-%d")
    
    slide_window_df['dayoffset'] = convert_date_str_to_dayoffset(slide_window_df['time'], 7, start_date)
    slide_window_df = remove_user_item_only_buy(slide_window_df)
    slide_window_df.index = range(np.shape(slide_window_df)[0])
    
    UIC = slide_window_df[['user_id', 'item_id', 'item_category']].drop_duplicates()
    UIC.index = range(np.shape(UIC)[0])

    feature_matrix_df = pd.DataFrame()
    feature_matrix_df = pd.concat([feature_matrix_df, UIC], axis = 1)

    return

if __name__ == '__main__':
    single_window()
    