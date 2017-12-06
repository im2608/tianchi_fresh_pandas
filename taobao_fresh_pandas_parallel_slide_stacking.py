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
from feature_extraction import *


# window_start_date 滑窗的开始日期
# slide_window_size 滑窗的天数
# 滑窗的开始日期 + 滑窗的天数 = label date
# 滑窗 = [滑窗的开始日期, label date)

def create_feature_matrix():
    start_date_str = sys.argv[1].split("=")[1]
    slide_window_size = int(sys.argv[2].split("=")[1])
    
    window_start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    checking_date = window_start_date + datetime.timedelta(days=slide_window_size)
    checking_date_str = convertDatatimeToStr(checking_date)
    
    feature_mat_filename = r"%s\..\featuremat_and_model\feature_mat_%s_%d.csv" % (runningPath, start_date_str, slide_window_size)
    if (os.path.exists(feature_mat_filename)):
        print("feature matrix %s, is existing, exiting..." % feature_mat_filename)
        return

    print("creating feature matrix for %s, %s" % (start_date_str, checking_date_str))

    data_filename = r"%s\..\input\preprocessed_user_data.csv" % (runningPath)
#     data_filename = r"%s\..\input\preprocessed_user_data_fcsted_item_only.csv" % (runningPath)
#     data_filename = r"%s\..\input\preprocessed_user_data_sold_item_only_no1212.csv" % (runningPath)

    print(getCurrentTime(), "reading csv ", data_filename)
    raw_data_df = pd.read_csv(data_filename, dtype={'user_id':np.str, 'item_id':np.str})

    # training...
    slide_window_df = create_slide_window_df(raw_data_df, window_start_date, checking_date, slide_window_size, None)
    slide_UI = slide_window_df[['user_id', 'item_id']].drop_duplicates()

    feature_matrix_df, UIC = extracting_features(slide_window_df, slide_window_size, None)
    Y_label = extracting_Y(UIC, raw_data_df[raw_data_df['time'] == checking_date_str])
    feature_matrix_df = pd.concat([feature_matrix_df, Y_label['buy']], axis=1)
    print("output feature matrix to ", feature_mat_filename)
    feature_matrix_df.to_csv(feature_mat_filename, index=False)
    return 0

   
if __name__ == '__main__':
    create_feature_matrix()
    