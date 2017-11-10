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

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold 
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix



def splitTo50_50(feature_matrix_df):
    samples_for_1 = feature_matrix_df.sample(frac=0.5, axis=0)
    
    index_for_gbdt = feature_matrix_df.index.difference(samples_for_1.index)
    samples_for_2 = feature_matrix_df.ix[index_for_gbdt]
    
    samples_for_1.index = range(samples_for_1.shape[0])
    samples_for_2.index = range(samples_for_2.shape[0])
    
    return samples_for_1, samples_for_2

def takeSamples(feature_matrix_df, checking_date_str):
    pos = feature_matrix_df[feature_matrix_df['buy'] == 1]
    print("%s samples POS:NAG = 1:%d (%d : %d) %s" % (getCurrentTime(), g_nag_times, pos.shape[0], pos.shape[0] * g_nag_times, checking_date_str))
    nag = feature_matrix_df[feature_matrix_df['buy'] == 0].sample(n = pos.shape[0] * g_nag_times, axis=0)
 
    # 正负样本的比例 1:g_nag_times
    samples = pd.concat([pos, nag], axis=0)
    
    return samples

def trainingModel(feature_matrix_df, checking_date_str):
    gbcf_1 = GradientBoostingClassifier(n_estimators=80, # gride searched to 80
                                       subsample=1.0, 
                                       criterion='friedman_mse', 
                                       min_samples_split=100, 
                                       min_samples_leaf=1,
                                       max_depth=3) # gride searched to 3

    features_names_for_model = get_feature_name_for_model(feature_matrix_df.columns)
    min_max_scaler = preprocessing.MinMaxScaler()

    samples_for_1, samples_for_2 = splitTo50_50(feature_matrix_df)
    samples_for_1 = takeSamples(samples_for_1, checking_date_str)
    samples_for_2 = takeSamples(samples_for_2, checking_date_str)

    print(getCurrentTime(), "training GBDT 1...")
    gbcf_1.fit(samples_for_1[features_names_for_model], samples_for_1['buy'])
    
    Y_fcsted_gbdt = pd.DataFrame(gbcf_1.predict(samples_for_2[features_names_for_model]), columns=['buy'])
#     print("%s GBDT 1st fit the training date: " % getCurrentTime())
#     print(classification_report(samples_for_2['buy'], Y_fcsted_gbdt['buy'], target_names=["not buy", "buy"]))
#     print("%s confusion matrix of LR: " % getCurrentTime())
#     print(confusion_matrix(samples_for_2['buy'],  Y_fcsted_gbdt['buy']))

    gbcf_2 = GradientBoostingClassifier(n_estimators=80, # gride searched to 80
                                        subsample=1.0, 
                                        criterion='friedman_mse', 
                                        min_samples_split=100, 
                                        min_samples_leaf=1,
                                        max_depth=3) # gride searched to 3

#     features_for_gbdt = min_max_scaler.fit_transform(samples_for_gbdt[features_names_for_model])

    print(getCurrentTime(), "training GBDT 2...")
    gbcf_2.fit(samples_for_2[features_names_for_model], Y_fcsted_gbdt['buy'])
    
#     gbcf_pre = gbcf_2.predict(samples_for_1[features_names_for_model])    
#     print("%s GradientBoostingClassifier() fit the training date: " % getCurrentTime())
#     print(classification_report(samples_for_1['buy'], gbcf_pre, target_names=["not buy", "buy"]))
#     print("%s confusion matrix of GBDT: " % getCurrentTime())
#     print(confusion_matrix(samples_for_1['buy'], gbcf_pre))

    return gbcf_1, gbcf_2

def calculate_slide_window(raw_data_df, slide_window_size, window_start_date, checking_date):    
    checking_date_str = convertDatatimeToStr(checking_date)
    start_date_str = convertDatatimeToStr(window_start_date)

    print("%s handling slide window %s" % (getCurrentTime(), checking_date_str))

    model_1_filename = r"%s\..\featuremat_and_model\model_1_%s_%s_%d.m" % (runningPath, start_date_str, checking_date_str, slide_window_size)
    model_2_filename = r"%s\..\featuremat_and_model\model_2_%s_%s_%d.m" % (runningPath, start_date_str, checking_date_str, slide_window_size)

    slide_window_df = create_slide_window_df(raw_data_df, window_start_date, checking_date, slide_window_size, None)
    slide_UI = slide_window_df[['user_id', 'item_id']].drop_duplicates()
 
    Y_label = extracting_Y(slide_UI, raw_data_df[raw_data_df['time'] == checking_date_str][['user_id', 'item_id', 'behavior_type']])
    
    feature_mat_filename = r"%s\..\featuremat_and_model\feature_mat_%s_%d.csv" % (runningPath, checking_date_str, slide_window_size)
    if (os.path.exists(feature_mat_filename)):
        print("%s reading feature matrix from: %s_%d.csv" % (getCurrentTime(), checking_date_str, slide_window_size))
        feature_matrix_df = pd.read_csv(feature_mat_filename)
    else:
        feature_matrix_df, UIC = extracting_features(slide_window_df, slide_window_size, None)
        feature_matrix_df = pd.concat([feature_matrix_df, Y_label['buy']], axis=1)

    gbcf_1, gbcf_2 = trainingModel(feature_matrix_df, checking_date_str)

#     # 保存特征矩阵
#     if (not os.path.exists(feature_mat_filename)):
#         print("%s feature matrix to %s" % (getCurrentTime(), feature_mat_filename))
#         pd.DataFrame(feature_matrix_df).to_csv(feature_mat_filename, index=False)
#         
    # 保存model
    print("%s dumping model 1to %s" % (getCurrentTime(), model_1_filename))
    joblib.dump(gbcf_1, model_1_filename)
    print("%s dumping model 2 to %s" % (getCurrentTime(), model_2_filename))
    joblib.dump(gbcf_2, model_2_filename)

    del slide_window_df

    return gbcf_1, gbcf_2

def single_window():
    start_date_str = sys.argv[1].split("=")[1]
    end_date_str = sys.argv[2].split("=")[1]
    slide_window_size = int(sys.argv[3].split("=")[1])
    
    print("running for %s, %s" % (start_date_str, end_date_str))

#     data_filename = r"%s\..\input\preprocessed_user_data.csv" % (runningPath)
#     data_filename = r"%s\..\input\preprocessed_user_data_fcsted_item_only.csv" % (runningPath)
    data_filename = r"%s\..\input\preprocessed_user_data_sold_item_only.csv" % (runningPath)

    print(getCurrentTime(), "reading csv ", data_filename)
    raw_data_df = pd.read_csv(data_filename, dtype={'user_id':np.str, 'item_id':np.str})

    training_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")    
    window_start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")

    # training...
    calculate_slide_window(raw_data_df, slide_window_size, window_start_date, training_date)  

if __name__ == '__main__':
    single_window()
#     slide_window()
    