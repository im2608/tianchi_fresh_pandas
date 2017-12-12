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


from sklearn.cross_validation import StratifiedKFold 
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from adodbapi.apibase import variantConvertDate



def calculate_slide_window(raw_data_df, slide_window_size, window_start_date, checking_date):    
    checking_date_str = convertDatatimeToStr(checking_date)
    start_date_str = convertDatatimeToStr(window_start_date)

    print("%s handling slide window %s" % (getCurrentTime(), checking_date_str))

    slide_window_df = create_slide_window_df(raw_data_df, window_start_date, checking_date, slide_window_size, None)

    feature_mat_filename = r"%s\..\featuremat_and_model\feature_mat_%s_%d.csv" % (runningPath, checking_date_str, slide_window_size)
    if (os.path.exists(feature_mat_filename)):
        print("%s reading feature matrix from: %s_%d.csv" % (getCurrentTime(), checking_date_str, slide_window_size))
        feature_matrix_df = pd.read_csv(feature_mat_filename)
    else:
        feature_matrix_df, UIC = extracting_features(slide_window_df, slide_window_size, None)
        Y_label = extracting_Y(UIC, raw_data_df[raw_data_df['time'] == checking_date_str])
        feature_matrix_df = pd.concat([feature_matrix_df, Y_label['buy']], axis=1)

    gbcf_1, gbcf_2 = trainingModel_2(feature_matrix_df, checking_date_str)

    feature_import_filename_1 = r"%s\..\featuremat_and_model\feature_importance_1_%s_%d.txt" % (runningPath, checking_date_str, slide_window_size)
    feature_import_filename_2 = r"%s\..\featuremat_and_model\feature_importance_2_%s_%d.txt" % (runningPath, checking_date_str, slide_window_size)
    
    features_names_for_model = get_feature_name_for_model(feature_matrix_df.columns)
    file_handle = open(feature_import_filename_1, encoding="utf-8", mode='w')
    for idx, importance in enumerate(gbcf_1.feature_importances_):
        file_handle.write("%s: %.4f\n"% (features_names_for_model[idx], importance))
    file_handle.close()

    file_handle = open(feature_import_filename_2, encoding="utf-8", mode='w')
    for idx, importance in enumerate(gbcf_1.feature_importances_):
        file_handle.write("%s: %.4f\n"% (features_names_for_model[idx], importance))
    file_handle.close()

#     # 保存特征矩阵
#     if (not os.path.exists(feature_mat_filename)):
#         print("%s feature matrix to %s" % (getCurrentTime(), feature_mat_filename))
#         pd.DataFrame(feature_matrix_df).to_csv(feature_mat_filename, index=False)
#         
    # 保存model
#     model_1_filename = r"%s\..\featuremat_and_model\model_1_%s_%s_%d.m" % (runningPath, start_date_str, checking_date_str, slide_window_size)
#     model_2_filename = r"%s\..\featuremat_and_model\model_2_%s_%s_%d.m" % (runningPath, start_date_str, checking_date_str, slide_window_size)    
#     print("%s dumping model 1to %s" % (getCurrentTime(), model_1_filename))
#     joblib.dump(gbcf_1, model_1_filename)
#     print("%s dumping model 2 to %s" % (getCurrentTime(), model_2_filename))
#     joblib.dump(gbcf_2, model_2_filename)

    del slide_window_df

    return gbcf_1, gbcf_2

def single_window():
    start_date_str = sys.argv[1].split("=")[1]
    fcsting_date_str = sys.argv[2].split("=")[1]
    slide_window_size = int(sys.argv[3].split("=")[1])
    
    window_start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    training_date = window_start_date + datetime.timedelta(days=slide_window_size)
    training_date_str = convertDatatimeToStr(training_date)

    print(getCurrentTime(), "running for %s, %s" % (start_date_str, training_date_str))

    data_filename = r"%s\..\input\preprocessed_user_data.csv" % (runningPath)
#     data_filename = r"%s\..\input\preprocessed_user_data_fcsted_item_only.csv" % (runningPath)
#     data_filename = r"%s\..\input\preprocessed_user_data_sold_item_only_no1212.csv" % (runningPath)

    print(getCurrentTime(), "reading csv ", data_filename)
    raw_data_df = pd.read_csv(data_filename, dtype={'user_id':np.str, 'item_id':np.str})

    # training...
    gbcf_1, gbcf_2 = calculate_slide_window(raw_data_df, slide_window_size, window_start_date, training_date)  

    # creating feature matrix for forecasting...
    fcsted_item_filename = r"%s\..\input\tianchi_fresh_comp_train_item.csv" % (runningPath)
    print(getCurrentTime(), "reading being forecasted items ", fcsted_item_filename)
    fcsted_item_df = pd.read_csv(fcsted_item_filename, dtype={'item_id':np.str})

    forecasting_date = datetime.datetime.strptime(fcsting_date_str, "%Y-%m-%d")
    window_start_date = forecasting_date - datetime.timedelta(days=4) # only consider user-item that interacted within last 4 days 
    print("%s forecasting for %s , slide windows %d" % (getCurrentTime(), fcsting_date_str, slide_window_size))

    # forecasting...
    data_filename = r"%s\..\input\preprocessed_user_data_fcsted_item_only.csv" % (runningPath)
    raw_data_df = pd.read_csv(data_filename, dtype={'user_id':np.str, 'item_id':np.str})
    fcsting_window_df = create_slide_window_df(raw_data_df, window_start_date, forecasting_date, slide_window_size, fcsted_item_df)

    fcsting_matrix_df, fcsting_UI = extracting_features(fcsting_window_df, slide_window_size, fcsted_item_df)

    Y_training_label = extracting_Y(fcsting_UI, raw_data_df[raw_data_df['time'] == fcsting_date_str][['user_id', 'item_id', 'behavior_type']])
    fcsting_matrix_df = pd.concat([fcsting_matrix_df, Y_training_label['buy']], axis=1)
    
    features_names_for_model = get_feature_name_for_model(fcsting_matrix_df.columns)    

    Y_gbdt1_predicted = pd.DataFrame(gbcf_1.predict_proba(fcsting_matrix_df[features_names_for_model]), columns=['not buy', 'buy'])    

    fcsted_index_1 = Y_gbdt1_predicted[Y_gbdt1_predicted['buy'] >= g_min_prob].index

    Y_gbdt2_predicted = pd.DataFrame(gbcf_2.predict_proba(fcsting_matrix_df[features_names_for_model].ix[fcsted_index_1]), columns=['not buy', 'buy'])

    fcsted_index_2 = Y_gbdt2_predicted[Y_gbdt2_predicted['buy'] >= g_min_prob].index

    output_filename = r"%s\..\output\subprocess\%s_%d_%s.csv" % (runningPath, start_date_str, slide_window_size, fcsting_date_str)
    print("%s output forecasting to %s" % (getCurrentTime(), output_filename))
    fcsted_ui = fcsting_matrix_df.ix[fcsted_index_1[fcsted_index_2]][['user_id', 'item_id']]
    fcsted_ui.index = range(0, fcsted_ui.shape[0])
    fcsted_proba =  Y_gbdt2_predicted[Y_gbdt2_predicted['buy'] >= g_min_prob]['buy']
    fcsted_proba.index = range(0, fcsted_proba.shape[0])
    fcsted_ui_proba = pd.concat([fcsted_ui, fcsted_proba], axis=1, ignore_index=True)
    fcsted_ui_proba.columns = ['user_id', 'item_id', 'proba']
    fcsted_ui_proba.to_csv(output_filename, index=False)
    
    calculate_precission(gbcf_1, gbcf_2, 
                         start_date_str, slide_window_size, 
                         forecasting_date, 
                         raw_data_df, fcsted_item_df, data_filename)
    return 0

def calculate_precission(gbcf_1, gbcf_2, start_date_str, slide_window_size, forecasting_date, raw_data_df, fcsted_item_df, data_filename):
    print(getCurrentTime(), "calculate_precission ...")
    forecasting_date_str = convertDatatimeToStr(forecasting_date)
    
    verifying_date = forecasting_date - datetime.timedelta(days=1)
    verifying_start_date = verifying_date - datetime.timedelta(days=slide_window_size)
    verifying_date_str = convertDatatimeToStr(verifying_date)
    
    Y_true_UI = raw_data_df[(raw_data_df['time'] == verifying_date_str)&\
                            (raw_data_df['behavior_type'] == 4) &
                            (np.in1d(raw_data_df['item_id'], fcsted_item_df['item_id']))][['user_id', 'item_id']].drop_duplicates()

    verifying_window_df = create_slide_window_df(raw_data_df, verifying_start_date, verifying_date, slide_window_size, fcsted_item_df)

    verifying_matrix_df, verifying_UI = extracting_features(verifying_window_df, slide_window_size, fcsted_item_df)

    Y_verify_label = extracting_Y(verifying_UI, raw_data_df[raw_data_df['time'] == verifying_date_str][['user_id', 'item_id', 'behavior_type']])
    verifying_matrix_df = pd.concat([verifying_matrix_df, Y_verify_label['buy']], axis=1)

    # forecasting...
    features_names_for_model = get_feature_name_for_model(verifying_matrix_df.columns)    

    Y_gbdt1_predicted = pd.DataFrame(gbcf_1.predict_proba(verifying_matrix_df[features_names_for_model]), columns=['not buy', 'buy'])    

    fcsted_index_1 = Y_gbdt1_predicted[Y_gbdt1_predicted['buy'] >= g_min_prob].index

    Y_gbdt2_predicted = pd.DataFrame(gbcf_2.predict_proba(verifying_matrix_df[features_names_for_model].ix[fcsted_index_1]), columns=['not buy', 'buy'])

    fcsted_index_2 = Y_gbdt2_predicted[Y_gbdt2_predicted['buy'] >= g_min_prob].index

    fcsted_ui = verifying_matrix_df.ix[fcsted_index_1[fcsted_index_2]][['user_id', 'item_id']]
    fcsted_ui.index = range(0, fcsted_ui.shape[0])                            
    p, r, f1 = calculate_POS_F1(Y_true_UI, fcsted_ui)
    
    print("%s slide window %s, %d, verified with %s, p %.6f, r %.6f, f1 %.6f" % 
          (getCurrentTime(),  start_date_str, slide_window_size, verifying_date_str, p, r, f1))
    output_filename = r"%s\..\output\subprocess\%s_%d_%s_p_r_f1.csv" % (runningPath, start_date_str, slide_window_size, forecasting_date_str)
    file_handle = open(output_filename, encoding="utf-8", mode='w')
    file_handle.write("%.6f,%.6f,%.6f\n" % (p, r, f1))
#     file_handle.write("using file %s" % data_filename)    
    file_handle.close()
 
    return 0
   
if __name__ == '__main__':
    single_window()
    