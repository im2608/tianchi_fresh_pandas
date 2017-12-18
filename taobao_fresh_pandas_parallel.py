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

    feature_matrix_df, UIC = extracting_features(slide_window_df, slide_window_size, None)
    Y_label = extracting_Y(UIC, raw_data_df[raw_data_df['time'] == checking_date_str])
    feature_matrix_df = pd.concat([feature_matrix_df, Y_label['buy']], axis=1)

    gbcf_1, gbcf_2 = trainingModel_2(feature_matrix_df, checking_date_str)

    del slide_window_df

    return gbcf_1, gbcf_2

def single_window():
    input_path =  r"%s\..\input" % (runningPath)
     
    start_date_str = sys.argv[1].split("=")[1]
    fcsting_date_str = sys.argv[2].split("=")[1]
    slide_window_size = int(sys.argv[3].split("=")[1])
    
    window_start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    training_date = window_start_date + datetime.timedelta(days=slide_window_size)
    training_date_str = convertDatatimeToStr(training_date)

    print(getCurrentTime(), "running for %s, %s" % (start_date_str, training_date_str))

#     data_filename = r"%s\preprocessed_user_data.csv" % (input_path)
    data_filename = r"%s\preprocessed_user_data_fcsted_item_only.csv" % (input_path)
#     data_filename = r"%s\preprocessed_user_data_sold_item_only_no1212.csv" % (input_path)

    print(getCurrentTime(), "reading csv ", data_filename)
    raw_data_df = pd.read_csv(data_filename, dtype={'user_id':np.str, 'item_id':np.str})

    # training...
    gbcf_1, gbcf_2 = calculate_slide_window(raw_data_df, slide_window_size, window_start_date, training_date)  

    # creating feature matrix for forecasting...
    fcsted_item_filename = r"%s\tianchi_fresh_comp_train_item.csv" % (input_path)
    print(getCurrentTime(), "reading being forecasted items ", fcsted_item_filename)
    fcsted_item_df = pd.read_csv(fcsted_item_filename, dtype={'item_id':np.str})

    forecasting_date = datetime.datetime.strptime(fcsting_date_str, "%Y-%m-%d")
    window_start_date = forecasting_date - datetime.timedelta(days=slide_window_size) 
    print("%s forecasting for %s , slide windows %d" % (getCurrentTime(), fcsting_date_str, slide_window_size))

    # forecasting...
    data_filename = r"%s\preprocessed_user_data_fcsted_item_only.csv" % (input_path)
    raw_data_df = pd.read_csv(data_filename, dtype={'user_id':np.str, 'item_id':np.str})
    fcsting_window_df = create_slide_window_df(raw_data_df, window_start_date, forecasting_date, slide_window_size, fcsted_item_df)

    fcsting_matrix_df, fcsting_UI = extracting_features(fcsting_window_df, slide_window_size, fcsted_item_df)

    Y_training_label = extracting_Y(fcsting_UI, raw_data_df[raw_data_df['time'] == fcsting_date_str][['user_id', 'item_id', 'behavior_type']])
    fcsting_matrix_df = pd.concat([fcsting_matrix_df, Y_training_label['buy']], axis=1)
    
    features_names_for_model = get_feature_name_for_model(fcsting_matrix_df.columns)    
    
    fcsting_mat = xgb.DMatrix(fcsting_matrix_df[features_names_for_model])

    if (len(gbcf_1.feature_names) == len(fcsting_mat.feature_names)):
        for i in range(0, len(gbcf_1.feature_names)):
            if (gbcf_1.feature_names[i] != fcsting_mat.feature_names[i]):
                print(i, gbcf_1.feature_names[i], fcsting_mat.feature_names[i])
    else:
        print("EXception: %d != %d" % (len(gbcf_1.feature_names), len(fcsting_mat.feature_names)))                

    Y_gbdt1_predicted = pd.DataFrame(gbcf_1.predict(fcsting_mat), columns=['buy_proba'])
    fcsted_index_1 = Y_gbdt1_predicted[Y_gbdt1_predicted['buy_proba'] >= g_min_prob].index

    fcsting_mat = xgb.DMatrix(fcsting_matrix_df[features_names_for_model].iloc[fcsted_index_1])    
    Y_gbdt2_predicted = pd.DataFrame(gbcf_2.predict(fcsting_mat), columns=['buy_proba'])
    Y_gbdt2_predicted = Y_gbdt2_predicted.sort_values('buy_proba',  axis=0, ascending=False)

    # 取前700
    fcsted_index_2 = Y_gbdt2_predicted[0:700].index

    output_filename = r"%s\..\output\subprocess\%s_%d_%s.csv" % (runningPath, start_date_str, slide_window_size, fcsting_date_str)
    print("%s output forecasting to %s" % (getCurrentTime(), output_filename))
    fcsted_ui = fcsting_matrix_df.iloc[fcsted_index_1[fcsted_index_2]][['user_id', 'item_id']]
    fcsted_ui.index = range(0, fcsted_ui.shape[0])
    fcsted_proba =  Y_gbdt2_predicted[0:700]['buy_proba']
    fcsted_proba.index = range(0, fcsted_proba.shape[0])
    fcsted_ui_proba = pd.concat([fcsted_ui, fcsted_proba], axis=1, ignore_index=True)
    fcsted_ui_proba.columns = ['user_id', 'item_id', 'proba']
    fcsted_ui_proba.to_csv(output_filename, index=False)

    output_filename = r"%s\..\output\subprocess\%s_%d_%s.mod1.csv" % (runningPath, start_date_str, slide_window_size, fcsting_date_str)
    fcsted_ui_mod1 = fcsting_matrix_df.iloc[fcsted_index_1][['user_id', 'item_id']]
    fcsted_ui_mod1.index = range(0, fcsted_ui_mod1.shape[0])
    fcsted_proba_mod1 =  Y_gbdt1_predicted[0:700]['buy_proba']
    fcsted_proba_mod1.index = range(0, fcsted_proba_mod1.shape[0])
    fcsted_ui_proba_mod1 = pd.concat([fcsted_ui_mod1, fcsted_proba_mod1], axis=1, ignore_index=True)
    fcsted_ui_proba_mod1.columns = ['user_id', 'item_id', 'proba']
    fcsted_ui_proba_mod1.to_csv(output_filename, index=False)

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
    
    fcsting_mat = xgb.DMatrix(verifying_matrix_df[features_names_for_model])

    Y_gbdt1_predicted = pd.DataFrame(gbcf_1.predict(fcsting_mat), columns=['buy_proba'])   
    fcsted_index_1 = Y_gbdt1_predicted[Y_gbdt1_predicted['buy_proba']>=g_min_prob].index

    fcsting_mat = xgb.DMatrix(verifying_matrix_df[features_names_for_model].iloc[fcsted_index_1])
    Y_gbdt2_predicted = pd.DataFrame(gbcf_2.predict(fcsting_mat), columns=['buy_proba'])
    Y_gbdt2_predicted = Y_gbdt2_predicted.sort_values('buy_proba', axis=0, ascending=False) 

    fcsted_index_2 = Y_gbdt2_predicted[0:700].index

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
 
    fcsted_ui = verifying_matrix_df.ix[fcsted_index_1][['user_id', 'item_id']]
    fcsted_ui.index = range(0, fcsted_ui.shape[0])                            
    p, r, f1 = calculate_POS_F1(Y_true_UI, fcsted_ui)
    print("%s slide window %s, %d, verified with MOD1 %s, p %.6f, r %.6f, f1 %.6f" % 
          (getCurrentTime(),  start_date_str, slide_window_size, verifying_date_str, p, r, f1))
 
    return 0
   
if __name__ == '__main__':
    single_window()
    