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

def takeSamples(feature_matrix_df):
    pos = feature_matrix_df[feature_matrix_df['buy'] == 1]
    print("%s samples POS:NAG = 1:%d (%d : %d)" % (getCurrentTime(), g_nag_times, pos.shape[0], pos.shape[0] * g_nag_times))
    nag = feature_matrix_df[feature_matrix_df['buy'] == 0].sample(n = pos.shape[0] * g_nag_times, axis=0)
 
    # 正负样本的比例 1:g_nag_times
    samples = pd.concat([pos, nag], axis=0)
    
    return samples

def trainingModel(feature_matrix_df):
    gbcf_1 = GradientBoostingClassifier(n_estimators=80, # gride searched to 80
                                       subsample=1.0, 
                                       criterion='friedman_mse', 
                                       min_samples_split=100, 
                                       min_samples_leaf=1,
                                       max_depth=3) # gride searched to 3

    features_names_for_model = get_feature_name_for_model(feature_matrix_df.columns)
    min_max_scaler = preprocessing.MinMaxScaler()

    samples_for_1, samples_for_2 = splitTo50_50(feature_matrix_df)
    samples_for_1 = takeSamples(samples_for_1)
    samples_for_2 = takeSamples(samples_for_2)

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
    print("%s handling slide window %s" % (getCurrentTime(), checking_date_str))

    model_filename = r"%s\..\featuremat_and_model\model_%s_%d.m" % (runningPath, checking_date_str, slide_window_size)
    if (os.path.exists(model_filename)):
        print("%s loading model from model_%s_%d.m" % (getCurrentTime(), checking_date_str, slide_window_size))
        gbcf = joblib.load(model_filename)
        return gbcf

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

    gbcf_1, gbcf_2 = trainingModel(feature_matrix_df)

#     # 保存特征矩阵
#     if (not os.path.exists(feature_mat_filename)):
#         print("%s feature matrix to %s" % (getCurrentTime(), feature_mat_filename))
#         pd.DataFrame(feature_matrix_df).to_csv(feature_mat_filename, index=False)
#         
#     # 保存model
#     if (not os.path.exists(model_filename)):
#         print("%s dumping model to %s" % (getCurrentTime(), model_filename))
#         joblib.dump(gbcf, model_filename)

    del slide_window_df

    return gbcf_1, gbcf_2

def single_window():

    slide_window_size = 30
#     data_filename = r"%s\..\input\preprocessed_user_data.csv" % (runningPath)
    data_filename = r"%s\..\input\preprocessed_user_data_fcsted_item_only.csv" % (runningPath)
#     data_filename = r"%s\..\input\preprocessed_user_data_sold_item_only.csv" % (runningPath)

    print(getCurrentTime(), "reading csv ", data_filename)
    raw_data_df = pd.read_csv(data_filename, dtype={'user_id':np.str, 'item_id':np.str})

    fcsted_item_filename = r"%s\..\input\tianchi_fresh_comp_train_item.csv" % (runningPath)
    print(getCurrentTime(), "reading being forecasted items ", fcsted_item_filename)
    fcsted_item_df = pd.read_csv(fcsted_item_filename, dtype={'item_id':np.str})

    training_date = datetime.datetime.strptime('2014-12-15', "%Y-%m-%d")
    window_start_date = training_date - datetime.timedelta(days=slide_window_size)

    # training...
    gbcf_1, gbcf_2 = calculate_slide_window(raw_data_df, slide_window_size, window_start_date, training_date)
    print("GradientBoostingClassifier() params: ", gbcf_1.get_params())

    # forecasting...
    forecasting_date = training_date + datetime.timedelta(days=1)
    forecasting_date_str = convertDatatimeToStr(forecasting_date)

    window_start_date = forecasting_date - datetime.timedelta(days=slide_window_size)
    
    forecasting_window_df = create_slide_window_df(raw_data_df, window_start_date, forecasting_date, slide_window_size, fcsted_item_df)

    forecasting_feature_matrix_df, forecasting_UI = extracting_features(forecasting_window_df, slide_window_size, fcsted_item_df)
    
    features_names_for_model = get_feature_name_for_model(forecasting_feature_matrix_df.columns)
    features_for_model = forecasting_feature_matrix_df[features_names_for_model]    
#     min_max_scaler = preprocessing.MinMaxScaler()
#     features_for_model = pd.DataFrame(min_max_scaler.fit_transform(forecasting_feature_matrix_df[features_names_for_model]))
    
    # 使用gbdt1过滤数据, 只将预测为正的数据传给gbdt2预测
    Y_fcsted_gbdt = pd.DataFrame(gbcf_1.predict_proba(features_for_model), columns=['not buy', 'buy'])    
    
    fcsted_index = Y_fcsted_gbdt[Y_fcsted_gbdt['buy'] >= g_min_prob].index
    features_for_model = features_for_model.ix[fcsted_index]    
    features_for_model.index = range(features_for_model.shape[0])
    
    fcsted_UI = forecasting_UI.ix[fcsted_index]
    fcsted_UI.index = range(fcsted_UI.shape[0]) 

    Y_fcsted = gbcf_2.predict_proba(features_for_model)

    print('%s fcsted_UI.shape %s' %(getCurrentTime(), fcsted_UI.shape))

    Y_fcsted_lable = pd.DataFrame(Y_fcsted, columns=['not_buy_prob', 'buy_prob'])
    Y_fcsted_lable = pd.concat([fcsted_UI, Y_fcsted_lable], axis=1)
    
    use_rule = 1
    if (use_rule):
        # 规则： 如果user 在  checking date 前一天 cart, 并且没有购买 ，则认为他checking date 会购买
        Y_fcsted_lable = rule_fav_cart_before_1day(forecasting_window_df, Y_fcsted_lable)

    if (forecasting_date_str == '2014-12-19'):
        UI_buy_allinfo = Y_fcsted_lable[Y_fcsted_lable['buy_prob'] >= g_min_prob]
        index = 0
        prob_output_filename, submit_output_filename = get_output_filename(index, use_rule)
        while (os.path.exists(submit_output_filename)):
            index += 1
            prob_output_filename, submit_output_filename = get_output_filename(index, use_rule)

        UI_buy_allinfo.to_csv(prob_output_filename, index=False)
        UI_buy_allinfo[['user_id', 'item_id']].to_csv(submit_output_filename, index=False)
    else:
        Y_true_UI = raw_data_df[(raw_data_df['time'] == forecasting_date_str)&\
                                (raw_data_df['behavior_type'] == 4)].drop_duplicates()
#                                 (np.in1d(raw_data_df['item_id'], fcsted_item_df['item_id']))][['user_id', 'item_id']].drop_duplicates()

        Y_fcsted_UI = Y_fcsted_lable[Y_fcsted_lable['buy_prob'] >= g_min_prob]
        p, r, f1 = calculate_POS_F1(Y_true_UI, Y_fcsted_UI)
        
        print("%s precision: %.4f, recall %.4f, F1 %.4f" % (getCurrentTime(), p, r, f1))

    return 0

def slide_window():
    
#     data_filename = r"%s\..\input\preprocessed_user_data.csv" % (runningPath)

    data_fcsted_item_only = 0
    if (data_fcsted_item_only):
        data_filename = r"%s\..\input\preprocessed_user_data_fcsted_item_only.csv" % (runningPath)
    else:
        data_filename = r"%s\..\input\preprocessed_user_data_sold_item_only.csv" % (runningPath)

    print(getCurrentTime(), "reading csv ", data_filename)
    raw_data_df = pd.read_csv(data_filename, dtype={'user_id':np.str, 'item_id':np.str})

    fcsted_item_filename = r"%s\..\input\tianchi_fresh_comp_train_item.csv" % (runningPath)
    print(getCurrentTime(), "reading being forecasted items ", fcsted_item_filename)
    fcsted_item_df = pd.read_csv(fcsted_item_filename, dtype={'item_id':np.str})

    slide_window_size = 7
    start_date = datetime.datetime.strptime('2014-12-06', "%Y-%m-%d")
    end_date = datetime.datetime.strptime('2014-12-15', "%Y-%m-%d")

    print("%s slide window size %d, start date %s, end date %s" % 
          (getCurrentTime(), slide_window_size, convertDatatimeToStr(start_date), convertDatatimeToStr(end_date)))

    checking_date = start_date + datetime.timedelta(days = slide_window_size)
    slide_window_models = []
    while (checking_date <= end_date):

        gbcf_1,gbcf_2 = calculate_slide_window(raw_data_df, slide_window_size, start_date, checking_date)
        
        slide_window_models.append((gbcf_1,gbcf_2))

        start_date = start_date + datetime.timedelta(days=1)
        checking_date = start_date + datetime.timedelta(days = slide_window_size)
    
    # forecasting...    
    forecasting_date = end_date + datetime.timedelta(days=1)
    forecasting_date_str = convertDatatimeToStr(forecasting_date)
    print("%s forecasting for %s , slide windows %d, data_fcsted_item_only %d"% 
          (getCurrentTime(), forecasting_date_str, len(slide_window_models), data_fcsted_item_only))

    fcsting_window_df = create_slide_window_df(raw_data_df, start_date, forecasting_date, slide_window_size, fcsted_item_df)
#                                                fcsted_item_df if (data_fcsted_item_only) else None)

    fcsting_matrix_df, training_UI = extracting_features(fcsting_window_df, slide_window_size, fcsted_item_df)
#                                                          fcsted_item_df if (data_fcsted_item_only) else None)
    Y_training_label = extracting_Y(training_UI, raw_data_df[raw_data_df['time'] == forecasting_date_str][['user_id', 'item_id', 'behavior_type']])
    fcsting_matrix_df = pd.concat([fcsting_matrix_df, Y_training_label['buy']], axis=1)
    
    features_names_for_model = get_feature_name_for_model(fcsting_matrix_df.columns)    
    for (gbcf_1,gbcf_2) in slide_window_models:
        Y_gbdt1_predicted = pd.DataFrame(gbcf_1.predict_proba(fcsting_matrix_df[features_names_for_model]), columns=['not buy', 'buy'])    
    
        fcsted_index_1 = Y_gbdt1_predicted[Y_gbdt1_predicted['buy'] >= g_min_prob].index

        Y_gbdt2_predicted = pd.DataFrame(gbcf_2.predict_proba(fcsting_matrix_df[features_names_for_model].ix[fcsted_index_1]), columns=['not buy', 'buy'])

        fcsted_index_2 = Y_gbdt2_predicted[Y_gbdt2_predicted['buy'] >= g_min_prob].index

        if ("predicted_cnt" not in fcsting_matrix_df.columns):
            fcsting_matrix_df['predicted_cnt'] = 0

        fcsting_matrix_df.loc[fcsted_index_1[fcsted_index_2], 'predicted_cnt'] += 1

    Y_fcsted_UI = fcsting_matrix_df[fcsting_matrix_df['predicted_cnt'] >= len(slide_window_models)/2][['user_id', 'item_id']]   
    
    if (forecasting_date_str == '2014-12-19'):
        index = 0
        Y_fcsted_UI = Y_fcsted_UI[(np.in1d(Y_fcsted_UI['item_id'], fcsted_item_df['item_id']))]
        prob_output_filename, submit_output_filename = get_output_filename(index, 0)
        while (os.path.exists(submit_output_filename)):
            index += 1
            prob_output_filename, submit_output_filename = get_output_filename(index, 0)

        Y_fcsted_UI.to_csv(submit_output_filename, index=False)
    else:
        Y_true_UI = raw_data_df[(raw_data_df['time'] == forecasting_date_str)&\
                                (raw_data_df['behavior_type'] == 4) &
                                (np.in1d(raw_data_df['item_id'], fcsted_item_df['item_id']))][['user_id', 'item_id']].drop_duplicates()

        p, r, f1 = calculate_POS_F1(Y_true_UI, Y_fcsted_UI)
        
        print("%s precision: %.4f, recall %.4f, F1 %.4f" % (getCurrentTime(), p, r, f1))

    return 0

if __name__ == '__main__':
#     single_window()
    slide_window()
    