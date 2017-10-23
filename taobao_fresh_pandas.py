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
from greenlet import getcurrent

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold 
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

# 规则： 如果user 在  checking date 前一天 cart, 并且没有购买 ，则认为他checking date 会购买
def rule_fav_cart_before_1day(forecasting_window_df, Y_forecasted):
    Y_forecasted = feature_user_item_opt_before1day(forecasting_window_df, 3, Y_forecasted)
    Y_forecasted = feature_user_item_opt_before1day(forecasting_window_df, 4, Y_forecasted)
    Y_forecasted['buy'][(Y_forecasted['item_cart_opt_before1day'] == 1)&(Y_forecasted['item_buy_opt_before1day'] == 0)] = 1

    return Y_forecasted

def calculate_slide_window(raw_data_df, slide_window_size, window_start_date, checking_date, fcsted_item_df):    
    checking_date_str = convertDatatimeToStr(checking_date)
    print("%s handling slide window %s" % (getCurrentTime(), checking_date_str))
    
    model_filename = r"%s\..\featuremat_and_model\model_%s_%d.m" % (runningPath, checking_date_str, slide_window_size)
    if (os.path.exists(model_filename)):
        print("%s loading model from model_%s_%d.m" % (getCurrentTime(), checking_date_str, slide_window_size))
        gbcf = joblib.load(model_filename)
        return gbcf
    
    slide_window_df = create_slide_window_df(raw_data_df, window_start_date, checking_date, slide_window_size, fcsted_item_df)
    slide_UI = slide_window_df[['user_id', 'item_id']].drop_duplicates()
 
    Y_label = extracting_Y(slide_UI, raw_data_df[raw_data_df['time'] == checking_date_str][['user_id', 'item_id', 'behavior_type']])
    
    feature_mat_filename = r"%s\..\featuremat_and_model\feature_mat_%s_%d.csv" % (runningPath, checking_date_str, slide_window_size)
    if (os.path.exists(feature_mat_filename)):
        print("%s reading feature matrix from: %s_%d.csv" % (getCurrentTime(), checking_date_str, slide_window_size))
        feature_matrix_df = pd.read_csv(feature_mat_filename)
    else:
        feature_matrix_df, UIC = extracting_features(slide_window_df, slide_window_size)

    gbcf = GradientBoostingClassifier(loss='deviance', 
                                      learning_rate=0.1, 
                                      n_estimators=80, # gride searched to 80
                                      subsample=1.0, 
                                      criterion='friedman_mse', 
                                      min_samples_split=100, 
                                      min_samples_leaf=1, 
                                      min_weight_fraction_leaf=0.0, 
                                      max_depth=3, # gride searched to 3
                                      max_leaf_nodes=None,
                                      min_impurity_split=None, init=None, 
                                      random_state=None, max_features=None,
                                      verbose=0, warm_start=False, presort='auto')
    if (0):
        gbcf_gs_param = {'min_samples_split':list(range(100,801,200)), 'min_samples_leaf':[5,10,15]}
        gbcf_gs = GridSearchCV(gbcf, gbcf_gs_param, n_jobs=-1, cv=5, refit=True)
        gbcf_gs.fit(feature_matrix_df, Y_label['buy'])
        Y = gbcf_gs.predict(feature_matrix_df)
        print("%s best score %.4f, best params %s" % (getCurrentTime(), gbcf_gs.best_score_, gbcf_gs.best_params_))
        gbcf = gbcf_gs.best_estimator_ 
    else:
        gbcf.fit(feature_matrix_df, Y_label['buy'])
        Y = gbcf.predict(feature_matrix_df)

    print(classification_report(Y_label['buy'], Y, target_names=["not buy", "buy"]))
    
    if (not os.path.exists(feature_mat_filename)):
        pd.DataFrame(feature_matrix_df).to_csv(feature_mat_filename, index=False)
        
    if (not os.path.exists(model_filename)):
        joblib.dump(gbcf, model_filename)

    del slide_window_df

    return gbcf

def create_slide_window_df(raw_data_df, window_start_date, window_end_date, slide_window_size, fcsted_item_df):
    if (fcsted_item_df is not None):
        slide_window_df = raw_data_df[(raw_data_df['time'] >= convertDatatimeToStr(window_start_date))&
                                      (raw_data_df['time'] < convertDatatimeToStr(window_end_date)) &
                                      (np.in1d(raw_data_df['item_id'], fcsted_item_df['item_id']))]
    else:
        slide_window_df = raw_data_df[(raw_data_df['time'] >= convertDatatimeToStr(window_start_date))&
                                      (raw_data_df['time'] < convertDatatimeToStr(window_end_date))]

    slide_window_df = remove_user_item_only_buy(slide_window_df)
    slide_window_df.index = range(slide_window_df.shape[0])  # 重要！！

    slide_window_df['dayoffset'] = convert_date_str_to_dayoffset(slide_window_df['time'], slide_window_size, window_end_date)

    return slide_window_df


def single_window():

    slide_window_size = 7
    data_filename = r"%s\..\input\preprocessed_user_data.csv" % (runningPath)
    print(getCurrentTime(), "reading csv ", data_filename)
    raw_data_df = pd.read_csv(data_filename)

    fcsted_item_filename = r"%s\..\input\preprocessed_user_data_fcsted_item_only.csv" % (runningPath)
    print(getCurrentTime(), "reading being forecasted items ", fcsted_item_filename)
    fcsted_item_df = pd.read_csv(fcsted_item_filename)
    
    training_date = datetime.datetime.strptime('2014-12-18', "%Y-%m-%d")
    window_start_date = training_date - datetime.timedelta(days=slide_window_size)
    
    forecasting_date = training_date + datetime.timedelta(days=1)
    forecasting_date_str = convertDatatimeToStr(forecasting_date)

    # training...
    gbcf = calculate_slide_window(raw_data_df, slide_window_size, window_start_date, training_date, fcsted_item_df)
    
    print("GradientBoostingClassifier params: ", gbcf.get_params())
    
    # forecasting...
    window_start_date = forecasting_date - datetime.timedelta(days=slide_window_size)
    
    forecasting_window_df = create_slide_window_df(raw_data_df, window_start_date, forecasting_date, slide_window_size, fcsted_item_df)

    forecasting_feature_matrix_df, forecasting_UI = extracting_features(forecasting_window_df, slide_window_size)

    Y_fcsted = gbcf.predict_proba(forecasting_feature_matrix_df)
    
    print('Y_fcsted.shape', Y_fcsted.shape)
    print('forecasting_UI.shape', forecasting_UI.shape)

    Y_fcsted_lable = pd.DataFrame(Y_fcsted, columns=['not_buy', 'buy'])
    Y_fcsted_lable = pd.concat([forecasting_UI, Y_fcsted_lable], axis=1)

    # 规则： 如果user 在  checking date 前一天 cart, 并且没有购买 ，则认为他checking date 会购买
    Y_fcsted_lable = rule_fav_cart_before_1day(forecasting_window_df, Y_fcsted_lable)

    if (forecasting_date_str == '2014-12-19'):
        UI_buy_allinfo = Y_fcsted_lable[Y_fcsted_lable['buy'] >= 0.5]
        UI_buy_allinfo.to_csv(r"%s\..\output\single_window_forecast_with_prob_%s.csv" % (runningPath, datetime.date.today()), index=False)

        UI_buy = UI_buy_allinfo[['user_id', 'item_id']]
        UI_buy.to_csv(r"%s\..\output\single_window_forecast_submit_%s.csv" % (runningPath, datetime.date.today()), index=False)
    else:
        Y_true_label = extracting_Y(forecasting_UI, raw_data_df[raw_data_df['time'] == forecasting_date_str][['user_id', 'item_id', 'behavior_type']])
        Y_fcsted_lable[Y_fcsted_lable['buy'] > 0.5] = 1
        Y_fcsted_lable[Y_fcsted_lable['buy'] <= 0.5] = 0

#         f1 = f1_score(Y_true_label['buy'], Y_fcsted_lable['buy'], average='macro')
#         print("F1 for %s, %.4f" % (forecasting_date_str, f1))
        print(classification_report(Y_true_label['buy'], Y_fcsted_lable['buy'], target_names=["not buy", "buy"]))

    return 0


def convert_feature_mat_to_leaf_node(feature_matrix_df, slide_window_models):
    X_leafnode_mat = None
    for gbcf_mod in slide_window_models:
        X_mat_enc = gbcf_mod.apply(feature_matrix_df)[:, :, 0]

        if (X_leafnode_mat is None):
            X_leafnode_mat = X_mat_enc
        else:
            X_leafnode_mat = np.column_stack((X_leafnode_mat, X_mat_enc))
    
    return X_leafnode_mat

def slide_window():
    data_filename = r"%s\..\input\preprocessed_user_data_no_hour.csv" % (runningPath)
    print(getCurrentTime(), "reading csv ", data_filename)
    raw_data_df = pd.read_csv(data_filename)
    
    fcsted_item_filename = r"%s\..\input\tianchi_fresh_comp_train_item.csv" % (runningPath)
    print(getCurrentTime(), "reading being forecasted items ", fcsted_item_filename)
    fcsted_item_df = pd.read_csv(fcsted_item_filename)

    slide_window_size = 4
    start_date = datetime.datetime.strptime('2014-12-10', "%Y-%m-%d")
    end_date = datetime.datetime.strptime('2014-12-16', "%Y-%m-%d")

    print("%s slide window size %d, start date %s, end date %s" % 
          (getCurrentTime(), slide_window_size, convertDatatimeToStr(start_date), convertDatatimeToStr(end_date)))

    checking_date = start_date + datetime.timedelta(days = slide_window_size)
    slide_window_models = []
    while (checking_date < end_date):

        gbcf = calculate_slide_window(raw_data_df, slide_window_size, start_date, checking_date, fcsted_item_df)
        
        slide_window_models.append(gbcf)

        start_date = start_date + datetime.timedelta(days=1)
        checking_date = start_date + datetime.timedelta(days = slide_window_size)

    checking_date_str = convertDatatimeToStr(checking_date)
    training_window_df = create_slide_window_df(raw_data_df, start_date, checking_date, slide_window_size, fcsted_item_df)

    training_matrix_df, training_UI = extracting_features(training_window_df, slide_window_size)
    Y_training_label = extracting_Y(training_UI, raw_data_df[raw_data_df['time'] == checking_date_str][['user_id', 'item_id', 'behavior_type']])

    # 滑动窗口训练出的model分别对[ 12-18 - slide_window-size, 12-18] 的数据生成叶节点, 生成一个大的特征矩阵，然后交给LR进行训练
    X_leafnode_mat = convert_feature_mat_to_leaf_node(training_matrix_df, slide_window_models)

    logisticReg = LogisticRegression()
    
    logisticReg.fit(X_leafnode_mat, Y_training_label['buy'])

    fcsting_window_start_date = start_date + datetime.timedelta(days=1)
    fcsting_date = fcsting_window_start_date + datetime.timedelta(days = slide_window_size)
    print("forecasting %s, window start date %s, window size %d" %(checking_date, start_date, slide_window_size))

    fcsting_window_df = create_slide_window_df(raw_data_df, fcsting_window_start_date, fcsting_date, slide_window_size, fcsted_item_df)
    fcsting_matrix_df, fcsting_UI = extracting_features(fcsting_window_df, slide_window_size)
    
    fcsting_matrix_df = convert_feature_mat_to_leaf_node(fcsting_matrix_df, slide_window_models)

    Y_pred = logisticReg.predict(fcsting_matrix_df)
    Y_pred = pd.DataFrame(Y_pred, columns=['buy'])

    Y_pred = pd.concat([fcsting_UI, Y_pred], axis=1)

    if (fcsting_date == ONLINE_FORECAST_DATE):
        UI_buy_allinfo = Y_pred[Y_pred['buy'] >= 0.5]
        UI_buy_allinfo.to_csv(r"%s\..\output\slide_window_forecast_with_prob_%s.csv" % (runningPath, datetime.date.today()), index=False)
        
        UI_buy = UI_buy_allinfo[['user_id', 'item_id']]
        UI_buy.to_csv(r"%s\..\output\slide_window_forecast_submit_%s.csv" % (runningPath, datetime.date.today()), index=False)
    else:
        Y_true_label = extracting_Y(training_UI, raw_data_df[raw_data_df['time'] == checking_date_str][['user_id', 'item_id', 'behavior_type']])
        print(classification_report(Y_true_label['buy'], Y_pred['buy'], target_names=["not buy", "buy"]))
    return 0


def run_in_ipython():
    df = pd.read_csv(r'F:\doc\ML\taobao\fresh_comp_offline\taobao_fresh_pandas\input\preprocessed_user_data_fcsted_item_only.csv')
    slide_window_df = df[(df['time'] < '2014-12-18')&(df['time'] >= '2014-12-11')]
    start_date = datetime.datetime.strptime('2014-12-18', "%Y-%m-%d")
    
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
#     slide_window()
    