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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# 规则： 如果user 在  checking date 前一天 cart, 并且没有购买 ，则认为他checking date 会购买
def rule_fav_cart_before_1day(forecasting_window_df, Y_forecasted):
    Y_forecasted = feature_user_item_opt_before1day(forecasting_window_df, 3, Y_forecasted)
    Y_forecasted = feature_user_item_opt_before1day(forecasting_window_df, 4, Y_forecasted)
    Y_forecasted.loc[(Y_forecasted['item_cart_opt_before1day'] == 1)&(Y_forecasted['item_buy_opt_before1day'] == 0), 'buy_prob'] = 1

    return Y_forecasted

def takeSamplesForTraining(feature_matrix_df):
    pos = feature_matrix_df[feature_matrix_df['buy'] == 1]
    nag = feature_matrix_df[feature_matrix_df['buy'] == 0].sample(n = pos.shape[0] * g_nag_times, axis=0)

    # 正负样本的比例 1:g_nag_times
    samples = pd.concat([pos, nag], axis=0)
    
    # 采样50%用于训练LR
    samples_for_LR = samples.sample(frac=0.5, axis=0)
    
    # 取另外的50%用LR来预测， 预测后的结果作为 Y_for_gbdt 用于训练gbdt
    index_for_gbdt = samples.index.difference(samples_for_LR.index)
    samples_for_gbdt = samples.ix[index_for_gbdt]
    
    print("%s samples_for_LR has %d POS, samples_for_gbdt has %d POS" % (getCurrentTime(), 
                                                                         samples_for_LR[samples_for_LR['buy']==1].shape[0], 
                                                                         samples_for_gbdt[samples_for_gbdt['buy']==1].shape[0]))
    return samples_for_LR, samples_for_gbdt

def trainingModel(feature_matrix_df):
    lr_params = {
        'penalty' : 'l2',
        'fit_intercept' : True,
        'intercept_scaling' : 1,
        'class_weight' : 'balanced',
        'solver':'liblinear',
        'max_iter' : 100,
        'multi_class' : 'ovr',
        'n_jobs' : -1
    }

#     lr_gs_param = {'class_weight':[None, 'balanced'], 'solver' : ['liblinear', 'newton-cg'],}
#     lr_gs = GridSearchCV(logiReg, lr_gs_param, n_jobs=-1, cv=5, refit=True)    
#     lr_gs.fit(samples_for_LR[features_for_model], samples_for_LR['buy'])
#     print("%s LogisticRegression() best score %.4f, best params %s" % (getCurrentTime(), lr_gs.best_score_, lr_gs.best_params_))
#     logiReg = lr_gs.best_estimator_
    features_names_for_model = get_feature_name_for_model(feature_matrix_df.columns)
    min_max_scaler = preprocessing.MinMaxScaler()

    run_times = 10
    i = 0
    logiReg = LogisticRegression(**lr_params)    
    while (i < run_times):
        samples_for_LR, samples_for_gbdt = takeSamplesForTraining(feature_matrix_df)
        features_for_model = samples_for_LR[features_names_for_model]
#         features_for_model = min_max_scaler.fit_transform(samples_for_LR[features_names_for_model])
        
        logiReg.fit(features_for_model, samples_for_LR['buy'])
        Y_pred_gbdt = logiReg.predict(samples_for_gbdt[features_names_for_model])   

        if (Y_pred_gbdt.sum() == 0):
            print("%s LogisticRegression predicted 0 POS, trying to resample" % getCurrentTime())
            i += 1
        else:
            print("%s LogisticRegression() fit the training date: " % getCurrentTime())
            print(classification_report(samples_for_gbdt['buy'], Y_pred_gbdt, target_names=["not buy", "buy"]))
            print("%s confusion matrix: " % getCurrentTime())
            print(confusion_matrix(samples_for_gbdt['buy'], Y_pred_gbdt))
            break

    if (Y_pred_gbdt.sum() == 0):
        print("%s, after %d times fitting, LogisticRegression() still predicted 0 POS, exiting" % (getCurrentTime(), run_times))
        exit(0)

    gbcf = GradientBoostingClassifier(n_estimators=80, # gride searched to 80
                                      subsample=1.0, 
                                      criterion='friedman_mse', 
                                      min_samples_split=100, 
                                      min_samples_leaf=1,
                                      max_depth=3) # gride searched to 3

    features_for_model = samples_for_gbdt[features_names_for_model]
#     features_for_model = min_max_scaler.fit_transform(samples_for_gbdt[features_names_for_model])

    gbcf.fit(features_for_model, Y_pred_gbdt)

    return logiReg, gbcf

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

    logiReg, gbcf = trainingModel(feature_matrix_df)

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

    return logiReg, gbcf

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

    slide_window_size = 30
    data_filename = r"%s\..\input\preprocessed_user_data.csv" % (runningPath)
#     data_filename = r"%s\..\input\preprocessed_user_data_fcsted_item_only.csv" % (runningPath)
    print(getCurrentTime(), "reading csv ", data_filename)
    raw_data_df = pd.read_csv(data_filename)

    fcsted_item_filename = r"%s\..\input\tianchi_fresh_comp_train_item.csv" % (runningPath)
    print(getCurrentTime(), "reading being forecasted items ", fcsted_item_filename)
    fcsted_item_df = pd.read_csv(fcsted_item_filename)

    training_date = datetime.datetime.strptime('2014-12-18', "%Y-%m-%d")
    window_start_date = training_date - datetime.timedelta(days=slide_window_size)

    # training...
    logiReg, gbcf = calculate_slide_window(raw_data_df, slide_window_size, window_start_date, training_date)
    print("LogisticRegression() params: ", logiReg.get_params())
    print("GradientBoostingClassifier() params: ", gbcf.get_params())
    
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
    
    # 使用LR过滤数据, 只将预测为正的数据传给gbdt预测
    Y_fcsted_gbdt = pd.DataFrame(logiReg.predict_proba(features_for_model), columns=['not buy', 'buy'])    
    
    min_prob = 0.7
    fcsted_index = Y_fcsted_gbdt[Y_fcsted_gbdt['buy'] >= min_prob].index
    features_for_model = features_for_model.ix[fcsted_index]    
    features_for_model.index = range(features_for_model.shape[0])
    
    fcsted_UI = forecasting_UI.ix[fcsted_index]
    fcsted_UI.index = range(fcsted_UI.shape[0]) 

    Y_fcsted = gbcf.predict_proba(features_for_model)

    print('%s fcsted_UI.shape %s' %(getCurrentTime(), fcsted_UI.shape))

    Y_fcsted_lable = pd.DataFrame(Y_fcsted, columns=['not_buy_prob', 'buy_prob'])
    Y_fcsted_lable = pd.concat([fcsted_UI, Y_fcsted_lable], axis=1)
    
    use_rule = 0
    if (use_rule):
        # 规则： 如果user 在  checking date 前一天 cart, 并且没有购买 ，则认为他checking date 会购买
        Y_fcsted_lable = rule_fav_cart_before_1day(forecasting_window_df, Y_fcsted_lable)

    if (forecasting_date_str == '2014-12-19'):
        UI_buy_allinfo = Y_fcsted_lable[Y_fcsted_lable['buy_probs'] >= min_prob]
        index = 0
        prob_output_filename, submit_output_filename = get_output_filename(index, use_rule)
        while (os.path.exists(submit_output_filename)):
            index += 1
            prob_output_filename, submit_output_filename = get_output_filename(index, use_rule)

        UI_buy_allinfo.to_csv(prob_output_filename, index=False)
        UI_buy_allinfo[['user_id', 'item_id']].to_csv(submit_output_filename, index=False)
    else:
        Y_true_label = extracting_Y(fcsted_UI, raw_data_df[raw_data_df['time'] == forecasting_date_str][['user_id', 'item_id', 'behavior_type']])
        Y_fcsted_lable['buy'] = 0
        Y_fcsted_lable.loc[Y_fcsted_lable['buy_prob'] >= min_prob, 'buy'] = 1

#         f1 = f1_score(Y_true_label['buy'], Y_fcsted_lable['buy'], average='macro')
#         print("F1 for %s, %.4f" % (forecasting_date_str, f1))
        print("final report :", getCurrentTime())
        print(classification_report(Y_true_label['buy'], Y_fcsted_lable['buy'], target_names=["not buy", "buy"]))
        print("confusion matrix: ")
        print(confusion_matrix(Y_true_label['buy'], Y_fcsted_lable['buy']))

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
    