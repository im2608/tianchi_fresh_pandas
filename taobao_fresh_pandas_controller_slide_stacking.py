'''
Created on Nov 9, 2017

@author: Heng.Zhang
'''

import subprocess  
import os
import time
import sys
import datetime
from global_variables import *
from common import *
from feature_extraction import *
import csv
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from sklearn.externals import joblib

runningSubProcesses = {}

def submiteOneSubProcess(start_date_str, slide_window_size):
    cmdLine = "python taobao_fresh_pandas_parallel_slide_stacking.py start_date=%s size=%d" % (start_date_str, slide_window_size)
    sub = subprocess.Popen(cmdLine, shell=True)
    runningSubProcesses[(start_date_str, slide_window_size, time.time())] = sub
    print("running cmd line: %s" % cmdLine)
    time.sleep(1)
    return


def waitSubprocesses():
    for start_end_date_str in runningSubProcesses:
        sub = runningSubProcesses[start_end_date_str]
        ret = subprocess.Popen.poll(sub)
        if ret == 0:
            runningSubProcesses.pop(start_end_date_str)
            return start_end_date_str
        elif ret is None:
            time.sleep(1) # running
        else:
            runningSubProcesses.pop(start_end_date_str)
            return start_end_date_str
    return (0, 0)

def main():
    start_date_str = sys.argv[1].split("=")[1]
    end_date_str = sys.argv[2].split("=")[1]
    forecasting_date_str = sys.argv[3].split("=")[1]
    window_size = int(sys.argv[4].split("=")[1])
    
    start_time = time.time()

    window_start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    checking_date = window_start_date + datetime.timedelta(days=window_size)

    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")

    forecasting_date = datetime.datetime.strptime(forecasting_date_str, "%Y-%m-%d")

    while (checking_date <= end_date):
        # 删除了12-12的数据， 不再计算12-12， 12-13的滑窗
        if (checking_date.month == 12 and (checking_date.day in [12, 13])):
            checking_date = datetime.datetime(2014,12,14,0,0,0)
            window_start_date = checking_date - datetime.timedelta(days=window_size)
 
        window_start_date_str = convertDatatimeToStr(window_start_date)
  
        submiteOneSubProcess(window_start_date_str, window_size)
      
        window_start_date = window_start_date + datetime.timedelta(days=1)
        checking_date = window_start_date + datetime.timedelta(days = window_size)
        if (len(runningSubProcesses) == 10):
            while True:
                start_end_date_str = waitSubprocesses()
                if ((start_end_date_str[0] != 0 and start_end_date_str[1] != 0)):
                    print("after waitSubprocesses, subprocess [%s, %s] finished, took %d seconds, runningSubProcesses len is %d" % 
                          (start_end_date_str[0], start_end_date_str[1], time.time() - start_end_date_str[2], len(runningSubProcesses)))
                    break
                if (len(runningSubProcesses) == 0):
                    break
    while True:
        start_end_date_str = waitSubprocesses()
        if ((start_end_date_str[0] != 0 and start_end_date_str[1] != 0)):
            print("after waitSubprocesses, subprocess [%s, %s] finished, took %d seconds, runningSubProcesses len is %d" % 
                  (start_end_date_str[0], start_end_date_str[1], time.time() - start_end_date_str[2], len(runningSubProcesses)))
        if (len(runningSubProcesses) == 0):
            break

    window_start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    checking_date = window_start_date + datetime.timedelta(days=window_size)

    data_filename = r"%s\..\input\preprocessed_user_data_fcsted_item_only.csv" % (runningPath)
#     data_filename = r"%s\..\input\preprocessed_user_data_sold_item_only_no1212.csv" % (runningPath)

    print(getCurrentTime(), "reading csv ", data_filename)
    raw_data_df = pd.read_csv(data_filename, dtype={'user_id':np.str, 'item_id':np.str})

    fcsted_item_filename = r"%s\..\input\tianchi_fresh_comp_train_item.csv" % (runningPath)
    print(getCurrentTime(), "reading being forecasted items ", fcsted_item_filename)
    fcsted_item_df = pd.read_csv(fcsted_item_filename, dtype={'item_id':np.str})

    window_start_date = forecasting_date - datetime.timedelta(days=window_size)
    fcsting_window_df = create_slide_window_df(raw_data_df, window_start_date, forecasting_date, window_size, fcsted_item_df)
    
    print("fcasting matrix shape ", fcsting_window_df.shape)
    fcsting_matrix_df, fcsting_UI = extracting_features(fcsting_window_df, window_size, fcsted_item_df)

    ui_fcsting_cnt = {}
    slide_windows = 0
    X = []
    
    window_start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    checking_date = window_start_date + datetime.timedelta(days=window_size)

    while (checking_date <= end_date):
        # 删除了12-12的数据， 不再计算12-12， 12-13的滑窗
        if (checking_date.month == 12 and (checking_date.day in [12, 13])):
            checking_date = datetime.datetime(2014,12,14,0,0,0)
            window_start_date = checking_date - datetime.timedelta(days=window_size)

        window_start_date_str = convertDatatimeToStr(window_start_date)

        feature_mat_filename = r"%s\..\featuremat_and_model\feature_mat_%s_%d.csv" % (runningPath, window_start_date_str, window_size)
        print("reading feature matrix ", feature_mat_filename)
        slide_feature_mat = pd.read_csv(feature_mat_filename)
        X.append(slide_feature_mat)

        window_start_date = window_start_date + datetime.timedelta(days=1)
        checking_date = window_start_date + datetime.timedelta(days = window_size)
        slide_windows += 1

    train_feature_mat = pd.concat(X, axis=0, ignore_index=True)
    print(getCurrentTime(), " stacked training matrix shape %d, %d/%d" % (train_feature_mat.shape[0], \
        train_feature_mat[train_feature_mat['buy'] == 1].shape[0], train_feature_mat[train_feature_mat['buy'] == 0].shape[0]))
    features_names_for_model = get_feature_name_for_model(train_feature_mat.columns)

    # ensemble forecasting...
    gbcf_1, gbcf_2 = trainingModel(train_feature_mat, forecasting_date)

    Y_gbdt1_predicted = pd.DataFrame(gbcf_1.predict_proba(fcsting_matrix_df[features_names_for_model]), columns=['not buy', 'buy'])    

    fcsted_index_1 = Y_gbdt1_predicted[Y_gbdt1_predicted['buy'] >= g_min_prob].index
    print(getCurrentTime(), " gbdt 1 forecasted %d", fcsted_index_1.shape[0])

    Y_gbdt2_predicted = pd.DataFrame(gbcf_2.predict_proba(fcsting_matrix_df[features_names_for_model].iloc[fcsted_index_1]), columns=['not buy', 'buy'])

    fcsted_index_2 = Y_gbdt2_predicted[Y_gbdt2_predicted['buy'] >= g_min_prob].index
    
    fcsted_ui = fcsting_matrix_df.iloc[fcsted_index_1[fcsted_index_2]][['user_id', 'item_id']]

    forecasting_date = end_date + datetime.timedelta(days=1)
    forecasting_date_str = convertDatatimeToStr(forecasting_date)
    print("%s forecasting for %s, slide window %s, forecasted count %d" % (getCurrentTime(), forecasting_date_str, slide_windows, Y_gbdt1_predicted.shape[0]))
    
    fcsted_ui_no_rule = fcsted_ui.copy()

    # 规则： 如果user 在  checking date 前一天 cart, 并且没有购买 ，则认为他checking date 会购买
    fcsted_ui_with_rule = rule_fav_cart_before_1day(fcsting_window_df, fcsted_ui)
    fcsted_ui_with_rule.fillna(1, inplace=True)

    if (forecasting_date_str == '2014-12-19'):
        index = 0
        use_rule = 0
        
        prob_output_filename, submit_output_filename = get_output_filename(index, "stacking", use_rule)
        while (os.path.exists(submit_output_filename)):
            index += 1
            prob_output_filename, submit_output_filename = get_output_filename(index, "stacking", use_rule)

        if (use_rule):
            fcsted_ui_with_rule.to_csv(submit_output_filename, index=False)
        else:
            fcsted_ui_no_rule.to_csv(submit_output_filename, index=False)

        param_filename = submit_output_filename + ".param.txt"

        param_f = open(param_filename, encoding="utf-8", mode='w')
        param_f.write("pos:nage=1:%d, window size=%d, start=%s, min prob %.4f" %
                      (g_nag_times, window_size, start_date_str, g_min_prob))
        param_f.close()
    else:
        data_filename = r"%s\..\input\preprocessed_user_data_sold_item_only_no1212.csv" % (runningPath)

        print(getCurrentTime(), "reading csv ", data_filename)
        raw_data_df = pd.read_csv(data_filename, dtype={'user_id':np.str, 'item_id':np.str})

        Y_true_UI = raw_data_df[(raw_data_df['time'] == forecasting_date_str)&\
                                (raw_data_df['behavior_type'] == 4) &
                                (np.in1d(raw_data_df['item_id'], fcsted_item_df['item_id']))][['user_id', 'item_id']].drop_duplicates()

        p, r, f1 = calculate_POS_F1(Y_true_UI, fcsted_ui_with_rule)
        print("%s WITH rule: precision: %.4f, recall %.4f, F1 %.4f" % (getCurrentTime(), p, r, f1))

        p, r, f1 = calculate_POS_F1(Y_true_UI, fcsted_ui_no_rule)
        print("%s WITHOUT rule: precision: %.4f, recall %.4f, F1 %.4f" % (getCurrentTime(), p, r, f1))

    end_time = time.time()
    
    print(getCurrentTime(), " done, ran %d seconds" % (end_time - start_time))

    return 0

if __name__ == '__main__':
    main()
