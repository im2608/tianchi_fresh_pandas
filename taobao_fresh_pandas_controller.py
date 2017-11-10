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

from sklearn.externals import joblib

runningSubProcesses = {}

def submiteOneSubProcess(start_date_str, end_date_str, slide_window_size):
    cmdLine = "python taobao_fresh_pandas_parallel.py start_date=%s end_date=%s size=%d" % (start_date_str, end_date_str, slide_window_size)
    sub = subprocess.Popen(cmdLine, shell=True)
    runningSubProcesses[(start_date_str, end_date_str)] = sub
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
    window_size = int(sys.argv[3].split("=")[1])
    
    window_start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    checking_date = window_start_date + datetime.timedelta(days=window_size)
 
    end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
     
    while (checking_date <= end_date):
        window_start_date_str = convertDatatimeToStr(window_start_date)
        checking_date_str = convertDatatimeToStr(checking_date)
         
        submiteOneSubProcess(window_start_date_str, checking_date_str, window_size)
 
        window_start_date = window_start_date + datetime.timedelta(days=1)
        checking_date = window_start_date + datetime.timedelta(days = window_size)
        if (len(runningSubProcesses) == 10):
            while True:
                start_end_date_str = waitSubprocesses()
                if ((start_end_date_str[0] != 0 and start_end_date_str[1] != 0)):
                    print("after waitSubprocesses, subprocess [%s, %s] finished, runningSubProcesses len is %d" % 
                          (start_end_date_str[0], start_end_date_str[1], len(runningSubProcesses)))
                    break
                if (len(runningSubProcesses) == 0):
                    break
 
    while True:
        start_end_date_str = waitSubprocesses()
        if ((start_end_date_str[0] != 0 and start_end_date_str[1] != 0)):
            print("after waitSubprocesses, subprocess [%s, %s] finished, runningSubProcesses len is %d" % 
                  (start_end_date_str[0], start_end_date_str[1], len(runningSubProcesses)))
        if (len(runningSubProcesses) == 0):
            break

    window_start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    checking_date = window_start_date + datetime.timedelta(days=window_size)
    slide_window_models = []

    while (checking_date <= end_date):
        window_start_date_str = convertDatatimeToStr(window_start_date)
        checking_date_str = convertDatatimeToStr(checking_date)

        model_1_filename = r"%s\..\featuremat_and_model\model_1_%s_%s_%d.m" % (runningPath, window_start_date_str, checking_date_str, window_size)
        model_2_filename = r"%s\..\featuremat_and_model\model_2_%s_%s_%d.m" % (runningPath, window_start_date_str, checking_date_str, window_size)

        print("loading model 1 ", model_1_filename)
        gbcf_1 = joblib.load(model_1_filename)

        print("loading model 2 ", model_2_filename)
        gbcf_2 = joblib.load(model_2_filename)

        slide_window_models.append((gbcf_1,gbcf_2))
        
        window_start_date = window_start_date + datetime.timedelta(days=1)
        checking_date = window_start_date + datetime.timedelta(days = window_size)

    # ensemble forecasting...    
    

    data_filename = r"%s\..\input\preprocessed_user_data_sold_item_only.csv" % (runningPath)

    print(getCurrentTime(), "reading csv ", data_filename)
    raw_data_df = pd.read_csv(data_filename, dtype={'user_id':np.str, 'item_id':np.str})
    
    fcsted_item_filename = r"%s\..\input\tianchi_fresh_comp_train_item.csv" % (runningPath)
    print(getCurrentTime(), "reading being forecasted items ", fcsted_item_filename)
    fcsted_item_df = pd.read_csv(fcsted_item_filename, dtype={'item_id':np.str})
    
    forecasting_date = end_date + datetime.timedelta(days=1)
    forecasting_date_str = convertDatatimeToStr(forecasting_date)
    print("%s forecasting for %s , slide windows %d" % (getCurrentTime(), forecasting_date_str, len(slide_window_models)))

    fcsting_window_df = create_slide_window_df(raw_data_df, window_start_date, forecasting_date, window_size, fcsted_item_df)
#                                                fcsted_item_df if (data_fcsted_item_only) else None)

    fcsting_matrix_df, fcsting_UI = extracting_features(fcsting_window_df, window_size, fcsted_item_df)
#                                                          fcsted_item_df if (data_fcsted_item_only) else None)
    Y_training_label = extracting_Y(fcsting_UI, raw_data_df[raw_data_df['time'] == forecasting_date_str][['user_id', 'item_id', 'behavior_type']])
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
    main()
