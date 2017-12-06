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

from sklearn.externals import joblib

runningSubProcesses = {}

def submiteOneSubProcess(start_date_str, end_date_str, slide_window_size):
    cmdLine = "python taobao_fresh_pandas_parallel.py start_date=%s end_date=%s size=%d" % (start_date_str, end_date_str, slide_window_size)
    sub = subprocess.Popen(cmdLine, shell=True)
    runningSubProcesses[(start_date_str, end_date_str, time.time())] = sub
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
    forecasting_date_str = sys.argv[1].split("=")[1]
    window_size_list = sys.argv[2].split("=")[1]
    window_size_list = window_size_list.split(",")
    
    start_time = time.time()

    forecasting_date = datetime.datetime.strptime(forecasting_date_str, "%Y-%m-%d")

    for window_size in window_size_list:
        window_size = int(window_size)
        window_start_date = forecasting_date - datetime.timedelta(days=window_size + 1)
        window_start_date_str = convertDatatimeToStr(window_start_date)
  
        submiteOneSubProcess(window_start_date_str, forecasting_date_str, window_size)
      
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

    
    slide_window_wieght = get_slide_window_wieght(window_size_list, forecasting_date)

    ui_fcsting_cnt = {}
    slide_windows = 0
    for window_size in window_size_list:
        window_size = int(window_size)
        window_start_date = forecasting_date - datetime.timedelta(days=window_size + 1)
        window_start_date_str = convertDatatimeToStr(window_start_date)

        subprocess_filename = r"%s\..\output\subprocess\%s_%d_%s.csv" % (runningPath, window_start_date_str, window_size, forecasting_date_str)
        print(getCurrentTime(), "reading sub process %s" % subprocess_filename)

        index = 0
        subprocess_fcsting = csv.reader(open(subprocess_filename, encoding="utf-8", mode='r'))
        for aline in subprocess_fcsting:
            if (index == 0):
                index += 1
                continue

            user_id = aline[0]
            item_id = aline[1]
            proba = float(aline[2])

            ui_tuple = (user_id, item_id)
            if (ui_tuple in ui_fcsting_cnt):
                ui_fcsting_cnt[ui_tuple] += (proba * slide_window_wieght[(window_start_date_str, window_size)])
            else:
                ui_fcsting_cnt[ui_tuple] = (proba * slide_window_wieght[(window_start_date_str, window_size)])

        slide_windows += 1

    # ensemble forecasting...       
    Y_fcsted_UI = []
    for ui_tuple in ui_fcsting_cnt:
        if (ui_fcsting_cnt[ui_tuple] >= 0.5):
            Y_fcsted_UI.append([ui_tuple[0], ui_tuple[1]])

    Y_fcsted_UI = pd.DataFrame(Y_fcsted_UI, columns=['user_id', 'item_id'])
    
    fcsted_item_filename = r"%s\..\input\tianchi_fresh_comp_train_item.csv" % (runningPath)
    print(getCurrentTime(), "reading being forecasted items ", fcsted_item_filename)
    fcsted_item_df = pd.read_csv(fcsted_item_filename, dtype={'item_id':np.str})

    use_rule = 1
    if (use_rule):
        slide_window_df = create_slide_window_df(fcsted_item_df, window_start_date, forecasting_date, 10, None)
        # 规则： 如果user 在  checking date 前一天 cart, 并且没有购买 ，则认为他checking date 会购买
        Y_fcsted_UI = rule_fav_cart_before_1day(slide_window_df, Y_fcsted_UI)

    if (forecasting_date_str == '2014-12-19'):
        index = 0
        Y_fcsted_UI = Y_fcsted_UI[(np.in1d(Y_fcsted_UI['item_id'], fcsted_item_df['item_id']))]
        prob_output_filename, submit_output_filename = get_output_filename(index, "half_slide", 0)
        while (os.path.exists(submit_output_filename)):
            index += 1
            prob_output_filename, submit_output_filename = get_output_filename(index, "half_slide", 0)

        Y_fcsted_UI.to_csv(submit_output_filename, index=False)
        
        param_filename = submit_output_filename + ".param.txt"

        param_f = open(param_filename, encoding="utf-8", mode='w')
        param_f.write("pos:nage=1:%d, window size list=%s, min prob %.4f" % (g_nag_times, window_size_list, g_min_prob))
        param_f.close()
    else:
        data_filename = r"%s\..\input\preprocessed_user_data_sold_item_only.csv" % (runningPath)

        print(getCurrentTime(), "reading csv ", data_filename)
        raw_data_df = pd.read_csv(data_filename, dtype={'user_id':np.str, 'item_id':np.str})

        Y_true_UI = raw_data_df[(raw_data_df['time'] == forecasting_date_str)&\
                                (raw_data_df['behavior_type'] == 4) &
                                (np.in1d(raw_data_df['item_id'], fcsted_item_df['item_id']))][['user_id', 'item_id']].drop_duplicates()

        p, r, f1 = calculate_POS_F1(Y_true_UI, Y_fcsted_UI)
        Y_fcsted_UI.to_csv(r"%s\..\output\fcst_%s.csv" % (runningPath, forecasting_date_str))

        print("%s precision: %.4f, recall %.4f, F1 %.4f" % (getCurrentTime(), p, r, f1))
        
    end_time = time.time()
    
    print(getCurrentTime(), " done, ran %d seconds" % (end_time - start_time))
    
    return 0


def get_slide_window_wieght(window_size_list, forecasting_date):
    forecasting_date_str = convertDatatimeToStr(forecasting_date)
    weight_dict = dict()
    total = 0
    for window_size in window_size_list:
        window_size = int(window_size)
        window_start_date = forecasting_date - datetime.timedelta(days=window_size + 1)
        window_start_date_str = convertDatatimeToStr(window_start_date)

        slidewindow_filename = r"%s\..\output\subprocess\%s_%d_%s_p_r_f1.csv" % (runningPath, window_start_date_str, window_size, forecasting_date_str)
        print("reading sub process", slidewindow_filename)

        index = 0
        slidewindow_fcsting = csv.reader(open(slidewindow_filename, encoding="utf-8", mode='r'))
        for aline in slidewindow_fcsting:
            p = float(aline[0])
            r = float(aline[1])
            f1 = float(aline[2])
            
        print("f1 ", f1)
        
        weight_dict[(window_start_date_str, window_size)] = f1
        
        total += f1
        
        window_start_date = window_start_date + datetime.timedelta(days=1)
        checking_date = window_start_date + datetime.timedelta(days = window_size)
        
    for k, f1 in weight_dict.items():
        weight_dict[k] = f1 / total

    print("slide window weight ", weight_dict)
    return weight_dict

if __name__ == '__main__':
    main()
