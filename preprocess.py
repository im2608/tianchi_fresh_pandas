'''
Created on Aug 4, 2017

@author: Heng.Zhang
'''

import csv
from common import *
from global_variables import *
import datetime


# 去掉geo， 
# 去掉没有购买过任何物品的user
# 只保留需要预测的item
def preprocess_data():
    data_filename = r"%s\..\input\tianchi_fresh_comp_train_user.csv" % (runningPath)
#     data_filename = r"%s\..\input\sample_tianchi_fresh_comp_train_user.csv" % (runningPath)
    print("loading %s", data_filename)
    data_filehandle = open(data_filename, encoding="utf-8", mode='r')
    user_behavior_csv = csv.reader(data_filehandle)
    
    fcsted_item_filename = r"%s\..\input\tianchi_fresh_comp_train_item.csv" % (runningPath)
    print(getCurrentTime(), "reading being forecasted items ", fcsted_item_filename)
    fcsted_item_df = pd.read_csv(fcsted_item_filename)
    
    fcsted_item_set = set(fcsted_item_df['item_id'].values)

    index = 0

    user_buy_set = set()

    for aline in user_behavior_csv:
        if (index == 0):
            index += 1
            continue

        user_id       = aline[0]
        item_id       = aline[1]
        behavior_type = int(aline[2])
        geo = aline[3]
        item_category = aline[4]        
        behavior_time = aline[5]

        if (index % 100000 == 0):
            print("%d lines read\r" % index,  end="")

        if (behavior_type == BEHAVIOR_TYPE_BUY):
            user_buy_set.add(user_id)
        
        index += 1

    preprocessed_data_filename = r"%s\..\input\preprocessed_user_data.csv" % (runningPath)    
    preprocessed_outputFile = open(preprocessed_data_filename, encoding="utf-8", mode='w')
    preprocessed_outputFile.write("user_id,item_id,behavior_type,item_category,time,hour\n")
    
    preprocessed_data_filename_fcsted_item_only = r"%s\..\input\preprocessed_user_data_fcsted_item_only.csv" % (runningPath)
    preprocessed_outputFile_fcsted_item_only = open(preprocessed_data_filename_fcsted_item_only, encoding="utf-8", mode='w')
    preprocessed_outputFile_fcsted_item_only.write("user_id,item_id,behavior_type,item_category,time,hour\n")

    data_filehandle.close()
    data_filehandle = open(data_filename, encoding="utf-8", mode='r')
    user_behavior_csv = csv.reader(data_filehandle)
    index = 0
    
    print("writing date into ", preprocessed_data_filename)

    for aline in user_behavior_csv:
        if (index == 0):
            index += 1
            continue

        user_id       = aline[0]
        item_id       = aline[1]
        behavior_type = int(aline[2])
        geo = aline[3]
        item_category = aline[4]
        behavior_datetime = aline[5].split(" ")
        behavior_date, behavior_hour = behavior_datetime[0], behavior_datetime[1]
        behavior_time_no_hour = aline[5].split(" ")[0]

        if (index % 100000 == 0):
            print("%d lines read\r" % index,  end="")

        if (user_id in user_buy_set):
            preprocessed_outputFile.write("%s,%s,%d,%s,%s,%s\n" % (user_id, item_id, behavior_type, item_category, behavior_date,behavior_hour))
            if (int(item_id) in fcsted_item_set):
                preprocessed_outputFile_fcsted_item_only.write("%s,%s,%d,%s,%s,%s\n" % 
                                                               (user_id, item_id, behavior_type, item_category, behavior_date,behavior_hour))
        index += 1

    data_filehandle.close()
    preprocessed_outputFile.close()
    return
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

if __name__ == '__main__':
    preprocess_data()