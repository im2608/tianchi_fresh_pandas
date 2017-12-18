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
# datetime分成date 和 hour 两列
# date 得到星期几， 星期一 weekday = 0, 星期日 weekday = 6 
# 删除只在12-12购买的用户，删掉只在12-12有人买的item
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

    user_buy_dict = {}
    item_sold_dict = {}
    
    # 只在12-12购买过的user,  删除
    user_1212_only_set = set()
    
    # 只在12-12卖出过的item,  删除
    item_1212_only_set = set()

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

        if (behavior_type == 4):
            behavior_date = behavior_time.split(" ")[0]
            
            if (user_id not in user_buy_dict):
                user_buy_dict[user_id] = set()
                
            user_buy_dict[user_id].add(behavior_date)
                
            if (item_id not in item_sold_dict):
                item_sold_dict[item_id] = set()
            
            item_sold_dict[item_id].add(behavior_date)
        index += 1

    preprocessed_data_filename = r"%s\..\input\preprocessed_user_data.csv" % (runningPath)    
    preprocessed_outputFile = open(preprocessed_data_filename, encoding="utf-8", mode='w')
    preprocessed_outputFile.write("user_id,item_id,behavior_type,item_category,time,hour,weekday\n")
    
    preprocessed_data_filename_fcsted_item_only = r"%s\..\input\preprocessed_user_data_fcsted_item_only.csv" % (runningPath)
    preprocessed_outputFile_fcsted_item_only = open(preprocessed_data_filename_fcsted_item_only, encoding="utf-8", mode='w')
    preprocessed_outputFile_fcsted_item_only.write("user_id,item_id,behavior_type,item_category,time,hour,weekday\n")
    
    preprocessed_data_filename_sold_item_only = r"%s\..\input\preprocessed_user_data_sold_item_only.csv" % (runningPath)
    preprocessed_outputFile_sold_item_only = open(preprocessed_data_filename_sold_item_only, encoding="utf-8", mode='w')
    preprocessed_outputFile_sold_item_only.write("user_id,item_id,behavior_type,item_category,time,hour,weekday\n")

    data_filehandle.close()
    data_filehandle = open(data_filename, encoding="utf-8", mode='r')
    user_behavior_csv = csv.reader(data_filehandle)
    index = 0
    
    users_buy = user_buy_dict.keys()
    for user_id in users_buy:
        if (len(user_buy_dict[user_id]) == 1):
            buy_date = list(user_buy_dict[user_id])[0]
            if (buy_date == '2014-12-12'):
                user_1212_only_set.add(user_id)
                
    item_sold = item_sold_dict.keys()
    for item_id in item_sold:
        if (len(item_sold_dict[item_id]) == 1):
            sold_date = list(item_sold_dict[item_id])[0]
            if (sold_date == '2014-12-12'):
                item_1212_only_set.add(item_id)                

    print("%s here are %d users bought only on 12-12, here are %d items sold only on 12-12" % 
          (getCurrentTime(), len(user_1212_only_set), len(item_1212_only_set)))

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
        weekday = datetime.datetime.strptime(behavior_date, "%Y-%m-%d").weekday()

        if (index % 100000 == 0):
            print("%d lines read\r" % index,  end="")

        if (user_id in user_buy_dict and user_id not in user_1212_only_set): 
            preprocessed_outputFile.write("%s,%s,%d,%s,%s,%s,%d\n" % 
                                          (user_id, item_id, behavior_type, item_category, behavior_date,behavior_hour,weekday))
            if (int(item_id) in fcsted_item_set):
                preprocessed_outputFile_fcsted_item_only.write("%s,%s,%d,%s,%s,%s,%d\n" % 
                                                               (user_id, item_id, behavior_type, item_category, behavior_date,behavior_hour,weekday))
            if (item_id in item_sold_dict and item_id not in item_1212_only_set):
                preprocessed_outputFile_sold_item_only.write("%s,%s,%d,%s,%s,%s,%d\n" % 
                                                             (user_id, item_id, behavior_type, item_category, behavior_date,behavior_hour,weekday))
        index += 1

    data_filehandle.close()
    preprocessed_outputFile.close()
    preprocessed_outputFile_sold_item_only.close()
    print(getCurrentTime(), "Done")
    return

def preprocess_data_no1212():

    data_filename = r"%s\..\input\tianchi_fresh_comp_train_user.csv" % (runningPath)
#     data_filename = r"%s\..\input\sample_tianchi_fresh_comp_train_user.csv" % (runningPath)
    print("loading %s", data_filename)
    data_filehandle = open(data_filename, encoding="utf-8", mode='r')
    user_behavior_csv = csv.reader(data_filehandle)
    
    fcsted_item_filename = r"%s\..\input\tianchi_fresh_comp_train_item.csv" % (runningPath)
    print(getCurrentTime(), "reading being forecasted items ", fcsted_item_filename)
    fcsted_item_df = pd.read_csv(fcsted_item_filename, dtype={'item_id':np.str})
    
    fcsted_item_set = set(fcsted_item_df['item_id'].values)

    index = 0

    user_buy_dict = {}
    item_sold_dict = {}
    
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

        if (behavior_type == 4):
            behavior_date = behavior_time.split(" ")[0]
            
            if (user_id not in user_buy_dict):
                user_buy_dict[user_id] = set()
                
            user_buy_dict[user_id].add(behavior_date)
                
            if (item_id not in item_sold_dict):
                item_sold_dict[item_id] = set()
            
            item_sold_dict[item_id].add(behavior_date)
        index += 1

    preprocessed_data_filename = r"%s\..\input\preprocessed_user_data.csv" % (runningPath)    
    preprocessed_outputFile = open(preprocessed_data_filename, encoding="utf-8", mode='w')
    preprocessed_outputFile.write("user_id,item_id,behavior_type,item_category,time,hour,weekday\n")
    
    preprocessed_data_filename_fcsted_item_only = r"%s\..\input\preprocessed_user_data_fcsted_item_only.csv" % (runningPath)
    preprocessed_outputFile_fcsted_item_only = open(preprocessed_data_filename_fcsted_item_only, encoding="utf-8", mode='w')
    preprocessed_outputFile_fcsted_item_only.write("user_id,item_id,behavior_type,item_category,time,hour,weekday\n")
    
    preprocessed_data_filename_sold_item_only = r"%s\..\input\preprocessed_user_data_sold_item_only.csv" % (runningPath)
    preprocessed_outputFile_sold_item_only = open(preprocessed_data_filename_sold_item_only, encoding="utf-8", mode='w')
    preprocessed_outputFile_sold_item_only.write("user_id,item_id,behavior_type,item_category,time,hour,weekday\n")

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
        
#          删除12-12的数据， 12-12之后的数据顺次前移一天,所以预测12-19变成了预测12-18
        if (behavior_date == '2014-12-12'):
            continue
        if (behavior_date > '2014-12-12'):
            behavior_date = datetime.datetime.strptime(behavior_date, "%Y-%m-%d")
            behavior_date = behavior_date - datetime.timedelta(days=1)
            behavior_date = convertDatatimeToStr(behavior_date)

        weekday = datetime.datetime.strptime(behavior_date, "%Y-%m-%d").weekday()

        if (index % 100000 == 0):
            print("%d lines read\r" % index,  end="")

        if (user_id in user_buy_dict): 
            preprocessed_outputFile.write("%s,%s,%d,%s,%s,%s,%d\n" % 
                                          (user_id, item_id, behavior_type, item_category, behavior_date,behavior_hour,weekday))
            if (item_id in fcsted_item_set):
                preprocessed_outputFile_fcsted_item_only.write("%s,%s,%d,%s,%s,%s,%d\n" % 
                                                               (user_id, item_id, behavior_type, item_category, behavior_date,behavior_hour,weekday))
            if (item_id in item_sold_dict):
                preprocessed_outputFile_sold_item_only.write("%s,%s,%d,%s,%s,%s,%d\n" % 
                                                             (user_id, item_id, behavior_type, item_category, behavior_date,behavior_hour,weekday))
        index += 1

    data_filehandle.close()
    preprocessed_outputFile.close()
    preprocessed_outputFile_sold_item_only.close()
    print(getCurrentTime(), "Done")
    return

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

if __name__ == '__main__':
    preprocess_data()