
import sys
import redis
import datetime

runningPath = sys.path[0]
sys.path.append("%s\\features\\" % runningPath)


USER_ID = "user_id"
ITEM_ID = "item_id"
BEHAVEIOR_TYPE = "behavior_type"
USER_GEO = "user_geohash"
ITEM_CATE = "item_category"
TIME = "time"

algo = ""

BEHAVIOR_TYPE_VIEW = 1
BEHAVIOR_TYPE_FAV  = 2
BEHAVIOR_TYPE_CART = 3
BEHAVIOR_TYPE_BUY  = 4


tianchi_fresh_comp_train_user = "%s\\..\\input\\preprocessed_tianchi_fresh_comp_train_user.csv" % runningPath
tianchi_fresh_comp_train_item = "%s\\..\\input\\tianchi_fresh_comp_train_item.csv" % runningPath


# tianchi_fresh_comp_train_user = "%s\\..\\input\\splitedInput\\datafile.000" % runningPath

#用户的购买记录
g_user_buy_transection = dict()  # 以 user id 为 key
g_user_buy_transection_item = dict()  # 以 item id 为 key

#在用户的操作记录中，从最后一条购买记录到train时间结束之间的操作行为记录，
#以它们作为patten
g_user_behavior_patten = dict() # 以 user id 为 key
g_user_behavior_patten_item = dict() # 以 item id 为 key

#总共的购买记录数
g_buy_record_cnt = 0

g_pattern_cnt = 0.0

g_users_for_alog = []

# 用户第一次接触商品到购买该商品之间的天数与用户购买该用户的可能性。天数越长，可能性越小
g_prob_bwteen_1st_days_and_buy = {1:0.0571, 
                                  2:0.032, 
                                  3:0.0221, 
                                  4:0.0164, 
                                  5:0.0138, 
                                  6:0.0098, 
                                  7:0.0089, 
                                  8:0.0077, 
                                  9:0.0062, 
                                  10:0.0055}

g_behavior_weight = {BEHAVIOR_TYPE_VIEW : 1,
                     BEHAVIOR_TYPE_FAV  : 33, 
                     BEHAVIOR_TYPE_CART : 47,
                     BEHAVIOR_TYPE_BUY  : 94}


g_pattern_cnt = 0.0

# 以 user id 为 key
global_user_item_dict = dict()

# 以item category 为 key
global_item_user_dict = dict()

# 测试集数据的item - category 对应关系，以 item id 为key
global_test_item_category = dict()
# 测试集数据的item - category 对应关系，以 category 为key
global_test_category_item = dict()

#训练数据集中的item - category 对应关系， 以 item 为 key
global_train_item_category = dict()
#训练数据集中的item - category 对应关系， 以 category 为 key
global_train_category_item = dict()

global_totalBehaviorWeightHash = dict()
global_user_behavior_cnt = dict()

redis_cli = redis.Redis(host='10.57.14.6', port=6379, db=1)

# category的数量
category_cnt = 9557

# 在category中最多有多少item
max_item_in_category = 786870


ONLINE_FORECAST_DATE = datetime.datetime.strptime("2014-12-19", "%Y-%m-%d").date()

g_only_from_test_set = False