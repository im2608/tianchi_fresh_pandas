
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


redis_cli = redis.Redis(host='10.57.14.6', port=6379, db=1)

# category的数量
category_cnt = 9557

# 在category中最多有多少item
max_item_in_category = 786870


ONLINE_FORECAST_DATE = datetime.datetime.strptime("2014-12-19", "%Y-%m-%d").date()

g_only_from_test_set = False

#正负样本比例 1： g_nag_times
g_nag_times = 5

