'''
Created on Oct 12, 2017

@author: Heng.Zhang
'''

import pandas as pd
from global_variables import *
import numpy as np
from common import *
from feature_common import *


# item 第一次, 最后一次 behavior 距离checking date 的天数, 以及第一次, 最后一次之间的天数， 返回 12 个特征
def feature_category_days_from_1st_last_behavior(slide_window_df, UIC, feature_matrix):
    behavior_dayoffset = feature_days_from_1st_last_behavior(slide_window_df, UIC, 'item_category')
    feature_matrix = pd.merge(feature_matrix, behavior_dayoffset, how='left', on=['item_category'])
    
    return feature_matrix