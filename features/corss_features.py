'''
Created on Oct 12, 2017

@author: Heng.Zhang
'''

import pandas as pd
from global_variables import *
import numpy as np
from common import *
from feature_common import *


# user 在 item上各个行为的次数占user在 category 上各个行为次数的比例
def feature_user_item_cnt_ratio_on_category(feature_matrix_df):
    feature_matrix_df['item_view_cnt_ratio_on_cat'] = SeriesDivision(feature_matrix_df['item_view_cnt'], feature_matrix_df['category_view_cnt'])
    feature_matrix_df['item_favorite_cnt_ratio_on_cat'] = SeriesDivision(feature_matrix_df['item_favorite_cnt'], feature_matrix_df['category_favorite_cnt'])
    feature_matrix_df['item_cart_cnt_ratio_on_cat'] = SeriesDivision(feature_matrix_df['item_cart_cnt'], feature_matrix_df['category_cart_cnt'])
    feature_matrix_df['item_buy_cnt_ratio_on_cat'] = SeriesDivision(feature_matrix_df['item_buy_cnt'], feature_matrix_df['category_buy_cnt'])
    
    return feature_matrix_df