'''
Created on Aug 4, 2017

@author: Heng.Zhang
'''

import datetime
import pandas as pd
import numpy as np


def convertDatatimeToStr(opt_datatime):
    return "%04d-%02d-%02d" % (opt_datatime.year, opt_datatime.month, opt_datatime.day)

# 将date string 转成与checking date 之间相差的天数 
def convert_date_str_to_dayoffset(date_str_series, slide_window_size, checking_date):
    slide_window_date_dayoffset = {}
    for i in range(slide_window_size, 0, -1):
        date_in_window = checking_date + datetime.timedelta(days=-i)
        date_str = "%4d-%02d-%02d" % (date_in_window.year, date_in_window.month, date_in_window.day)
        slide_window_date_dayoffset[date_str] =  i
        
    dayoffset_series = pd.Series(list(map(lambda x : slide_window_date_dayoffset[x], date_str_series)))
    
    return dayoffset_series
        
        
def SeriesDivision(divisor, dividend):
    quotient = divisor / dividend
    quotient.fillna(0, inplace=True)
    quotient[np.isinf(quotient)] = 0
    return quotient