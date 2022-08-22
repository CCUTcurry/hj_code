# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:36:50 2022

@author: shrlin
"""

import pandas as pd
from functools import reduce


def fotmat_series(data,col = 'date',format_ = 'yyyymmdd',dim = 'D',method = 'mean'):
    """
    更改数据集日期格式，将数据集汇总至指定维度
    :param data:数据集
    :param col:日期标签列
    :param format_:原始数据格式
    :param dim:转换后的维度
    :param method:转换方法，目前支持平均或者求和
    :return data:处理好的数据
    """
    
    data[col] = data[col].apply(lambda x : str(x).replace('-', ''))
    
    if format_ == 'yyyy':
        data[col] = data[col].apply(lambda x : str(x)+'0101')
    elif format_ == 'yyyymm' or format_ == 'yyyy-mm':
        data[col] = data[col].apply(lambda x : str(x)+'01')
    
    data[col] = pd.to_datetime(data[col])
    data.set_index(col,drop=True, append=False, inplace=True)
    
    if method == 'mean':
        hand = 'data.resample(dim).mean()'
    elif method == 'sum':
        hand = 'data.resample(dim).sum()'
    data = eval(hand)
    
    return data

def get_df_merge(col,how,*df_name):
    """
    更改数据集日期格式，将数据集汇总至指定维度
    :param col:合并列
    :param how:left/right/outer
    :param *df_name:输入数据框
    :return df_merged:合并后的数据
    """

    df_groups = list(df_name)
    df_merged = reduce(lambda left, right: pd.merge(left, right, how = how,on = col), df_groups)
    
    return df_merged

if __name__ == '__main__':
    data1 = pd.DataFrame({
        'date' : [1999,2000,2001,2002,2003,2004,2006,2008,2010,2022],
        'data' : [1,2,3,4,5,6,7,8,9,10]
        })
    
    data2 = pd.DataFrame({
        'date' : [199910,200001,200102,200203,200304,200402,200607,200810,201010,202202,202203],
        'data' : [1,2,3,4,5,6,7,8,9,10,11]
        })
    
    d1 = fotmat_series(data1.copy(),col = 'date',format_ = 'yyyy',dim = 'Y',method = 'mean')
    d2 = fotmat_series(data2.copy(),col = 'date',format_ = 'yyyymm',dim = 'Y',method = 'sum')
    df_merge = get_df_merge('date','left',d2,d1)
