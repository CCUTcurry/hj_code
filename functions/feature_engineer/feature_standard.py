#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler


class DataStandard():
    """
    数据标准化：
        1）z = (x - mean) / std ：standard_scaler
        2）z = (X - X_min) / (X_max - X_min) ：minmax_scaler
        3) 小数定标规范化,通过移动小数点的位置来进行规范化。
           小数点移动多少位取决于属性A的取值中的最大绝对值:min_label_scaler
    """
    def __init__(self,data):
        self.data = data
        self.transform_ = None
        
    def standard_scaler(self,X_test=None):
        """
        作用：数据标准化        
        公式：z = (x - mean) / std
        X_test：DataFrame ,测试集，X_test = DataFrame 则应用在测试集上
        return: DataFrame，处理后数据        
        """
        s = StandardScaler() 
        self.data = s.fit_transform(self.data)   # x_train_scale
        self.transform_ = s.transform
        
        if X_test is not None:                           
            X_test_scale = s.transform(X_test)
            return self.data,X_test_scale
        else:
            return self.data
            
    def minmax_scaler(self,X_test=None):
        """
        作用：数据标准化        
        公式：z = (X - X_min) / (X_max - X_min) 
        X_test：DataFrame ,测试集，X_test = DataFrame 则应用在测试集上
        return: DataFrame，处理后数据       
        """
        m = MinMaxScaler()
        self.data = m.fit_transform(self.data)   # x_train_scale
        self.transform_ = m.transform
        
        if X_test is not None:                               
            X_test_scale = m.transform(X_test)
            return self.data,X_test_scale
        else:
            return self.data 
   
    def min_label_scaler(self,X_test=None):
        """
        作用：数据标准化        
        原理：通过移动小数点的位置来进行规范化。小数点移动多少位取决于属性A的取值中的最大绝对值 
        X_test：DataFrame ,测试集，X_test = DataFrame 则应用在测试集上
        return: DataFrame，处理后数据    
        """
        mul = np.ceil(np.log10(np.max(abs(self.data))))
        self.data = self.data / (10**mul)     # x_train_scale  
        if X_test is not None:                               
            X_test_scale = X_test / (10**mul)
            return self.data,X_test_scale
        else:
            return self.data

        
if __name__ == '__main__':
    df1 = pd.DataFrame({'id':[1,2,3,4,5],'grade':list('abaab'),'color':['红','黄','绿','红','黄'],'ii':[10,70,80,14,30]})
    fe = DataStandard(df1['ii'])
    fe.minmax_scaler()

