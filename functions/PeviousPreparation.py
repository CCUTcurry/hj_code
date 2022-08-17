# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 21:36:52 2022

@author: shrlin
"""
import numpy as np
import pandas as pd

class Pevious_Preparation:
    '''
    用于前期对数据类型的定义
    '''
    def __init__(self,data,config):
        self.data     = data
        self.__config = config
        self.variable_type       = self.__config[['var_name', 'var_type']]
        self.variable_type       = self.variable_type.set_index(['var_name'])
        self.candidate_var_list  = self.__config[self.__config['candidate_var_list'] == 1]['var_name']
        self.continuous_var_list = self.__config[(self.__config['continuous_var_list'] == 1) & (self.__config['candidate_var_list'] == 1)]['var_name']
        self.discrete_var_list   = self.__config[(self.__config['discrete_var_list'  ] == 1) & (self.__config['candidate_var_list'] == 1)]['var_name']
        
    def change_feature_dtype(self):
        """
        更改数据框字段类型
        """
        s = '更改特征的字段类型'
        print(s.center(60,'-'))
        for vname in self.data.columns:
            if self.variable_type.loc[vname,'var_type'] == 1:
                type = 'float64'
            elif self.variable_type.loc[vname,'var_type'] == 0:
                type = 'object'
            try:
                self.data[vname] = self.data[vname].astype(type)
                print(vname,' '*(40-len(vname)),'{0: >10}'.format(type))
            except Exception:
                print('[error]',vname)
                print('[original dtype] ',self.data.dtypes[vname],' [astype] ',type)
                print('[unique value]',np.unique(self.data[vname]))

        s = '更改完成'
        print(s.center(60,'-'))

if __name__ == '__main__':
    data = pd.read_csv('mfd_bank_shibor.csv')
    config = pd.DataFrame({
        'var_name' : data.columns,
        'var_type' : [0,1,1,1,1,1,1,1,1,0],
        'candidate_var_list': [0,1,1,1,1,1,1,1,1,1],
        'continuous_var_list': [0,1,1,1,1,1,1,1,1,0],
        'discrete_var_list' : [0,0,0,0,0,0,0,0,0,1]
        })
    
    pp = Pevious_Preparation(data,config)
    pp.change_feature_dtype()
    data = pp.data
