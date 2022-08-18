# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 21:36:52 2022

@author: shrlin
"""
import numpy as np
import pandas as pd
import pymysql

class DataLoad:
    """
    读取数据：
    1、读取常用文件格式：
        1）csv/txt读取：read_csv
        2）excel读取：read_excel
    2、读取常见数据库
        1）MySQL数据库读取：mysql_load 
        注意：引用格式为 mysql_load(db_config) 注意这里的表名需要加单引号
    """
    def __init__(self,file_path,sep=',',encoding='utf-8',db_config=None):
        """
        :param file_path 相应文件的存储路径
        :param sep 分隔符
        :param encoding  编码 
        :param db_config  数据库配置文件
        """
        self.file_path = file_path
        self.sep = sep
        self.encoding = encoding
        self.db_config = db_config
        
    def read_csv(self):
        """
        pd.read_csv可以读csv及txt等文本文件
        """        
        data = pd.read_csv(self.file_path,self.sep,self.encoding)
        return data

    def read_excel(self):
        """
        pd.read_excel可以读xls及xlsx等Excel两种格式文件
        """        
        data = pd.read_csv(self.file_path,self.sep,self.encoding)
        return data
    
    def mysql_load(self):
        """
        pd.read_sql可以读取MySQL数据库文件
        建议：对于固定位置场景，将配置直接在类中进行写入，避免每次调用，需要重复输入
        """        
        conn = pymysql.connect(host=self.db_config['host'],
                               port=self.db_config['port'],
                               user=self.db_config['user'],
                               password=self.db_config['password'],
                               db=self.db_config['db'],
                               charset=self.db_config['charset'])
        sql = "SELECT * FROM {}".format(self.db_config['table_name'])
        data = pd.read_sql(sql, conn)
        return data
    
    
class Data_Preparation:
    '''
    用于前期对数据类型的定义。
    用户把对数据的定义写进config文件/字典，
    声明字段类型有两个作用：
        1.规范字段类型  
        2.在做数据处理时可根据字段类型进行，如字符型变量不进入异常值检测
    其中：
        variable_type  0:字符型  1:整型  2:浮点型
        candidate_var_list  0:非必要变量  1:必要变量
        continuous_var_list  0:非连续型变量  1:连续型变量
        discrete_var_list  0:非离散型变量  1:离散型变量
    '''
    def __init__(self,data,config):
        """
        :param data 数据集
        :param config 数据配置文件
        """
        self.data = data
        # 读取字段类型
        self.variable_type = config[['var_name', 'var_type']].set_index(['var_name'])
        # 候选变量
        self.candidate_var_list = config[config['candidate_var_list'] == 1]['var_name']
        # 连续型变量
        self.continuous_var_list = config[(config['continuous_var_list'] == 1) & (config['candidate_var_list'] == 1)]['var_name']
        # 离散型变量
        self.discrete_var_list = config[(config['discrete_var_list'  ] == 1) & (config['candidate_var_list'] == 1)]['var_name']
        
    def change_feature_dtype(self):
        """
        更改数据框字段类型
        """
        s = '更改特征的字段类型'
        print(s.center(60,'-'))
        for vname in self.data.columns:
            
            if self.variable_type.loc[vname,'var_type'] == 0:
                type = 'category'
            elif self.variable_type.loc[vname,'var_type'] == 1:
                type = 'int64'
            elif self.variable_type.loc[vname,'var_type'] == 2:
                type = 'float'
                
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
    data = pd.read_csv('data/mfd_bank_shibor.csv')
    config = pd.DataFrame({
        'var_name' : data.columns,
        'var_type' : [0,1,1,1,1,1,1,1,1,0],
        'candidate_var_list': [0,1,1,1,1,1,1,1,1,1],
        'continuous_var_list': [0,1,1,1,1,1,1,1,1,0],
        'discrete_var_list' : [0,0,0,0,0,0,0,0,0,1]
        })
    
    pp = Data_Preparation(data,config)
    pp.change_feature_dtype()
