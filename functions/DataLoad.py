#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pymysql
import pandas as pd

class Data_Load():
    """
    读取数据：
    1、读取常用文件格式：
        1）csv读取：csv_load
        2）txt读取：csv_load
        3）excel读取：excel_load
    2、读取常见数据库
        1）MySQL数据库读取：mysql_load 
        注意：引用格式为 mysql_load('表名') 注意这里的表名需要加单引号
    """
    def __init__(self, path):
        """
        :param path			相应文件的存储路径
        """
        self.path = path
        self.table_name = table_name
    def csv_load(self):
        """
        pd.read_csv可以读csv及txt等文本文件
        """        
        data = pd.read_csv(self.path)
        return data

    def excel_load(self):
        """
        pd.read_excel可以读xls及xlsx等Excel两种格式文件
        """        
        data = pd.read_excel(self.path)
        return data
    
    def mysql_load(self):
        """
        pd.read_sql可以读取MySQL数据库文件
        建议：对于固定位置场景，将配置直接在类中进行写入，避免每次调用，需要重复输入
        """        
        conn = pymysql.connect(host="127.0.0.1",
                           port=3306,
                           user="root",
                           password="1234",
                           db="test",
                           charset="utf8")
        sql = "SELECT * FROM {}".format(self.table_name)
        data = pd.read_sql(sql, conn)
        return data

