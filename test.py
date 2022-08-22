# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 12:17:46 2022

@author: shrlin
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from functions.feature_engineer import feature_select,feature_encode

data = pd.read_csv('data/german_credit_data_dataset.csv')
data['customer_type'].replace({1:0,2:1},inplace=True)    # 数据框
target = 'customer_type'                                 # y值

Xtr,Xts,ytr,yts = train_test_split(data.drop(target,axis=1),data[target],test_size=0.25,random_state=450)
data_tr = pd.concat([Xtr,ytr],axis=1)
data_ts = pd.concat([Xts,yts],axis=1)
data_tr.reset_index(drop=True,inplace=True)
data_ts.reset_index(drop=True,inplace=True)
# 增加一列区分训练/测试的特征
data_tr['type'] = 'train'
data_ts['type'] = 'test'

selected_data, drop_lst= feature_select.select(data_tr,target = target, empty = 0.5, iv = 0.05, corr = 0.7, return_drop=True, exclude=['type'])

selected_test = data_ts[selected_data.columns]
#quality = toad.quality(data,target)   # 数据质量检查
combiner = feature_encode.Combiner()
combiner.fit(selected_data,y=target,method='chi',min_samples = 0.05,exclude='type')
bins = combiner.export()  # 以字典形式保存分箱结果

binned_data = combiner.transform(selected_data)   # 将特征的值转化为分箱的箱号。
transer = feature_encode.WOETransformer()        # 计算WOE

# 对WOE的值进行转化，映射到原数据集上。对训练集用fit_transform,测试集用transform.
data_tr_woe = transer.fit_transform(binned_data, binned_data[target], exclude=[target,'type'])
data_ts_woe = transer.transform(combiner.transform(selected_test))

print(data_tr_woe)
print(data_ts_woe)


combiner.
combiner.fit
combiner.transform
combiner.export
combiner.rules
combiner.update
combiner.load
combiner.set_rules()










from functions.feature_engineer.feature_standard import DataStandard
import pickle as pkl
import pandas as pd

train = pd.DataFrame({'a':list(range(0,100)),'b':list(range(-100,0))})
DS = DataStandard(train)
tmp = DS.minmax_scaler()

output = open('result/fit.pkl', 'wb') 
pkl.dump(DS.transform_, output, -1)
output.close()


'/**********************************************************************/'
import pickle as pkl
import pandas as pd
pkl_file = open('result/fit.pkl', 'rb')
transform_ = pkl.load(pkl_file)

test = pd.DataFrame({'a':[1,5,7],'b':[-7,-5,-1]})
transform_(test)
