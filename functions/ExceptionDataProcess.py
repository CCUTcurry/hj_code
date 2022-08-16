# creator       :Hazel
# create date   :2022/08/15

import pandas as pd
import numpy as np

class ExceptionDataCheck():
    """
    异常值检测
        1. 四分位距（IQR）检测异常值
        2. 3σ原则检测异常值
    """

    def __init__(self, data):
        """
        :param  data 待检测数据集
        """
        self.__data = data
        self.__edc_cnt = 0      # 记录变量异常值个数
        self.__edc_max_cnt = 0  # 记录变量异常大值个数
        self.__edc_min_cnt = 0  # 记录变量异常小值个数
        self.__max = 0          # 正常范围区间的上限
        self.__min = 0          # 正常范围区间的下限
        self.data_check_IQR = pd.DataFrame() # 四分位距（IQR）检测结果
        self.data_check_STD = pd.DataFrame() # 3σ原则检测结果

    def __EDC_check_IQR(self, data_list):
        """
        使用四分位距（IQR），计算上下限，用于异常值检测
        :param  data_list 待检测数据
        """
        ID_p25 = pd.DataFrame(data_list).quantile(0.25)[0]  # 下四分位数位置
        ID_p75 = pd.DataFrame(data_list).quantile(0.75)[0]  # 上四分位数位置
        IQR = ID_p75 - ID_p25  # 计算数组IQR
        self.__max = ID_p75 + 1.5 * IQR  # 计算上限
        self.__min = ID_p25 - 1.5 * IQR  # 计算下限

    def __EDC_check_STD(self, data_list):
        """
        使用3σ原则检测异常值，计算上下限，用于异常值检测
        :param  data_list 待检测数据
        """
        self.__max = np.mean(data_list) + 3 * np.std(data_list)  # 计算上限
        self.__min = np.mean(data_list) - 3 * np.std(data_list)  # 计算下限

    def __EDC_check(self, type, data_list):
        """
        异常值检测
        :param  type 异常值检测方法
        :param  data_list 待检测数据
        """
        if type == 'IQR':
            self.__EDC_check_IQR(data_list)
        if type == 'STD':
            self.__EDC_check_STD(data_list)

        for i in data_list:     # 异常值判断
            if i < self.__min:
                self.__edc_min_cnt = self.__edc_min_cnt + 1
            if i > self.__max:
                self.__edc_max_cnt = self.__edc_max_cnt + 1
        self.__edc_cnt = self.__edc_min_cnt + self.__edc_max_cnt

    def EDC_check(self, type ,feature=None):
        """
        检测结果输出
        :param  type 异常值检测方法
        :param  feature 需要检测的变量，若无指定变量，则data集所有变量都进行检测
        :return data_check 检测结果
        """
        if feature is None:
            data = self.__data
        else:
            data = pd.DataFrame(self.__data, columns=feature)

        columns = ['异常值个数','异常大值个数','异常小值个数','上限','下限']
        data_check = pd.DataFrame(columns=columns)

        edc_feature = []
        for data_list in data:
            self.__EDC_check(type=type,data_list=data[data_list])
            if self.__edc_cnt > 0:
                data_check.loc[len(data_check.index)] = [self.__edc_cnt, self.__edc_max_cnt, self.__edc_min_cnt,
                                                         self.__max, self.__min]
                edc_feature.append(data_list)

        data_check.index = edc_feature
        if type == 'IQR':
            self.data_check_IQR = data_check
        if type == 'STD':
            self.data_check_STD = data_check

        return data_check


class ExceptionDataClean():
    """
    异常值清洗
        1. 删除异常值记录
        2. 平均值修正异常值
        3. 自定义数值修正异常值
    """

    def __init__(self, data):
        """
        :param  data 待处理数据集
        """
        self.data = pd.DataFrame(data)

    def EDC_drop(self, feature ,e_data):
        """
        删除异常值记录
        :param  feature 需要进行异常值处理的变量名
        :param  e_data 异常值
        """
        self.data = self.data.drop(self.data[self.data[feature] == e_data].index)
        print(feature, '值为', e_data, '的记录已删除')

    def EDC_mean(self, feature ,e_data):
        """
        平均值修正异常值
        :param  feature 需要进行异常值处理的变量名
        :param  e_data 异常值
        """
        self.data.loc[self.data[feature].isin([e_data]), feature] = np.mean(self.data[feature])
        print(feature, '已将', e_data, '替换为平均值', np.mean(self.data[feature]))

    def EDC_myself(self, feature ,e_data, my_data):
        """
        自定义数值修正异常值
        :param  feature 需要进行异常值处理的变量名
        :param  e_data 异常值
        :param  my_data 自定义数值
        """
        self.data.loc[self.data[feature].isin([e_data]), feature] = my_data
        print(feature, '已将', e_data, '替换为', my_data)




if __name__ == '__main__':
    data = pd.read_csv('C:/Users/Hazel/Desktop/Hazel/学习/02.数据挖掘/02.建模实践/01.资金流入流出预测/分析代码/'
                       'Purchase Redemption Data/mfd_bank_shibor.csv')

    # 异常值检测类测试
    feature = ['Interest_1_M', 'Interest_3_M', 'Interest_9_M']
    EDC1 = ExceptionDataCheck(data)
    EDC1.EDC_check(type='IQR', feature=feature)
    EDC1.EDC_check(type='STD', feature=feature)

    EDC2 = ExceptionDataCheck(data)
    EDC2.EDC_check(type='IQR')
    EDC2.EDC_check(type='STD')

    print(EDC1.data_check_IQR)
    print(EDC1.data_check_STD)
    print(EDC2.data_check_IQR)
    print(EDC2.data_check_STD)

    # 异常值清洗类测试
    EDClean = ExceptionDataClean(data)
    EDClean.EDC_drop(feature='Interest_1_M', e_data=999)
    EDClean.EDC_mean(feature='Interest_3_M', e_data=999)
    EDClean.EDC_myself(feature='Interest_9_M', e_data=999, my_data=100)