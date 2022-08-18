# creator       :Hazel
# create date   :2022/8/17

import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class MultiEvaluate():
    """
    多分类模型评价指标：准确率（accuracy）、精确率（precision）、召回率（recall）、F1值
    """

    def __init__(self, y_true=None, y_pred=None, labels=None):
        """
        :param y_true:真实值
        :param y_pred:预测值
        :param labels:类别标签
        """
        self.__y_true = y_true
        self.__y_pred = y_pred
        self.__labels = labels

    def get_CM(self, y_true, y_pred, labels):
        """
        计算混淆矩阵
        :param y_true:真实值
        :param y_pred:预测值
        :param labels:类别标签
        :return cm:混淆矩阵
        """
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
        return cm

    def CM_heatmap(self, cm, labels):
        """
        计算混淆矩阵，并绘制混淆矩阵热力图
        :param cm:混淆矩阵
        :param labels:类别标签
        """
        sns.set(font="simhei")
        heatmap = sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues',
                              annot_kws={"fontsize": 15})
        heatmap.set_title('混淆矩阵热力图')
        heatmap.set_xlabel('predict')
        heatmap.set_ylabel('true')
        plt.show()

    def TP_FN_FP_TN(self, cm, labels):
        """
        根据混淆矩阵，计算各个分类的的TP，FN，FP，TN值
        :param cm:混淆矩阵
        :param labels:类别标签
        :return tp_fn_fp_tn:各类别的TP，FN，FP，TN值
        """
        l = len(cm)
        tp_fn_fp_tn = pd.DataFrame(columns=labels)
        tp, fn, fp, tn = [], [], [], []
        for i in range(0, l):
            tp_ = 0
            fn_ = 0
            fp_ = 0
            for j in range(0, l):
                if i == j:
                    tp_ = tp_ + cm[i][j]
                else:
                    fn_ = fn_ + cm[i][j]
                    fp_ = fp_ + cm[j][i]
            tp.append(tp_)
            fn.append(fn_)
            fp.append(fp_)
            tn.append(sum(sum(cm)) - tp_ - fn_ - fp_)

        tp_fn_fp_tn.loc['TP'] = tp
        tp_fn_fp_tn.loc['FN'] = fn
        tp_fn_fp_tn.loc['FP'] = fp
        tp_fn_fp_tn.loc['TN'] = tn
        return tp_fn_fp_tn

    def accuracy_score(self, tp_fn_fp_tn):
        """
        计算模型准确率，准确率计算方式仅一种
        :param tp_fn_fp_tn:各类别的TP，FN，FP，TN值
        :return a_score:准确率
        """
        TP = sum(tp_fn_fp_tn.loc['TP'])
        N = sum(tp_fn_fp_tn[tp_fn_fp_tn.columns[0]])
        a_score = TP / N
        return a_score

    def micro_P_R_F(self, tp_fn_fp_tn):
        """
        MICRO，全局计算精确率，召回率，F1分数
        :param tp_fn_fp_tn:各类别的TP，FN，FP，TN值
        :return p_score:精确率
        :return r_score:召回率
        :return f_score:F1分数
        """
        TP, FN, FP = sum(tp_fn_fp_tn.loc['TP']), sum(tp_fn_fp_tn.loc['FN']), sum(tp_fn_fp_tn.loc['FP'])
        p_score = TP / (TP + FP)
        r_score = TP / (TP + FN)
        f_score = 2 * p_score * r_score / (p_score + r_score)
        return p_score, r_score, f_score

    def macro_P_R_F(self, tp_fn_fp_tn):
        """
        MACRO，分别计算每一类的精确率、召回率、f1分数，后求算术平均值
        :param tp_fn_fp_tn:各类别的TP，FN，FP，TN值
        :return p_score:精确率
        :return r_score:召回率
        :return f_score:F1分数
        """
        n, p_s, r_s, f_s = 0, 0, 0, 0
        for i in tp_fn_fp_tn:
            TP, FP, FN = tp_fn_fp_tn[i]['TP'], tp_fn_fp_tn[i]['FP'], tp_fn_fp_tn[i]['FN']
            p, r = TP / (TP + FP), TP / (TP + FN)
            n, p_s, r_s, f_s = n+1, p_s + p, r_s + r, f_s + 2 * p * r / (p + r)
        p_score = p_s / n
        r_score = r_s / n
        f_score = f_s / n
        return p_score, r_score, f_score

    def weighted_P_R_F(self, tp_fn_fp_tn):
        """
        WEIGHTED, 分别计算每一类的精确率、召回率、f1分数，后求加权平均值
        :param tp_fn_fp_tn:各类别的TP，FN，FP，TN值
        :return p_score:精确率
        :return r_score:召回率
        :return f_score:F1分数
        """
        N = sum(tp_fn_fp_tn[tp_fn_fp_tn.columns[0]])
        n, p_s, r_s, f_s = 0, 0, 0, 0
        for i in tp_fn_fp_tn:
            TP, FP, FN = tp_fn_fp_tn[i]['TP'], tp_fn_fp_tn[i]['FP'], tp_fn_fp_tn[i]['FN']
            p, r, k = TP / (TP + FP), TP / (TP + FN), (TP + FN) / N
            n, p_s, r_s, f_s = n + 1, p_s + p * k, r_s + r * k, f_s + 2 * p * r / (p + r) * k
        p_score = p_s
        r_score = r_s
        f_score = f_s
        return p_score, r_score, f_score

    def evaluate_report(self, model=None):
        """
        输出模型效果报告
        :param model:指标计算方式（micro、macro、weighted）
        """
        cm = self.get_CM(self.__y_true, self.__y_pred, self.__labels)
        self.CM_heatmap(cm, self.__labels)
        tp_fn_fp_tn = self.TP_FN_FP_TN(cm, self.__labels)
        a_score = self.accuracy_score(tp_fn_fp_tn)
        print( '多分类模型效果评估')
        print(('------------------------------'))
        print('准确率：','{:.3%}'.format(a_score))
        if model == 'micro' or model is None:
            p_score, r_score, f_score = self.micro_P_R_F(tp_fn_fp_tn)
            print('MICRO：全局计算精确率，召回率，F1分数')
            print('精确率：', '{:.3%}'.format(p_score))
            print('召回率：', '{:.3%}'.format(r_score))
            print('F1值：', '{:.3%}'.format(f_score))
        if model == 'macro' or model is None:
            p_score, r_score, f_score = self.macro_P_R_F(tp_fn_fp_tn)
            print('MACRO：分别计算每一类的精确率、召回率、f1分数，后求算术平均值')
            print('精确率：', '{:.3%}'.format(p_score))
            print('召回率：', '{:.3%}'.format(r_score))
            print('F1值：', '{:.3%}'.format(f_score))
        if model == 'weighted' or model is None:
            p_score, r_score, f_score = self.weighted_P_R_F(tp_fn_fp_tn)
            print('WEIGHTED：分别计算每一类的精确率、召回率、f1分数，后求加权平均值')
            print('精确率：', '{:.3%}'.format(p_score))
            print('召回率：', '{:.3%}'.format(r_score))
            print('F1值：', '{:.3%}'.format(f_score))
        else:
            print('请选择正确的model：micro，macro，weighted')


class MultiCompare():
    """
    多分类模型效果比较
    """

    def __init__(self, y_true, y_pred, model_nm, labels):
        """
        :param y_true:真实值
        :param y_pred:预测值
        :param model_nm:模型名单，模型名单顺序与预测值名单顺序一致
        :param labels:类别标签
        """
        self.__y_true = y_true
        self.__y_pred = y_pred
        self.__model_nm = model_nm
        self.__labels = labels

    def compare_report(self):
        """
        输出模型效果比较报告
        """
        ME = MultiEvaluate()
        c_report = pd.DataFrame(columns=self.__model_nm)
        for i in range(len(self.__model_nm)):
            model_nm = self.__model_nm[i]
            cm = ME.get_CM(self.__y_true, self.__y_pred[i], self.__labels)
            tp_fn_fp_tn = ME.TP_FN_FP_TN(cm, self.__labels)
            a_s = ME.accuracy_score(tp_fn_fp_tn)
            micro_p, micro_r, micro_f = ME.micro_P_R_F(tp_fn_fp_tn)
            macro_p, macro_r, macro_f = ME.macro_P_R_F(tp_fn_fp_tn)
            weigh_p, weigh_r, weigh_f = ME.weighted_P_R_F(tp_fn_fp_tn)
            c_report[model_nm] = [a_s, micro_p, micro_r, micro_f, macro_p, macro_r, macro_f, weigh_p, weigh_r, weigh_f]
        index = ['准确率', 'micro精确率', 'micro召回率', 'microF1值', 'macro精确率', 'macro召回率', 'macroF1值',
                 'weighted精确率', 'weighted召回率', 'weightedF1值']
        c_report.index = index
        return c_report

if __name__ == '__main__':
    # 模型效果评估报告输出测试
    y_true = [1, 2, 2, 1, 3, 3, 1, 1]
    y_pred = [1, 2, 1, 1, 3, 3, 3, 1]
    labels = [1, 2, 3]
    MultiEvaluate(y_true, y_pred, labels).evaluate_report()

    # 模型效果比较结果输出
    y_true = [1, 2, 2, 1, 3, 3, 1, 1]
    y_pred1 = [1, 2, 1, 1, 3, 3, 3, 1]
    y_pred2 = [2, 2, 1, 1, 2, 3, 3, 1]
    y_pred3 = [3, 2, 1, 1, 2, 3, 2, 1]
    y_pred = [y_pred1, y_pred2, y_pred3]
    model_nm = ['model1', 'model2', 'model3']
    labels = [1, 2, 3]
    compare_report = MultiCompare(y_true, y_pred, model_nm, labels).compare_report()
    print('')
    print('模型比较结果：')
    print(compare_report)