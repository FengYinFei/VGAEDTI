
from sklearn.ensemble import RandomForestClassifier

import numpy as np
from utils import load_data
from fivefoldcv import GNNq, GNNp

class randomforest():
 def randomforest(self):
    mtrain, dtrain, label = load_data()

    encoder, m_data1 = GNNq(mtrain)
    encoder, d_data1 = GNNp(dtrain)
    num_cross = 5
    probaresult = []
    ae_y_pred_probresult = []

    for fold in range(num_cross):
        train_m = np.array([x for i, x in enumerate(m_data1) if i % num_cross != fold])
        test_m = np.array([x for i, x in enumerate(m_data1) if i % num_cross == fold])
        train_d = np.array([x for i, x in enumerate(d_data1) if i % num_cross != fold])
        test_d = np.array([x for i, x in enumerate(d_data1) if i % num_cross == fold])
        train_label = np.array([x for i, x in enumerate(m_data1) if i % num_cross != fold])

        train_label_new = []

        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)

        prefilter_mtrain = train_m
        prefilter_mtest = test_m
        prefilter_dtrain = train_d
        prefilter_dtest = test_d

        # 随机森林分类器
        # drug
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(prefilter_mtrain, train_label_new)# clf.fit用训练数据拟合分类器模型
        # clf.predict_proba返回预测属于某标签的概率,是取二维数组中第二维的所有数据
        mae_y_pred_prob = clf.predict_proba(prefilter_mtest)[:, 1]
        # 调用下面的函数如果大于0.5为1，否则0
        mproba = transfer_label_from_prob(mae_y_pred_prob)
        # disease
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(prefilter_dtrain, train_label_new)
        dae_y_pred_prob = clf.predict_proba(prefilter_dtest)[:, 1]
        dproba = transfer_label_from_prob(dae_y_pred_prob)

        mproba = np.array(mproba)
        dproba = np.array(dproba)

        proba = (mproba + dproba)/2
        ae_y_pred_prob = (mae_y_pred_prob + dae_y_pred_prob)/2

        probaresult.extend(proba)
        ae_y_pred_probresult.extend(ae_y_pred_prob)
    return probaresult, ae_y_pred_probresult, m_data1


def transfer_label_from_prob(proba):
    # 将像素指定为0和1
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label
