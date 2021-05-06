import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from xgboost.fs import featureSelection


def selectFeature(comment):
    sentence = []
    for w in str(comment).split(" "):
        if w in all_word:
            sentence.append(w)
    return ' '.join(sentence)


def process(comments, labels):
    des_com = []
    imp_com = []
    def_com = []
    des_lab = []
    imp_lab = []
    def_lab = []
    for i in range(len(comments)):
        # comments[i] = str(comments[i])
        comments[i] = selectFeature(comments[i])
        if labels[i] == 'DESIGN':
            des_com.append(comments[i])
            des_lab.append(0)
        elif labels[i] == 'IMPLEMENTATION':
            imp_com.append(comments[i])
            imp_lab.append(1)
        elif labels[i] == 'DEFECT':
            def_com.append(comments[i])
            def_lab.append(2)
    return des_com, imp_com, def_com, des_lab, imp_lab, def_lab


# 计算词频
# pro_id = 3
result = pd.DataFrame(
        columns=['name', 'prec', 'prec_1', 'prec_2', 'prec_3', 'recall', 'recall_1', 'recall_2', 'recall_3',
                 'f1', 'f1_1', 'f1_2', 'f1_3', 'macro-f1'])
for pro_id in range(10):
    rs=38
    ctv = CountVectorizer(min_df=1, max_df=0.5, ngram_range=(1, 2))
    df = pd.read_csv('./data/preprocess/' + str(pro_id) + '_train.csv', header=None)
    # 使用徐的方法扩充样本
    data3 = df[df[1] == 'DESIGN']
    data1 = pd.read_csv('./data_augmentation/' + str(pro_id) + '-defect.csv', header=None)
    data2 = pd.read_csv('./data_augmentation/' + str(pro_id) + '-implementation.csv', header=None)
    data = data1.append(data2, ignore_index=True)
    data = data.append(data3, ignore_index=True)
    labels = data[1]
    comments = data[2]
    all_word = featureSelection(comments, labels, 0.1)

    desgin_list, implementation_list, defect_list, des_label, imp_label, def_label = process(comments, labels)
    com_train = desgin_list + implementation_list + defect_list
    labels_train = des_label + imp_label + def_label

    data1 = pd.read_csv('./data/preprocess/' + str(pro_id) + '_test.csv',
                        header=None)
    labels1 = data1[1]
    comments1 = data1[2]
    desgin_list1, implementation_list1, defect_list1, des_label1, imp_label1, def_label1 = process(comments1,
                                                                                                   labels1)
    test = desgin_list1 + implementation_list1 + defect_list1
    labels_test = des_label1 + imp_label1 + def_label1

    # # 使用Count Vectorizer来fit训练集和测试集（半监督学习）
    ctv.fit(com_train + test)
    xtrain_ctv = ctv.transform(com_train)
    xvalid_ctv = ctv.transform(test)


    X_train, X_test, y_train, y_test = train_test_split(xtrain_ctv, labels_train, test_size=0.2,
                                                        random_state=40)  ##test_size测试集合所占比例

    clf = xgb.XGBClassifier(max_depth=6, n_estimators=1000, colsample_bytree=0.8, objective='multi:softmax',
                            subsample=0.8, nthread=10, learning_rate=0.06)
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=30,
            eval_metric=["mlogloss", "merror"])

    y_pre = clf.predict_proba(xvalid_ctv)
    y_pre = np.array(y_pre)
    y_pre = np.argmax(y_pre, axis=1)

    print(y_pre)
    pre_1 = precision_score(labels_test, y_pre, average='macro')
    pre_2 = precision_score(labels_test, y_pre, average=None)

    # Calculate recall score
    rec_1 = recall_score(labels_test, y_pre, average='macro')
    rec_2 = recall_score(labels_test, y_pre, average=None)

    # Calculate f1 score
    f1_1 = f1_score(labels_test, y_pre, average='macro')
    f1_2 = f1_score(labels_test, y_pre, average=None)
    macro_f1 = 2 * pre_1 * rec_1 / (pre_1 + rec_1)

    print('prec: {},{}\nrecall: {},{}\nf1: {},{}\nmacro-f1: {}\n'.format(pre_1, pre_2, rec_1, rec_2, f1_1, f1_2,
                                                                         macro_f1))








