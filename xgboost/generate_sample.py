from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import random


# from multiprocessing import Pool, Lock, Manager
# import os, pickles
import pandas as pd


def rand_text(l):
    tmp = l.split(' ')
    random.shuffle(tmp)
    return ' '.join(tmp)


def create(df, n, label_name):
    df = df.reset_index(drop=True)
    length = len(df)
    for i in range(length, n, 1):
        if i % 2 == 0:
            s = df.loc[i % length][2]
            s = rand_text(s)
            df.loc[i] = ['create', label_name, s]
        else:
            rand1, rand2 = int(random.random() * length), int(random.random() * length)
            str1 = df.loc[rand1][2]
            str2 = df.loc[rand2][2]
            str_l1 = str1.split(' ')
            str_l2 = str2.split(' ')
            str_l = str_l1[:int(len(str_l1) / 2)]
            str_l.extend(str_l2[int(len(str_l2) / 2):])
            s = ' '.join(str_l).strip()
            df.loc[i] = ['create', label_name, s]
    return df


def calc_jd(design_df, defect_df, implemntation_df):
    cls1 = len(design_df)
    cls2 = len(defect_df)
    cls3 = len(implemntation_df)
    all_df = design_df.append(defect_df).append(implemntation_df)
    all_list = all_df[2].tolist()

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()

    X = vectorizer.fit_transform(all_list)

    tfidf = transformer.fit_transform(X).todense()

    print('start')
    ans = np.array(tfidf)
    (_, dim) = ans.shape

    # calculate Sb
    ans_1 = ans[0:cls1, :]
    ans_2 = ans[cls1:cls1 + cls2, :]
    ans_3 = ans[cls1 + cls2:cls1 + cls2 + cls3, :]
    m = np.mean(ans, axis=0)
    m1 = np.mean(ans_1, axis=0)
    m2 = np.mean(ans_2, axis=0)
    m3 = np.mean(ans_3, axis=0)
    m = m.reshape(dim, 1)
    m1 = m1.reshape(dim, 1)
    m2 = m2.reshape(dim, 1)
    m3 = m3.reshape(dim, 1)
    Sb = ((m1 - m).dot((m1 - m).T) + (m2 - m).dot((m2 - m).T) + (m3 - m).dot((m3 - m).T)) / 3


    J1 = np.trace(Sb)

    Jd = J1
    return Jd


def jd_create(df, defect_n=1000, implemntation_n=1000):
    df = df[df[2] != '']
    desgin_df = df[df[1] == 'DESIGN']
    desgin_df = desgin_df.reset_index(drop=True)
    defect_df = df[df[1] == 'DEFECT']
    defect_df = defect_df.reset_index(drop=True)
    implemntation_df = df[df[1] == 'IMPLEMENTATION']
    implemntation_df = implemntation_df.reset_index(drop=True)
    best_jd = 0.0
    best_defect_df = None
    best_implemntation_df = None
    for i in range(50):
        tmp_defect_df = create(defect_df, defect_n, 'DEFECT')
        tmp_implemntation_df = create(implemntation_df, implemntation_n, 'IMPLEMENTATION')
        jd = calc_jd(desgin_df, tmp_defect_df, tmp_implemntation_df)
        if (best_jd < jd):
            best_jd = jd
            best_defect_df = tmp_defect_df
            best_implemntation_df = tmp_implemntation_df
        print('best_jd:{}'.format(best_jd))
    return best_defect_df, best_implemntation_df




for pro_id in range(10):
    df = pd.read_csv('./data/preprocess/' + str(pro_id) + '_train.csv', header=None)

    best_design_df, best_implemntation_df = jd_create(df)
    best_design_df.to_csv('./data_augmentation/pro_' + str(pro_id) + '-defect' + '.csv', header=None,
                          index=None)
    best_implemntation_df.to_csv('./data_augmentation/pro_' + str(pro_id) + '-implementations' + '.csv',
                                 header=None, index=None)

