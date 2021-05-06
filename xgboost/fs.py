# # -*- coding: utf-8 -*-
import numpy as np
import pandas as pd



def get_term_dict(doc_terms_list):
    term_set_dict = {}
    for doc_terms in doc_terms_list:
        for term in doc_terms:
            term_set_dict[term] = 1
    term_set_list = sorted(term_set_dict.keys())
    term_set_dict = dict(zip(term_set_list, range(len(term_set_list))))
    return term_set_dict


def get_class_dict(doc_class_list):
    class_set = sorted(list(set(doc_class_list)))
    class_dict = dict(zip(class_set, range(len(class_set))))
    return class_dict


def stats_term_df(doc_terms_list, term_dict, term_set):
    term_df_dict = {}.fromkeys(term_dict.keys(), 0)
    for term in term_set:
        for doc_terms in doc_terms_list:
            if term in doc_terms:
                term_df_dict[term] += 1
    return term_df_dict


def stats_class_df(doc_class_list, class_dict):
    class_df_list = [0] * len(class_dict)
    for doc_class in doc_class_list:
        class_df_list[class_dict[doc_class]] += 1
    return class_df_list


def stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict):
    term_class_df_mat = np.zeros((len(term_dict), len(class_dict)), np.float32)
    for k in range(len(doc_class_list)):
        class_index = class_dict[doc_class_list[k]]
        doc_terms = doc_terms_list[k]
        for term in set(doc_terms):
            term_index = term_dict[term]
            term_class_df_mat[term_index][class_index] += 1
    return term_class_df_mat


def feature_selection_chi_mi(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    D = N - A - B - C
    class_set_size = len(class_df_list)
    term_score_mat_chi =N*((A * D - B * C) * (A * D - B * C)) / ((A + B) * (C + D))
    term_score_mat_mi = (A / N) * (np.log(((A + 1.0) * N) / ((A + C) * (A + B))))
    E = np.sum(A, axis=1)
    for x in range(len(A)):
        for y in range(len(A[x])):
            A[x][y] /= E[x]
    term_score_mat = term_score_mat_chi

    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)
    sorted_term_score_index = term_score_array.argsort()[:: -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index]
    return term_set_fs


def feature_selection(doc_terms_list, doc_class_list, fs_method):
    class_dict = get_class_dict(doc_class_list)
    term_dict = get_term_dict(doc_terms_list)
    class_df_list = stats_class_df(doc_class_list, class_dict)
    term_class_df_mat = stats_term_class_df(doc_terms_list, doc_class_list, term_dict,
                                            class_dict)

    term_set = [term[0] for term in sorted(term_dict.items(), key=lambda x: x[1])]
    print("before selection length %s" % (len(term_set)))

    term_set_fs = []

    if fs_method == 'CHMI':
        term_set_fs = feature_selection_chi_mi(class_df_list, term_set, term_class_df_mat)
    return term_set_fs


def selectWord(wordlist, percentage):
    sub_wordLst = wordlist[:int(percentage * len(wordlist))]
    return sub_wordLst


def featureSelection(comments, label, percentage):
    word_set = set()
    text_list = comments
    label_list = label
    doc_term_list = []
    cls_list = []
    for i in range(len(text_list)):
        if label_list[i] == 'DESIGN':
            doc_term_list.append(str(text_list[i]).split(" "))
            cls_list.append(0)
        elif label_list[i] == 'IMPLEMENTATION':
            doc_term_list.append(str(text_list[i]).split(" "))
            cls_list.append(1)
        elif label_list[i] == 'DEFECT':
            doc_term_list.append(str(text_list[i]).split(" "))
            cls_list.append(2)
    doc_term_list = np.array(doc_term_list)
    doc_class_list = cls_list
    fs_method = 'CHMI'
    result = feature_selection(doc_term_list, doc_class_list, fs_method)
    selectResult = selectWord(result, percentage)
    return selectResult


