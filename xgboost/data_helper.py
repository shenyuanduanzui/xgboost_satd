# created by wl
# 一 category
import pandas as pd
import os
import re
import csv
from nltk.tokenize import word_tokenize

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import  PorterStemmer

pst = PorterStemmer()
lst = LancasterStemmer()
stem_wordnet = WordNetLemmatizer()


def clean_str(string):
    pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  # 删除链接
    string = pattern.sub(' ', string)
    # clean logs
    string = re.sub(r'\S+-\S+-\S+\s:\s+(\S|\s)+', '', string)
    string = re.sub(r'Contributor\(s\)(.+\n)+', '', string)
    string = re.sub(r'\*|/|=', '', string)
    string = re.sub(r'-', ' ', string)

    string = re.sub(r"<TABLE .*?>((.|\n)*?)</TABLE>", " ", string)
    string = re.sub(r"<table .*?>((.|\n)*?)</table>", " ", string)
    # clean html labels
    string=re.sub('<[^>]*>','', string)

    string = re.sub(r"[^A-Za-z(),\+!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)  #
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", "  ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\+", " ", string)
    string = re.sub(r"\"", " ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def stem(string):
    words = word_tokenize(string)
    text = []
    for word in words:
        if word == 'ca':
            text.append('can')
        elif word == 'n\'t':
            text.append('not')
        elif word == 'wo':
            text.append('will')
        else:
            text.append(stem_wordnet.lemmatize(word))
    text = [pst.stem(w) for w in text]
    text = [lst.stem(w) for w in text]

    text = ' '.join(text)
    return text


df = pd.read_csv('data/technical_debt_dataset.csv', header=None)

df = df.drop_duplicates()

df.to_csv('./data/allSamples.csv', header=None)

projects = ['apache-ant-1.7.0', 'apache-jmeter-2.10', 'argouml', 'columba-1.4-src',
            'emf-2.4.1', 'hibernate-distribution-3.3.2.GA', 'jEdit-4.2', 'jfreechart-1.0.19',
            'jruby-1.4.0', 'sql12']


for i in range(10):
    data = pd.read_csv('data/allSamples.csv', header=None, encoding="utf-8")
    file0 = open('./data/preprocess/' + str(i) + '_train.csv', 'w', newline='', encoding='utf-8')
    csvfile0 = csv.writer(file0)
    file1 = open('./data/preprocess/' + str(i) + '_test.csv', 'w', newline='', encoding='utf-8')
    csvfile1 = csv.writer(file1)
    for j in range(len(data[2])):
        if data[1][j] == projects[i]:
            sentence = clean_str(str(data[3][j]))
            sentence = stem(sentence)
            if sentence != '':
                csvfile1.writerow([data[1][j], data[2][j], sentence])
        else:
            sentence = clean_str(str(data[3][j]))
            sentence = stem(sentence)
            if sentence != '':
                csvfile0.writerow([data[1][j], data[2][j], sentence])

    file0.close()
    file1.close()
