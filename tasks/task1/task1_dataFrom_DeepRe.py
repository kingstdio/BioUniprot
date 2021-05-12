import numpy as np
import pandas as pd
import random
import time
import gzip
import re
from Bio import SeqIO
import datetime
import sys
from tqdm import tqdm
import os

from functools import reduce
import matplotlib.pyplot as plt

sys.path.append("./tools/")
import funclib

from pyecharts.globals import CurrentConfig, NotebookType
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB

from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.faker import Faker
from pyecharts.globals import ThemeType

from jax_unirep import get_reps

def split_train_test():
    f = open("./tasks/task1/data/deepre/new_data_label_sequence.txt")
    line = f.readline()

    counter = 0
    enzyme = []
    while line:
        seq = line.split('>')[1].replace('\n','')
        enzyme.append([seq,1])
        line = f.readline()
        counter +=1
    #     if counter ==5:
    #         break
    f.close()

    enzyme = pd.DataFrame(enzyme)
    enzyme.columns=['seq', 'label']
    enzyme['seqlength'] = enzyme['seq'].map(lambda x : len(x))


    f = open("./tasks/task1/data/deepre/non_enzyme_new_data_sequence.txt")
    line = f.readline()

    counter = 0
    non_enzyme = []
    while line:
        seq = line.split('>')[1].replace('\n','')
        non_enzyme.append([seq,0])
        line = f.readline()
        counter +=1
    #     if counter ==5:
    #         break
    f.close()

    non_enzyme = pd.DataFrame(non_enzyme)
    non_enzyme.columns=['seq', 'label']
    non_enzyme['seqlength'] = non_enzyme['seq'].map(lambda x : len(x))

    train = enzyme.iloc[:19951,]
    test = enzyme.iloc[19951:-1,]

    train = train.append(non_enzyme.iloc[:19951,])
    test = test.append(non_enzyme.iloc[19951:-1,])

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train['id'] = train.index
    test['id'] = test.index

    return train, test


def unirep(train, test):
    X_train = train['seq']
    Y_train = train.label.astype('int')

    X_test = test['seq']
    Y_test = test.label.astype('int')

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    X_train_res = []
    X_test_res =[]
    counter = 0
    step = 100
    for i in tqdm(range(0,len(X_train),step)):
        train_h_avg, train_h_final, train_c_final= get_reps(X_train[i:i+step])
        X_train_res.append(train_h_final)
        # counter += 1
        # if counter ==4:
        #     break

    for i in tqdm(range(0,len(X_test),step)):
        test_h_avg, test_h_final, test_c_final= get_reps(X_test[i:i+step])
        X_test_res.append(test_h_final)
        # counter += 1
        # # if counter ==8:
        # #     break     
    
    X_train_res = pd.DataFrame(np.array(X_train_res).reshape((-1,1900)))
    X_test_res = pd.DataFrame(np.array(X_test_res).reshape((-1,1900)))

    X_train_res['lablel'] = Y_train
    X_test_res['label'] = Y_test    
    return X_train_res, X_test_res


if __name__ =="__main__":
    file_train = r'./tasks/task1/data/deepre/train_1.tsv'
    file_test = r'./tasks/task1/data/deepre/test_1.tsv'

    train = pd.read_csv(file_train)

    # train,test = split_train_test()
    trainset = train[['id', 'label','seq', 'seqlength']].reset_index(drop=True)
    testset = test[['id', 'label','seq', 'seqlength']].reset_index(drop=True)
    X_train_res, X_test_res = unirep(train=trainset, test=testset)
    X_train_res.to_csv('./tasks/task1/data/deeretrain.tsv',sep='\t',header=0,index=0)
    X_test_res.to_csv('./tasks/task1/data/deepretest.tsv',sep='\t',header=0,index=0)