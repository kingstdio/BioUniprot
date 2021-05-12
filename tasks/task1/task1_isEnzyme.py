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

    table_head = [  'id', 
                'isemzyme',
                'isMultiFunctional', 
                'functionCounts', 
                'ec_number', 
                'date_integraged',
                'date_sequence_update',
                'date_annotation_update',
                'seq', 
                'seqlength'
            ]

    #加载数据并转换时间格式
    sprot = pd.read_csv('./data/sprot_full.tsv', sep='\t',names=table_head) #读入文件
    sprot.date_integraged = pd.to_datetime(sprot['date_integraged'])
    sprot.date_sequence_update = pd.to_datetime(sprot['date_sequence_update'])
    sprot.date_annotation_update = pd.to_datetime(sprot['date_annotation_update'])


    t_thres = datetime.datetime(2010, 2, 10, 0, 0)

    #训练集
    train = sprot[sprot.date_integraged <= t_thres ].sort_values(by='date_integraged')
    #测试集
    test = sprot[sprot.date_integraged > t_thres ].sort_values(by='date_integraged')

    return train, test


def unirep(train, test):

    X_train = trainset['seq']
    Y_train = trainset.isemzyme.astype('int')

    X_test = testset['seq']
    Y_test = testset.isemzyme.astype('int')

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    X_train_res = []
    X_test_res =[]
    counter = 0
    step = 50
    for i in tqdm(range(0,len(X_train),step)):
        train_h_avg, train_h_final, train_c_final= get_reps(X_train[i:i+step])
        X_train_res.append(train_h_final)
        counter += 1
        if counter ==2:
            break

    for i in tqdm(range(0,len(X_test),step)):
        test_h_avg, test_h_final, test_c_final= get_reps(X_test[i:i+step])
        X_test_res.append(test_h_final)
        counter += 1
        # if counter ==4:
        #     break     
    
    X_train_res = pd.DataFrame(np.array(X_train_res, dtype=object).reshape((-1,1900)))
    X_test_res = pd.DataFrame(np.array(X_test_res, dtype=object).reshape((-1,1900)))

    X_train_res['lablel'] = Y_train
    X_test_res['label'] = Y_test    
    return X_train_res, X_test_res


if __name__ =="__main__":
    train,test = split_train_test()
    trainset = train[['id', 'label','seq', 'seqlength']].reset_index(drop=True)
    testset = test[['id', 'label','seq', 'seqlength']].reset_index(drop=True)
    X_train_res, X_test_res = unirep(train=trainset, test=testset)
    X_train_res.to_csv('./tasks/task1/data/uni_train.tsv',sep='\t',header=0,index=0)
    X_test_res.to_csv('./tasks/task1/data/uni_test.tsv',sep='\t',header=0,index=0)