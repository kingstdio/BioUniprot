import numpy as np
import pandas as pd
import random
import time
import re
from Bio import SeqIO
import datetime
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append("../../tools/")
# import funclib
from jax_unirep import get_reps


def loaddata(path):
    table_head = [
                'id', 
                'name',
                'isemzyme',
                'isMultiFunctional', 
                'functionCounts', 
                'ec_number', 
                'ec_specific_level',
                'date_integraged',
                'date_sequence_update',
                'date_annotation_update',
                'seq', 
                'seqlength'
            ]
    enzyme_noemzyme = pd.read_csv(path, sep='\t',names=table_head) #读入文件
    enzyme_noemzyme = enzyme_noemzyme.reset_index(drop=True)
    return enzyme_noemzyme

def getunirep(enzyme_noemzyme, step):
    unirep_res = []
    counter = 1
    for i in tqdm(range(0, len(enzyme_noemzyme), step)):
        train_h_avg, train_h_final, train_c_final= get_reps(list(enzyme_noemzyme.seq[i:i+step]))
        checkpoint = np.hstack((np.array(enzyme_noemzyme[i:i+step]),train_h_final))
        unirep_res.append(checkpoint)
        np.save(r'/tmp/task_unirep_'+str(counter)+'.tsv', checkpoint)
        if len(train_h_final) != step:
            print('length not match')
        counter += 1
    unirep_res = np.array(unirep_res, dtype=object)
    return unirep_res

def rununi_baselines(): 
    uni_data = np.load(sys.path[0] +r'/data/emzyme_noemzyme_unirep.npy', allow_pickle=True)
    uni_data =pd.DataFrame([i for k in uni_data for i in k ])
    uni_data[7] = pd.to_datetime(uni_data[7])

    t_thres = datetime.datetime(2009, 12, 14, 0, 0)

    #训练集
    train = uni_data[uni_data[7] <= t_thres ].sort_values(by=7)
    #测试集
    test = uni_data[uni_data[7] > t_thres ].sort_values(by=7)

    X_train = train.iloc[:,12:-1]
    X_test = test.iloc[:,12:-1]

    Y_train = train.iloc[:,2:3].astype('int')
    Y_test = test.iloc[:,2:3].astype('int')

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train).flatten()
    Y_test = np.array(Y_test).flatten()

    # funclib.evaluate(X_train, Y_train, X_test, Y_test)

if __name__ == '__main__':
    data_path = sys.path[0] +r'/data/emzyme_noemzyme_data_supply.tsv'
    temp_path = sys.path[0] +r'/data/'
    STEP = 200
    ori_data = loaddata(data_path)
    res = getunirep(ori_data, STEP)

    
    np.save(sys.path[0] +r'/data/emzyme_noemzyme_unirep_supply.npy',res)

    # rununi_baselines()
    