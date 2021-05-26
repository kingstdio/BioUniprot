import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import os
from jax_unirep import get_reps
import sys

def unirep(data, filein):

    X= data['seq']
    Y = data.label.astype('int')
    X = np.array(X)
    Y = np.array(Y)

    X_res = []
    counter = 0
    step = 100
    for i in tqdm(range(0,len(X),step)):
        train_h_avg, train_h_final, train_c_final= get_reps(X[i:i+step])
        X_res.append(train_h_final)

    X_res = pd.DataFrame(np.array(X_res, dtype=object)[0].reshape((-1,1900)))
    X_res['lablel'] = Y[i]
    return X_res

def run_onece(filein, fileout):
    data = pd.read_csv(filein, header=0, sep='\t')
    res = unirep(data, filein)
    res.to_csv(fileout,sep='\t',header=1,index=0)

def run_main(dirin, dirout):

    files= os.listdir(dirin) #得到文件夹下的所有文件名称
    exist = os.listdir(dirout)
    files = list(set(files).difference(set(exist)))
    files = [s for s in files if '.tsv' in s]


    counter =1
    for row in files:
        print('iter {}-{}'.format(counter, row))
        filein = os.getcwd() +r'/' +dirin + row
        fileout = os.getcwd() +r'/'+ dirout + row
        run_onece(filein,fileout)
        time.sleep(2)
        
    print('finished')

if __name__ =="__main__":
    dir_in  = r"./tasks/task1/data/deepre/"
    dir_out = r"./tasks/task1/data/deepre/res/"
    file  = sys.argv[0]
    print(file)
    run_main(dir_in, dir_out)
