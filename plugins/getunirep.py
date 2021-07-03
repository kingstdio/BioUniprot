import numpy as np
from numpy.core.defchararray import index
import pandas as pd
import random
import os
from tqdm import tqdm
from jax_unirep import get_reps


def getunirep(enzyme_noemzyme, step):
    unirep_res = []
    counter = 1
    for i in tqdm(range(0, len(enzyme_noemzyme), step)):
        train_h_avg, train_h_final, train_c_final= get_reps(list(enzyme_noemzyme.seq[i:i+step]))
        checkpoint = np.hstack((np.array(enzyme_noemzyme[i:i+step]),train_h_final))
        
        if counter == 1:
            unirep_res = np.array(checkpoint)
        else:
            unirep_res = np.concatenate((unirep_res,checkpoint))

        np.save(r'/tmp/task_unirep_'+str(counter)+'.tsv', checkpoint)

        if len(train_h_final) != step:
            print('length not match')
        counter += 1
    
    return unirep_res


def load_file(file):
    return pd.read_csv(file,sep='\t')

def save_file(file, data):
    data = pd.DataFrame(data, columns=['id', 'seq'] + ['f'+str(i) for i in range(1, 1901) ])
    h5 = pd.HDFStore(file, 'w', complevel=4, complib='blosc')
    h5['data'] = data
    h5.close()

if __name__ == '__main__':
    infile = os.getcwd() + '/data/needUnirep.tsv'
    outfile = os.getcwd() + '/data/needUnirep_results.h5'
    seq = load_file(infile)
    STEP= 400
    res = getunirep(seq, STEP)
    save_file(outfile, res)
    print('unirep finish')