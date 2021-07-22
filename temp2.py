import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
import tools.ucTools as ucTools
import tools.funclib as funclib



if __name__ =='__main__':
    snap2018 = pd.read_csv('./data/201802/sprot_full.tsv', sep='\t', header=None, names=funclib.table_head)
    snap2018.drop_duplicates(subset=['seq'], keep='first', inplace=True)
    snap2018.reset_index(drop=True, inplace=True)

    # 2020 快照
    snap2020 = pd.read_csv('./data/202006/sprot_full.tsv', sep='\t', header=None, names=funclib.table_head)
    snap2020.drop_duplicates(subset=['seq'], keep='first', inplace=True)
    snap2020.reset_index(drop=True, inplace=True)

    # snap2018_fuc_split = funclib.split_ecdf_to_single_lines(snap2018)
    # snap2018_fuc_split.to_csv('./data/201802/snap2018_siglelineec.tsv', sep='\t', index=None)

    snap2020_fuc_split = funclib.split_ecdf_to_single_lines(snap2020)
    snap2020_fuc_split.to_csv('./data/202006/snap2020_siglelineec.tsv', sep='\t', index=None)


    print('finished')