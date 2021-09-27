import numpy as np
import pandas as pd
import sys
import os
from tqdm import tqdm
import tools.ucTools as ucTools
import tools.funclib as funclib



if __name__ =='__main__':
    # snap2018 = pd.read_csv('./data/201802/sprot_full.tsv', sep='\t', header=None, names=funclib.table_head)
    # snap2018.drop_duplicates(subset=['seq'], keep='first', inplace=True)
    # snap2018.reset_index(drop=True, inplace=True)

    # # 2020 快照
    # snap2020 = pd.read_csv('./data/202006/sprot_full.tsv', sep='\t', header=None, names=funclib.table_head)
    # snap2020.drop_duplicates(subset=['seq'], keep='first', inplace=True)
    # snap2020.reset_index(drop=True, inplace=True)

    # # snap2018_fuc_split = funclib.split_ecdf_to_single_lines(snap2018)
    # # snap2018_fuc_split.to_csv('./data/201802/snap2018_siglelineec.tsv', sep='\t', index=None)

    # snap2020_fuc_split = funclib.split_ecdf_to_single_lines(snap2020)
    # snap2020_fuc_split.to_csv('./data/202006/snap2020_siglelineec.tsv', sep='\t', index=None)
    table_head = [  'id', 
                'isenzyme',
                'isMultiFunctional', 
                'functionCounts', 
                'ec_number', 
                'ec_specific_level',
                'date_integraged'
            ]
    f = open('/home/shizhenkun/codebase/BioUniprot/data/trembl/uniprot_trembl.tsv', 'r')
    line = f.readline()

    counter =0

    file_write_obj = open(r'/home/shizhenkun/codebase/BioUniprot/data/trembl/uniprot_trembl_simple.tsv','w')
    file_write_obj.writelines('\t'.join(table_head))
    file_write_obj.writelines('\n')

    while line:
        
        lineArray = line.split('\t')
        wline  = lineArray[0] + '\t' + lineArray[2]+ '\t' + lineArray[3]+ '\t' + lineArray[4]+ '\t' + lineArray[5]+ '\t' + lineArray[6]+ '\t' + lineArray[7]
        file_write_obj.writelines(wline)
        file_write_obj.writelines('\n')
        line = f.readline()
        
        counter +=1
        if counter %1000000==0:
            file_write_obj.flush()
            print(counter)
            
    file_write_obj.close()


    print('finished')