from Bio import SeqIO
import gzip
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import time
import subprocess


#region 从gizp读取数据
def read_file_from_gzip(file_in_path, file_out_path, extract_type):
    """从原始Zip file 中解析数据

    Args:
        file_in_path ([string]): [输入文件路径]
        file_out_path ([string]): [输出文件路径]]
        extract_type ([string]): [抽取数据的类型：with_ec, without_ec, full]]
    """
    table_head = [  'id', 
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
    counter = 0
    saver = 0 
    reslist = []
    #reslist.append(['id', 'description', 'seq'])
    with gzip.open(file_in_path, "rt") as handle:
        
        for record in tqdm( SeqIO.parse(handle, 'swiss')):
            res = process_record(record, extract_type= extract_type)
            counter+=1
            # if counter %10000==0:
            #     print('lines:{0}/{1}'.format(saver, counter))
            if len(res) >0:
                reslist.append(res)
                saver +=1
            else:
                continue

    result = pd.DataFrame(reslist, columns= table_head)
    result.date_integraged = pd.to_datetime(result['date_integraged'])
    result.date_sequence_update = pd.to_datetime(result['date_sequence_update'])
    result.date_annotation_update = pd.to_datetime(result['date_annotation_update'])

    result.sort_values(by='date_integraged', inplace=True)

    result.to_csv(file_out_path, sep="\t", index=0, header=0 )
 
 #endregion

#region 提取单条含有EC号的数据
def process_record(record, extract_type='with_ec'):
    """
    提取单条含有EC号的数据
    Args:
        record ([type]): uniprot 中的记录节点
        extract_type (string, optional): 提取的类型，可选值：with_ec, without_ec, full。默认为有EC号（with_ec).
    Returns:
        [type]: [description]
    """

    description = record.description
    isEnzyme = 'EC=' in description
    isMultiFunctional = False
    functionCounts = 1
    ec_specific_level =0 
    if isEnzyme:
        ec = str(re.findall(r"EC=[0-9,.\-;]*",description)).replace('EC=','').replace('\'','').replace(']','').replace('[','').replace(';','')
        isMultiFunctional = ',' in ec
        functionCounts = ec.count(',') + 1

        #大于三级EC号的被认为是酶，小于三级的不认为是酶
        # - 单功能酶
        if not isMultiFunctional:
            levelCount = ec.count('-')
            ec_specific_level = 4-levelCount
            if levelCount>1:
                isEnzyme = False
        else: # -多功能酶
            ecarray = ec.split(',')
            for subec in ecarray:
                if subec.count('-') ==0:
                    isEnzyme = True
                    ec_specific_level = 4
                    break
                elif subec.count('-') ==1:
                    isEnzyme = True
                    ec_specific_level = 3
                    break
                else:
                    isEnzyme=False
                    
                    if (4- subec.count('-')) > ec_specific_level:
                        ec_specific_level = (4- subec.count('-'))
                    

    else:
        ec = '-'
    
    id = record.id
    name = record.name
    seq = record.seq
    date_integraged = record.annotations.get('date')
    date_sequence_update = record.annotations.get('date_last_sequence_update')
    date_annotation_update = record.annotations.get('date_last_annotation_update')
    seqlength = len(seq)   
    res = [id, name, isEnzyme, isMultiFunctional, functionCounts, ec,ec_specific_level, date_integraged, date_sequence_update, date_annotation_update,  seq, seqlength]

    if extract_type == 'full':
        return res

    if extract_type == 'with_ec':
        if isEnzyme:
            return res
        else:
            return []

    if extract_type == 'without_ec':
        if isEnzyme:
            return []
        else:
            return res
#endregion

#将表格存储为fasta文件    
def table_2_fasta(table, file_out):
    file = open(file_out, 'w')
    for index, row in table.iterrows():
        file.write('>{0}\n'.format(row['id']))
        file.write('{0}\n'.format(row['seq']))
    file.close()
    print('Write finished')

# 将给定的数据随机划分为2份
def split_random(data):
    index_ref = random.sample(range(0,len(data)), int(len(data)/2))  #创建随机index
    ref = data.iloc[index_ref]     #筛选ref
    query = data.iloc[~data.index.isin(index_ref)] # 剩下的是Query
    table_2_fasta(ref, './data/sprot_with_ec_ref.fasta')   #写入文件ref fasta
    table_2_fasta(query, './data/sprot_with_ec_query.fasta') #写入文件query fasta
    return query, ref

# 按时序前一半后一般划分
def split_time_half(data):
    index_ref = range(int(len(data)/2))
    ref = data.iloc[index_ref]     #筛选ref
    query = data.iloc[~data.index.isin(index_ref)] # 剩下的是Query
    table_2_fasta(ref, './data/sprot_with_ec_ref.fasta')   #写入文件ref fasta
    table_2_fasta(query, './data/sprot_with_ec_query.fasta') #写入文件query fasta

if __name__ =="__main__":
    start =  time.process_time()
    in_filepath = r'./data/uniprot_sprot.dat.gz'
    out_filepath = r'./data/sprot_full.tsv'
    extract_type ='full'
    read_file_from_gzip(file_in_path=in_filepath, file_out_path=out_filepath, extract_type=extract_type)
    end =  time.process_time()
    print('finished use time %6.3f s' % (end - start))

   