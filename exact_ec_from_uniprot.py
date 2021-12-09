from Bio import SeqIO
import gzip
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import time
import random
import tools.funclib as funclib

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
                    'isenzyme',
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
    file_write_obj = open(file_out_path,'w')
    #reslist.append(['id', 'description', 'seq'])
    with gzip.open(file_in_path, "rt") as handle:
        file_write_obj.writelines('\t'.join(table_head))
        file_write_obj.writelines('\n')
        for record in tqdm( SeqIO.parse(handle, 'swiss')):
            res = process_record(record, extract_type= extract_type)
            counter+=1
            if counter %10==0:
                file_write_obj.flush()
            if saver%100000==0:
                print(saver)
            if len(res) >0:
                saver +=1
                file_write_obj.writelines('\t'.join(map(str,res)))
                file_write_obj.writelines('\n')
            else:
                continue

    file_write_obj.close()
    # result = pd.DataFrame(reslist, columns= table_head)
    # result.date_integraged = pd.to_datetime(result['date_integraged'])
    # result.date_sequence_update = pd.to_datetime(result['date_sequence_update'])
    # result.date_annotation_update = pd.to_datetime(result['date_annotation_update'])
    # result.sort_values(by='date_integraged', inplace=True)
    # result.to_csv(file_out_path, sep="\t", index=0, header=0 )
 
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
    isEnzyme = 'EC=' in description     #有EC号的被认为是酶，否则认为是酶
    isMultiFunctional = False
    functionCounts = 0
    ec_specific_level =0


    if isEnzyme:
        ec = str(re.findall(r"EC=[0-9,.\-;]*",description)
                 ).replace('EC=','').replace('\'','').replace(']','').replace('[','').replace(';','').strip()

        #统计酶的功能数
        isMultiFunctional = ',' in ec
        functionCounts = ec.count(',') + 1


        # - 单功能酶
        if not isMultiFunctional:
            levelCount = ec.count('-')
            ec_specific_level = 4-levelCount

        else: # -多功能酶
            ecarray = ec.split(',')
            for subec in ecarray:
                current_ec_level = 4- subec.count('-')
                if ec_specific_level < current_ec_level:
                    ec_specific_level = current_ec_level
    else:
        ec = '-'
    
    id = record.id.strip()
    name = record.name.strip()
    seq = record.seq.strip()
    date_integrated = record.annotations.get('date').strip()
    date_sequence_update = record.annotations.get('date_last_sequence_update').strip()
    date_annotation_update = record.annotations.get('date_last_annotation_update').strip()
    seqlength = len(seq)
    res = [id, name, isEnzyme, isMultiFunctional, functionCounts, ec,ec_specific_level, date_integrated, date_sequence_update, date_annotation_update,  seq, seqlength]

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



# 将给定的数据随机划分为2份
def split_random(data):
    index_ref = random.sample(range(0,len(data)), int(len(data)/2))  #创建随机index
    ref = data.iloc[index_ref]     #筛选ref
    query = data.iloc[~data.index.isin(index_ref)] # 剩下的是Query
    funclib.table_2_fasta(ref, './data/sprot_with_ec_ref.fasta')   #写入文件ref fasta
    funclib.table_2_fasta(query, './data/sprot_with_ec_query.fasta') #写入文件query fasta
    return query, ref

# 按时序前一半后一般划分
def split_time_half(data):
    index_ref = range(int(len(data)/2))
    ref = data.iloc[index_ref]     #筛选ref
    query = data.iloc[~data.index.isin(index_ref)] # 剩下的是Query
    funclib.table_2_fasta(ref, './data/sprot_with_ec_ref.fasta')   #写入文件ref fasta
    funclib.table_2_fasta(query, './data/sprot_with_ec_query.fasta') #写入文件query fasta


if __name__ =="__main__":
    start =  time.process_time()
    # in_filepath = r'/home/shizhenkun/codebase/BioUniprot/data/201802/uniprot_sprot.dat.gz'
    # out_filepath = r'/home/shizhenkun/codebase/BioUniprot/data/201802/sprot_full.tsv'

    in_filepath = r'/public/data/uniprot/uniprot_trembl.dat.gz'
    out_filepath = r'/home/shizhenkun/codebase/BioUniprot/data/trembl/uniprot_trembl.tsv'
    extract_type ='full'
    read_file_from_gzip(file_in_path=in_filepath, file_out_path=out_filepath, extract_type=extract_type)
    end =  time.process_time()
    print('finished use time %6.3f s' % (end - start))

   