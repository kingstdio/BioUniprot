from baselines.deepec.deepec.ec_prediction_dl import preprocessing
import pandas as pd
import numpy as np
import joblib
from tools.ucTools import ucTools
from tqdm import tqdm
from jax_unirep import get_reps
from xgboost import XGBClassifier, data
import os
import subprocess

#region 需要写fasta的dataFame格式
def save_table2fasta(dataset, file_out):
    """[summary]
    Args:
        dataset ([DataFrame]): 需要写fasta的dataFame格式[id, seq]
        file_out ([type]): [description]
    """
    if ~os.path.exists(file_out):
        file = open(file_out, 'w')
        for index, row in dataset.iterrows():
            file.write('>{0}\n'.format(row['id']))
            file.write('{0}\n'.format(row['seq']))
        file.close()
    print('Write finished')
#endregion

#region 获取序列比对结果
def getblast(refdata, querydata, blastres_file):
    """[获取序列比对结果]
    Args:
        refdata ([string]]): [训练集fasta文件]
        querydata ([DataFrame]): [测试数据集]
        blastres_file ([string]): [要保存的结果文件]

    Returns:
        [DataFrame]: [比对结果]
    """
    res_table_head = ['id', 'sseqid', 'pident', 'length','mismatch','gapopen','qstart','qend','sstart','send','evalue','bitscore']

    if os.path.exists(blastres_file):
        res_data = pd.read_csv(blastres_file, sep='\t', names=res_table_head)
        return res_data
    
    save_table2fasta(querydata, '/tmp/test.fasta')   
    cmd1 = r'diamond makedb --in {0} -d /tmp/train.dmnd'.format(refdata)
    cmd2 = r'diamond blastp -d /tmp/train.dmnd  -q  /tmp/test.fasta -o {0} -b5 -c1 -k 1'.format(blastres_file)
    cmd3 = r'rm -rf /tmp/*.fasta /tmp/*.dmnd /tmp/*.tsv'
    print(cmd1)
    os.system(cmd1)
    print(cmd2)
    os.system(cmd2)
    res_data = pd.read_csv(blastres_file, sep='\t', names=res_table_head)
    os.system(cmd3)
    return res_data

#endregion

#region 获取「酶｜非酶」预测结果
def get_isEnzymeRes(querydata, model_file):
    """[获取「酶｜非酶」预测结果]
    Args:
        querydata ([DataFrame]): [需要预测的数据]
        model_file ([string]): [模型文件]
    Returns:
        [DataFrame]: [预测结果、预测概率]
    """
    model = joblib.load(model_file)
    predict = model.predict(querydata)
    predictprob = model.predict_proba(querydata)
    return predict, predictprob[:,1]
#endregion 获取「酶｜非酶」预测结果

def get_slice_res(query_data, model_path):
    

if __name__ == '__main__':

    # 1. 加载序列序列文件
    file_1515 = './tasks/task3/data/ecoli1515/1515testset_with_unirep.tsv'
    data_1515 = pd.read_csv(file_1515, sep='\t', index_col=0)
    colname =  [['id', 'ec_number', 'seq', 'isemzyme', 'isMultiFunctional', 'blattner_id', 'blattner_ec_number' ] +['f'+str(i) for i in range(1,1901)]]
    data_1515.columns =colname

    file_ref = './data/preprocess/enzyme_train.fasta'
    blast_res_file = './results/1515/balastres.tsv'

    # 2. 获取序列比对结果
    blast_res = getblast(querydata=data_1515, refdata=file_ref, blastres_file=blast_res_file)

    # 3.进行序列比对
    isEnzyme_model = './model/isenzyme.model'
    isEnzyme_pred, isEnzyme_pred_prob = get_isEnzymeRes(querydata=data_1515.iloc[:,8:1909], model_file=isEnzyme_model )

    # 4.获取Slice预测结果

    print(blast_res)


    print('ok')