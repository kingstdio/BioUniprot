from benchmark_config import FEATURE_NUM
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
import benchmark_config as cfg

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

#region 获取slice预测结果
def get_slice_res(query_data, model_path, res_file):
    """[获取slice预测结果]

    Args:
        query_data ([DataFrame]): [需要预测的数据]
        model_path ([string]): [Slice模型路径]
        res_file ([string]]): [预测结果文件]
    Returns:
        [DataFrame]: [预测结果]
    """
    file_slice_x_1515 = './data/preprocess/slice_x_1515.txt'
    if ~os.path.exists(file_slice_x_1515):
        to_file_matrix(file=file_slice_x_1515, ds=query_data.round(cfg.FEATURE_NUM),col_num=FEATURE_NUM, stype='feature')
    
    cmd = '''./slice_predict {0} {1} {2} -o 32 -b 0 -t 32 -q 0'''.format(file_slice_x_1515, model_path, res_file)
    print(cmd)
    os.system(cmd)
    result_slice = pd.read_csv(res_file,  header = None, skiprows=1 ,sep=' ')
    return result_slice
#endregion

def get_blast_with_pred_label(blast_data, cnx):
    sql ='''SELECT id, 
            isemzyme, 
            "isMultiFunctional",
            "functionCounts", 
            ec_number, 
            ec_specific_level from tb_sprot 
        where id in ('{0}') '''.format('\',\''.join(blast_data.sseqid.values))
    blast_tags = pd.read_sql_query(sql,cnx)
    blast_final = blast_data.merge(blast_tags,  left_on='sseqid', right_on='id', how='left')
    blast_final.drop_duplicates(subset=['id_x'], keep='first', inplace =True)
    blast_final=blast_final[['id_x', 'pident', 'isemzyme', 'isMultiFunctional', 'functionCounts', 'ec_number']]
    blast_final=blast_final.rename(columns={'id_x':'id', 
                                                'isemzyme':'isemzyme_blast', 
                                                'isMultiFunctional':'isMultiFunctional_blast', 
                                                'functionCounts':'functionCounts_blast', 
                                                'ec_number':'ec_number_blast'
                                            })
    blast_final.reset_index(drop=True, inplace=True)
    return blast_final



def get_final_results(ori_data, blast_data, slice_data, cnx):
    blast_res = get_blast_with_pred_label(blast_data=blast_data, cnx=cnx)


    ori_data = ori_data[['id', 'ec_number', 'blattner_ec_number']]
    ori_data.drop_duplicates(subset=None, keep='first', inplace =True)

    data_test_blast_res = ori_data.merge(blast_res, on=['id'], how='left')


def sort_results(result_slice):
    """
    将slice的实验结果排序，并按照推荐顺序以两个矩阵的形式返回
    @pred_top：预测结果排序
    @pred_pb_top：预测结果评分排序
    """
    pred_top =[]
    pred_pb_top =[]
    aac =[]
    for index, row in result_slice.iterrows():
        row_trans= [*row.apply(lambda x: x.split(':')).values]
        row_trans = pd.DataFrame(row_trans).sort_values(by=[1], ascending=False)
        pred_top += [list(np.array(row_trans[0]).astype('int'))]
        pred_pb_top += [list(np.array(row_trans[1]).astype('float'))]
    pred_top = pd.DataFrame(pred_top)
    pred_pb_top = pd.DataFrame(pred_pb_top)
    return pred_top,  pred_pb_top



if __name__ == '__main__':
    
    uctools =  ucTools('172.16.25.20')
    cnx_ecdb = uctools.db_conn()

    # 1. 加载序列序列文件
    file_1515 = './tasks/task3/data/ecoli1515/1515testset_with_unirep.tsv'
    data_1515 = pd.read_csv(file_1515, sep='\t', index_col=0)
    colname =  [['id', 'ec_number', 'seq', 'isemzyme', 'isMultiFunctional', 'blattner_id', 'blattner_ec_number' ] +['f'+str(i) for i in range(1,1901)]]
    data_1515.columns =colname

    file_ref = './data/preprocess/enzyme_train.fasta'
    blast_res_file = './results/1515/balastres.tsv'

    # 2. 获取序列比对结果
    blast_res = getblast(querydata=data_1515, refdata=file_ref, blastres_file=blast_res_file)


    # 3.获取酶-非酶预测结果
    isEnzyme_model = './model/isenzyme.model'
    isEnzyme_pred, isEnzyme_pred_prob = get_isEnzymeRes(querydata=data_1515.iloc[:,8:1909], model_file=isEnzyme_model )


    # 4.获取Slice预测结果
    slice_pred = get_slice_res(data_1515.iloc[:,7:1909], './model', './results/1515/slice_results.txt')
    


    print('ok')