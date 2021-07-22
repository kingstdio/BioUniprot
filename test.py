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


# region 需要写fasta的dataFame格式
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


# endregion

# region 获取序列比对结果
def getblast(refdata, querydata, blastres_file):
    """[获取序列比对结果]
    Args:
        refdata ([string]]): [训练集fasta文件]
        querydata ([DataFrame]): [测试数据集]
        blastres_file ([string]): [要保存的结果文件]

    Returns:
        [DataFrame]: [比对结果]
    """
    res_table_head = ['id', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send',
                      'evalue', 'bitscore']

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
    res_data.drop_duplicates(subset=None, keep='first', inplace=True)
    return res_data


# endregion

# region 获取「酶｜非酶」预测结果
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
    return predict, predictprob[:, 1]


# endregion 获取「酶｜非酶」预测结果

# region 创建slice需要的数据文件
def to_file_matrix(file, ds, col_num, stype='label'):
    """[创建slice需要的数据文件]

    Args:
        file ([string]): [要保存的文件名]
        ds ([DataFrame]): [数据]
        col_num ([int]): [有多少列]
        stype (str, optional): [文件类型：feature，label]. Defaults to 'label'.
    """
    if os.path.exists(file):
        return 'file exist'

    if stype == 'label':
        seps = ':'
    if stype == 'feature':
        seps = ' '
    ds.to_csv(file, index=0, header=0, sep=seps)

    cmd = '''sed -i '1i {0} {1}' {2}'''.format(len(ds), col_num, file)
    os.system(cmd)


# endregion

# region 获取slice预测结果
def get_slice_res(query_data, model_path, res_file, file_slice='./data/preprocess/slice_x_1515.txt'):
    """[获取slice预测结果]

    Args:
        file_slice: slice模型需要的x文件，默认为./data/preprocess/slice_x_1515.txt，不存在则创建
        query_data ([DataFrame]): [需要预测的数据]
        model_path ([string]): [Slice模型路径]
        res_file ([string]]): [预测结果文件]
    Returns:
        [DataFrame]: [预测结果]
    """

    if ~os.path.exists(file_slice):
        to_file_matrix(file=file_slice, ds=query_data.round(4), col_num=1900, stype='feature')

    cmd = '''./slice_predict {0} {1} {2} -o 32 -b 0 -t 32 -q 0'''.format(file_slice, model_path, res_file)
    print(cmd)
    os.system(cmd)
    result_slice = pd.read_csv(res_file, header=None, skiprows=1, sep=' ')
    return result_slice


# endregion

# region 给比对结果添加标签
def get_blast_with_pred_label(blast_data, cnx):
    """
    给比对结果添加标签
    Args:
        blast_data: 比对结果
        cnx: 数据库连接

    Returns: 添加标签后的比对结果

    """
    sql = '''SELECT id, 
            isenzyme, 
            "isMultiFunctional",
            "functionCounts", 
            ec_number, 
            ec_specific_level from tb_sprot 
        where id in ('{0}') '''.format('\',\''.join(blast_data.sseqid.values))
    blast_tags = pd.read_sql_query(sql, cnx)
    blast_final = blast_data.merge(blast_tags, left_on='sseqid', right_on='id', how='left')
    blast_final.drop_duplicates(subset=['id_x'], keep='first', inplace=True)
    blast_final = blast_final[['id_x', 'pident', 'isenzyme', 'isMultiFunctional', 'functionCounts', 'ec_number']]
    blast_final = blast_final.rename(columns={'id_x': 'id',
                                              'isenzyme': 'isenzyme_blast',
                                              'isMultiFunctional': 'isMultiFunctional_blast',
                                              'functionCounts': 'functionCounts_blast',
                                              'ec_number': 'ec_number_blast'
                                              })
    blast_final.reset_index(drop=True, inplace=True)
    return blast_final


# endregion


def get_final_results(ori_data, blast_data, isenzyme_data, slice_data, cnx, file_label_ec):
    # 获取blast后的结果
    blast_res_final = get_blast_with_pred_label(blast_data=blast_data, cnx=cnx)
    ori_data = ori_data[['id', 'ec_number']]
    ori_data['isenzyme_slice'] = isenzyme_data

    data_test_blast_res = ori_data.merge(blast_res_final, on=['id'], how='left')

    # 获取slice结果
    slice_pred_top, slice_pred_top_pb = sort_slice_results(slice_data)
    slice_pred_top = slice_label2ec(slice_pred=slice_pred_top, file_label_ec=file_label_ec)

    slice_id_out = pd.concat([ori_data.iloc[:, 0], slice_pred_top], axis=1)
    final_pred = data_test_blast_res.merge(slice_id_out, on='id', how='left')

    final_pred.drop_duplicates(subset=None, keep='first', inplace=True)
    return final_pred


# region 将slice预测的标签转换为可读的EC号
def slice_label2ec(slice_pred, file_label_ec):
    """
    将slice预测的标签转换为可读的EC号
    Args:
        slice_pred: slice预测结果标签
        file_label_ec: 标签转EC号的字典文件
    Returns: 转化完成的EC号

    """
    dict_label_ec = np.load(file_label_ec, allow_pickle=True).item()
    slice_pred_ec_top = pd.DataFrame()
    for i in range(slice_pred.shape[1]):
        slice_pred_ec_top[i] = slice_pred[i].apply(lambda x: dict_label_ec.get(x))
    return slice_pred_ec_top


# endregion

# region 将slice的实验结果排序，并按照推荐顺序以两个矩阵的形式返回
def sort_slice_results(result_slice):
    """
    将slice的实验结果排序，并按照推荐顺序以两个矩阵的形式返回
    @pred_top：预测结果排序
    @pred_pb_top：预测结果评分排序
    """
    pred_top = []
    pred_pb_top = []
    aac = []
    for index, row in result_slice.iterrows():
        row_trans = [*row.apply(lambda x: x.split(':')).values]
        row_trans = pd.DataFrame(row_trans).sort_values(by=[1], ascending=False)
        pred_top += [list(np.array(row_trans[0]).astype('int'))]
        pred_pb_top += [list(np.array(row_trans[1]).astype('float'))]
    pred_top = pd.DataFrame(pred_top)
    pred_pb_top = pd.DataFrame(pred_pb_top)
    return pred_top, pred_pb_top


# endregion

if __name__ == '__main__':
    uctools = ucTools('172.16.25.20')
    cnx_ecdb = uctools.db_conn()

    # 1. 加载序列序列文件
    file_1515 = './tasks/task3/data/ecoli1515/1515testset_with_unirep.tsv'
    data_1515 = pd.read_csv(file_1515, sep='\t', index_col=0)
    colname = [
        ['id',
         'ec_number',
         'seq',
         'isenzyme',
         'isMultiFunctional',
         'blattner_id',
         'blattner_ec_number'] + ['f' + str(i) for i in range(1, 1901)]]

    data_1515.columns = np.array(colname).flatten()
    data_1515.drop_duplicates(subset=['seq'], keep='first', inplace=True)

    file_ref = './data/preprocess/enzyme_train.fasta'
    blast_res_file = './results/1515/balastres.tsv'

    # 2. 获取序列比对结果
    blast_res = getblast(querydata=data_1515, refdata=file_ref, blastres_file=blast_res_file)

    # 3.获取酶-非酶预测结果
    isEnzyme_model = './model/isenzyme.model'
    isEnzyme_pred, isEnzyme_pred_prob = get_isEnzymeRes(querydata=data_1515.iloc[:, 8:1909], model_file=isEnzyme_model)

    # 4.获取Slice预测结果
    slice_pred = get_slice_res(data_1515.iloc[:, 7:1909], './model', './results/1515/slice_results.txt')

    # 5. 获取最终结果
    file_label_ec = './data/dict_label_ec.npy'
    final_pred = get_final_results(ori_data=data_1515,
                                   blast_data=blast_res,
                                   isenzyme_data=isEnzyme_pred,
                                   slice_data=slice_pred,
                                   cnx=cnx_ecdb,
                                   file_label_ec=file_label_ec
                                   )

    print(final_pred)
