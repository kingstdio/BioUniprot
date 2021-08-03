import pandas as pd
import numpy as np
from tools.ucTools import ucTools
from tqdm import tqdm
from jax_unirep import get_reps
from xgboost import XGBClassifier, data
import os


#region 准备Slice使用的文件
def prepare_slice_file(x_data, y_data, x_file, y_file, ec_label_dict):
    """
    准备Slice使用的文件
    Args:
        x_data: X数据
        y_data: Y数据
        x_file: X文件路径
        y_file: Y文件路径
        ec_label_dict: EC转标签字典

    Returns:

    """
    if ~os.path.exists(x_file):
         to_file_matrix(file=x_file, ds=x_data.round(4), col_num=1900, stype='feature')

    if ~os.path.exists(y_file):
        yds = pd.DataFrame()
        yds['label'] = y_data.apply(lambda x: ec_label_dict.get(x))
        yds['tags'] = 1

        to_file_matrix(
            file=y_file,
            ds=yds,
            col_num=max(ec_label_dict.values()),
            stype='label'
        )
    print('Write success')
#endregion

#region 创建slice需要的数据文件
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
    
    if stype== 'label':
        seps = ':'
    if stype == 'feature':
        seps = ' '
    ds.to_csv(file, index= 0, header =0 , sep= seps)

    cmd ='''sed -i '1i {0} {1}' {2}'''.format(len(ds), col_num, file)
    os.system(cmd)
#endregion

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
def getblast(ref_fasta, query_fasta, results_file):
    """[获取序列比对结果]
    Args:
        ref_fasta ([string]]): [训练集fasta文件]
        query_fasta ([string]): [测试数据集fasta文件]
        results_file ([string]): [要保存的结果文件]

    Returns:
        [DataFrame]: [比对结果]
    """
    res_table_head = ['id', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send',
                      'evalue', 'bitscore']

    if os.path.exists(results_file):
        res_data = pd.read_csv(results_file, sep='\t', names=res_table_head)
        return res_data


    cmd1 = r'diamond makedb --in {0} -d /tmp/train.dmnd'.format(ref_fasta)
    cmd2 = r'diamond blastp -d /tmp/train.dmnd  -q  {0} -o {1} -b8 -c1 -k 1'.format(query_fasta, results_file)
    cmd3 = r'rm -rf /tmp/*.dmnd'

    print(cmd1)
    os.system(cmd1)
    print(cmd2)
    os.system(cmd2)
    res_data = pd.read_csv(results_file, sep='\t', names=res_table_head)
    os.system(cmd3)
    return res_data


# endregion
if __name__ =='__main__':
    print('success')