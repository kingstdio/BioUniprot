import re
import pandas as pd
import numpy as np
import joblib
import os

from pandas._config.config import reset_option
import benchmark_common as bcommon
import benchmark_config as cfg


#region 加载各种算法的预测结果，并拼合成一个大表格
def load_res_data(file_slice, file_blast, file_deepec, file_ecpred, train, test):
    """[加载各种算法的预测结果，并拼合成一个大表格]

    Args:
        file_slice ([string]]): [slice 集成预测结果文件]
        file_blast ([string]): [blast 序列比对结果文件]
        file_deepec ([string]): [deepec 预测结果文件]
        file_ecpred ([string]): [ecpred 预测结果文件]
        train ([DataFrame]): [训练集]
        test ([DataFrame]): [测试集]

    Returns:
        [DataFrame]: [拼合完成的大表格]
    """
    res_slice = pd.read_csv(file_slice, sep='\t')
    res_blast = pd.read_csv(file_blast, sep='\t',  names =cfg.BLAST_TABLE_HEAD)
    res_blast = bcommon.blast_add_label(blast_df=res_blast, trainset=train)
    ground_truth = test.iloc[:, np.r_[0,1,2,3]]

    big_res = res_slice.merge(ground_truth, on='id', how='left')
    big_res = big_res.merge(res_blast, on='id', how='left') 

    # DeepEC
    res_deepec = pd.read_csv(file_deepec, sep='\t',names=['id', 'ec_number'], header=0 )
    res_deepec.ec_number=res_deepec.apply(lambda x: x['ec_number'].replace('EC:',''), axis=1)
    res_deepec.columns = ['id','ec_number_deepecpred']
    big_res = big_res.merge(res_deepec, on='id', how='left').drop_duplicates(subset='id')

    # ECpred
    res_ecpred = pd.read_csv(file_ecpred, sep='\t', header=0)
    res_ecpred['isemzyme_ecpred'] = ''
    with pd.option_context('mode.chained_assignment', None):
        res_ecpred.isemzyme_ecpred[res_ecpred['EC Number']=='non Enzyme'] = False
        res_ecpred.isemzyme_ecpred[res_ecpred['EC Number']!='non Enzyme'] = True
        
    res_ecpred.columns = ['id','ec_number_ecpred', 'conf', 'isemzyme_ecpred']
    res_ecpred = res_ecpred.iloc[:,np.r_[0,1,3]]

    big_res = big_res.merge(res_ecpred, on='id', how='left').drop_duplicates(subset='id')
    big_res=big_res.sort_values(by=['isemzyme', 'id'], ascending=False)
    big_res.reset_index(drop=True, inplace=True)
    return big_res
#endregion


def integrage_reslults(big_table):
    # 拼合多个标签
    big_table['pred_eci']=big_table.apply(lambda x : ', '.join(x.iloc[1:(x.pred_functionCounts+1)].values.astype('str')), axis=1)

    #给非酶赋-
    with pd.option_context('mode.chained_assignment', None):
        big_table.pred_eci[big_table.pred_eci=='nan']='-'
        big_table.pred_eci[big_table.pred_eci=='']='-'




if __name__ =='__main__':

    # 2. 读入数据
    train = pd.read_feather(cfg.DATADIR+'train.feather').iloc[:,:4]
    test = pd.read_feather(cfg.DATADIR+'test.feather').iloc[:,:4]
    train_fasta = cfg.DATADIR+'train.fasta'
    test_fasta = cfg.DATADIR+'test.fasta'

    # EC-标签字典
    dict_ec_label = np.load(cfg.DATADIR + 'ec_label_dict.npy', allow_pickle=True).item()
    file_blast_res = cfg.RESULTSDIR + r'test_blast_res.tsv'

    flat_table = load_res_data(
        file_slice=cfg.FILE_INTE_RESULTS,
        file_blast=cfg.FILE_BLAST_RESULTS,
        file_deepec=cfg.FILE_DEEPEC_RESULTS,
        file_ecpred=cfg.FILE_ECPRED_RESULTS,
        train=train,
        test=test
        )

    print(flat_table)

    print('success')