from operator import index
from random import sample
import pandas as pd
import numpy as np
import joblib
import os
from sklearn import metrics

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
    res_deepec.columns = ['id','ec_deepec']
    big_res = big_res.merge(res_deepec, on='id', how='left').drop_duplicates(subset='id')

    # ECpred
    res_ecpred = pd.read_csv(file_ecpred, sep='\t', header=0)
    res_ecpred['isemzyme_ecpred'] = ''
    with pd.option_context('mode.chained_assignment', None):
        res_ecpred.isemzyme_ecpred[res_ecpred['EC Number']=='non Enzyme'] = False
        res_ecpred.isemzyme_ecpred[res_ecpred['EC Number']!='non Enzyme'] = True
        
    res_ecpred.columns = ['id','ec_ecpred', 'conf', 'isemzyme_ecpred']
    res_ecpred = res_ecpred.iloc[:,np.r_[0,1,3]]

    big_res = big_res.merge(res_ecpred, on='id', how='left').drop_duplicates(subset='id')
    big_res=big_res.sort_values(by=['isemzyme', 'id'], ascending=False)
    big_res.reset_index(drop=True, inplace=True)
    return big_res
#endregion


def integrate_reslults(big_table):
    # 拼合多个标签
    big_table['pred_eci']=big_table.apply(lambda x : ', '.join(x.iloc[1:(x.pred_functionCounts+1)].values.astype('str')), axis=1)

    #给非酶赋-
    with pd.option_context('mode.chained_assignment', None):
        big_table.pred_eci[big_table.pred_eci=='nan']='-'
        big_table.pred_eci[big_table.pred_eci=='']='-'
    
    big_table = big_table.iloc[:,np.r_[0,11:23]]
    #添加blast是否是酶
    big_table['isemzyme_deepec']=big_table.ec_deepec.apply(lambda x : 0 if str(x)=='nan' else 1)
    big_table = big_table[[
                        'id', 
                        'isemzyme',
                        'functionCounts', 
                        'ec_number', 
                        'pred_isEnzyme', 
                        'pred_functionCounts', 
                        'pred_eci',  
                        'isemzyme_blast', 
                        'functionCounts_blast',
                        'ec_number_blast', 
                        'ec_deepec',
                        'isemzyme_deepec',
                        'ec_ecpred',
                        'isemzyme_ecpred']]
    big_table.columns = [   'id', 
                            'isenzyme_groundtruth', 
                            'functionCounts_groundtruth', 
                            'ec_groundtruth', 
                            'isenzyme_slice', 
                            'functionCounts_slice', 
                            'ec_slice',  
                            'isenzyme_blast',
                            'functionCounts_blast',
                            'ec_blast', 
                            'ec_deepec',
                            'isenzyme_deepec',
                            'ec_ecpred',
                            'isenzyme_ecpred']    
    # 拼合训练测试样本数
    samplecounts = pd.read_csv(cfg.DATADIR + 'ecsamplecounts.tsv', sep = '\t')
    big_table.merge(samplecounts, left_on='ec_groundtruth', right_on='ec_number', how='left')                
    big_table.to_excel(cfg.FILE_EVL_RESULTS, index=None)
    return big_table

def caculateMetrix(groundtruth, predict, baslineName):
    acc = metrics.accuracy_score(groundtruth, predict)
    precision = metrics.precision_score(groundtruth, predict, zero_division=1 )
    recall = metrics.recall_score(groundtruth, predict)
    f1 = metrics.f1_score(groundtruth, predict)
    tn, fp, fn, tp = metrics.confusion_matrix(groundtruth, predict).ravel()

    npv = tn/(fn+tn+1.4E-45)
    print(baslineName, '\t\t%f' %acc,'\t%f'% precision,'\t\t%f'%npv,'\t%f'% recall,'\t%f'% f1, '\t', 'tp:',tp,'fp:',fp,'fn:',fn,'tn:',tn)


def evalueate_performance(evalutation_table):
    print('\n\n1. isEnzyme prediction evalueation metrics')
    print('*'*140+'\n')
    print('baslineName', '\t', 'accuracy','\t', 'precision(PPV) \t NPV \t\t', 'recall','\t', 'f1', '\t\t', '\t confusion Matrix')
    caculateMetrix( groundtruth=evalutation_table.isenzyme_groundtruth, 
                    predict=evalutation_table.isenzyme_slice, 
                    baslineName='ours')
    caculateMetrix( groundtruth=evalutation_table.isenzyme_groundtruth.astype('int'), 
                    predict=evalutation_table.isenzyme_blast.fillna('0').astype('int') ,
                    baslineName='blast')
    caculateMetrix( groundtruth=evalutation_table.isenzyme_groundtruth.astype('int'), 
                    predict=evalutation_table.isenzyme_ecpred.fillna('0').astype('int') ,
                    baslineName='ecpred')
    caculateMetrix( groundtruth=evalutation_table.isenzyme_groundtruth.astype('int'), 
                    predict=evalutation_table.isenzyme_deepec ,
                    baslineName='deepec')

    



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
    evalutation_table = integrate_reslults(flat_table)
    
    evalueate_performance(evalutation_table)

    print('success')