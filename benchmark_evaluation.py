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
def load_res_data(file_slice, file_deepec, file_ecpred, train, test):
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
    isEnzyme_blast = bcommon.get_blast_prediction(  reference_db=cfg.FILE_ISENZYME_DB, 
                                                train_frame=train, 
                                                test_frame=test,
                                                results_file=cfg.FILE_BLAST_ISENAYME_RESULTS,
                                                type='isenzyme',
                                                identity_thres=cfg.TRAIN_BLAST_IDENTITY_THRES
                                            )

    ec_blast = bcommon.get_blast_prediction(  reference_db=cfg.FILE_EC_DB, 
                                                train_frame=train, 
                                                test_frame=test,
                                                results_file=cfg.FILE_BLAST_ISENAYME_RESULTS,
                                                type='ec',
                                                identity_thres=cfg.TRAIN_BLAST_IDENTITY_THRES
                                            )
    res_slice = pd.read_csv(file_slice, sep='\t')
    # res_blast = pd.read_csv(file_blast, sep='\t',  names =cfg.BLAST_TABLE_HEAD)
    # res_blast = bcommon.blast_add_label(blast_df=res_blast, trainset=train)
    ground_truth = test.iloc[:, np.r_[0,1,2,3]]

    big_res = res_slice.merge(ground_truth, on='id', how='left')
    big_res = big_res.merge(isEnzyme_blast, on='id', how='left') 
    big_res = big_res.merge(ec_blast, on='id', how='left') 

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
    with pd.option_context('mode.chained_assignment', None):
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
    big_table = big_table.merge(samplecounts, left_on='ec_groundtruth', right_on='ec_number', how='left')                
    big_table.iloc[:,np.r_[0:14,15:17]].to_excel(cfg.FILE_EVL_RESULTS, index=None)
    return big_table

def caculateMetrix(groundtruth, predict, baselineName, type='binary'):
    acc = metrics.accuracy_score(groundtruth, predict)
    if type == 'binary':
        precision = metrics.precision_score(groundtruth, predict, zero_division=True )
        recall = metrics.recall_score(groundtruth, predict,  zero_division=True)
        f1 = metrics.f1_score(groundtruth, predict, zero_division=True)
        tn, fp, fn, tp = metrics.confusion_matrix(groundtruth, predict).ravel()
        npv = tn/(fn+tn+1.4E-45)
        print(baselineName, '\t\t%f' %acc,'\t%f'% precision,'\t\t%f'%npv,'\t%f'% recall,'\t%f'% f1, '\t', 'tp:',tp,'fp:',fp,'fn:',fn,'tn:',tn)
    
    if type == 'multi':
        precision = metrics.precision_score(groundtruth, predict, average='macro', zero_division=True )
        recall = metrics.recall_score(groundtruth, predict, average='macro', zero_division=True)
        f1 = metrics.f1_score(groundtruth, predict, average='macro', zero_division=True)
        print('%12s'%baselineName, ' \t\t%f '%acc,'\t%f'% precision, '\t\t%f'% recall,'\t%f'% f1)

def evalueate_performance(evalutation_table):
    print('\n\n1. isEnzyme prediction evalueation metrics')
    print('*'*140+'\n')
    print('baslineName', '\t', 'accuracy','\t', 'precision(PPV) \t NPV \t\t', 'recall','\t', 'f1', '\t\t', '\t confusion Matrix')
    ev_isenzyme={'ours':'isenzyme_slice', 'blast':'isenzyme_blast', 'ecpred':'isenzyme_ecpred', 'deepec':'isenzyme_deepec'}

    for k,v in ev_isenzyme.items():
        caculateMetrix( groundtruth=evalutation_table.isenzyme_groundtruth, 
                        predict=evalutation_table[v].fillna('0').astype('int'), 
                        baselineName=k, 
                        type='binary')


    print('\n\n2. EC prediction evalueation metrics')
    print('*'*140+'\n')
    print('%12s'%'baslineName', '\t\t', 'accuracy','\t', 'precision-macro \t', 'recall-macro','\t', 'f1-macro')
    ev_ec = {'ours':'ec_slice', 'blast':'ec_blast', 'ecpred':'ec_ecpred', 'deepec':'ec_deepec'}
    for k, v in ev_ec.items():
        caculateMetrix( groundtruth=evalutation_table.ec_groundtruth, 
                predict=evalutation_table[v].fillna('NaN'), 
                baselineName=k, 
                type='multi')

    print('\n\n3. Function counts evalueation metrics')
    print('*'*140+'\n')
    print('%12s'%'baslineName', '\t\t', 'accuracy','\t', 'precision-macro \t', 'recall-macro','\t', 'f1-macro')
    ev_ec = {'ours':'functionCounts_slice', 'blast':'functionCounts_blast',  'ecpred':'functionCounts_ecpred','deepec':'functionCounts_deepec'}
    evalutation_table['functionCounts_deepec'] = evalutation_table.ec_deepec.apply(lambda x :len(str(x).split(',')))
    evalutation_table['functionCounts_ecpred'] = evalutation_table.ec_ecpred.apply(lambda x :len(str(x).split(',')))
    for k, v in ev_ec.items():
        caculateMetrix( groundtruth=evalutation_table.functionCounts_groundtruth, 
                predict=evalutation_table[v].fillna('-1').astype('int'), 
                baselineName=k, 
                type='multi')

if __name__ =='__main__':

    # 2. 读入数据
    train = pd.read_feather(cfg.DATADIR+'train.feather').iloc[:,:6]
    test = pd.read_feather(cfg.DATADIR+'test.feather').iloc[:,:6]
    test = test[(test.ec_specific_level>=cfg.TRAIN_USE_SPCIFIC_EC_LEVEL) |(~test.isemzyme)]
    test.reset_index(drop=True, inplace=True)
    # EC-标签字典
    dict_ec_label = np.load(cfg.DATADIR + 'ec_label_dict.npy', allow_pickle=True).item()
    file_blast_res = cfg.RESULTSDIR + r'test_blast_res.tsv'

    flat_table = load_res_data(
        file_slice=cfg.FILE_INTE_RESULTS,
        file_deepec=cfg.FILE_DEEPEC_RESULTS,
        file_ecpred=cfg.FILE_ECPRED_RESULTS,
        train=train,
        test=test
        )
        
    evalutation_table = integrate_reslults(flat_table)
    
    evalueate_performance(evalutation_table)

    print('\n Evaluation Finished \n\n')