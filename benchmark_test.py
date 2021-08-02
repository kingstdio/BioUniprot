import re
import pandas as pd
import numpy as np
import joblib
import os
import benchmark_common as bcommon

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

# region 获取slice预测结果
def get_slice_res(slice_query_file, model_path, res_file):
    """[获取slice预测结果]

    Args:
        slice_query_file ([string]): [需要预测的数据sliceFile]
        model_path ([string]): [Slice模型路径]
        res_file ([string]]): [预测结果文件]
    Returns:
        [DataFrame]: [预测结果]
    """

    cmd = '''./slice_predict {0} {1} {2} -o 32 -b 0 -t 32 -q 0'''.format(slice_query_file, model_path, res_file)
    print(cmd)
    os.system(cmd)
    result_slice = pd.read_csv(res_file, header=None, skiprows=1, sep=' ')
    return result_slice


# endregion

#region 将slice的实验结果排序，并按照推荐顺序以两个矩阵的形式返回
def sort_results(result_slice):
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
#endregion

#region 划分测试集XY
def get_test_set(data):
    """[划分测试集XY]

    Args:
        data ([DataFrame]): [测试数据]

    Returns:
        [DataFrame]: [划分好的XY]
    """
    testX = data.iloc[:,4:]
    testY = data.iloc[:,:4]
    return testX, testY
#endregion

#region 将slice预测的标签转换为EC号
def translate_slice_pred(slice_pred, ec2label_dict, test_set):
    """[将slice预测的标签转换为EC号]

    Args:
        slice_pred ([DataFrame]): [slice预测后的排序数据]
        ec2label_dict ([dict]]): [ec转label的字典]
        test_set ([DataFrame]): [测试集用于取ID]

    Returns:
        [type]: [description]
    """
    label_ec_dict = {value:key for key, value in ec2label_dict.items()}
    res_df = pd.DataFrame()
    res_df['id'] = test_set.id
    colNames = slice_pred.columns.values
    for colName in colNames:
        res_df['top'+str(colName)] = slice_pred[colName].apply(lambda x: label_ec_dict.get(x))
    return res_df
#endregion

#region 为blast序列比对添加结果标签
def blast_add_label(blast_df, trainset,):
    """[为blast序列比对添加结果标签]

    Args:
        blast_df ([DataFrame]): [序列比对结果]
        trainset ([DataFrame]): [训练数据]

    Returns:
        [Dataframe]: [添加标签后的数据]
    """
    res_df = blast_df.merge(trainset, left_on='sseqid', right_on='id', how='left')
    res_df = res_df.rename(columns={'id_x': 'id',
                                              'isemzyme': 'isemzyme_blast',
                                              'functionCounts': 'functionCounts_blast',
                                              'ec_number': 'ec_number_blast'
                                              })
    return res_df.iloc[:,np.r_[0,13:16]]
#endregion


def run_integrage(slice_pred):
    #酶集成标签
    slice_pred['is_enzyme_i'] = slice_pred.apply(lambda x: int(x.isemzyme_blast) if str(x.isemzyme_blast)!='nan' else x.isEnzyme_pred_xg, axis=1)

    # 清空非酶的EC预测标签
    for i in range(20):
        slice_pred['top'+str(i)] = slice_pred.apply(lambda x: '' if x.is_enzyme_i==0 else x['top'+str(i)], axis=1)
        slice_pred['top0'] = slice_pred.apply(lambda x: x.ec_number_blast if str(x.ec_number_blast)!='nan' else x.top0, axis=1)

    # 清空有比对结果的预测标签
    for i in range(1,20):
        slice_pred['top'+str(i)] = slice_pred.apply(lambda x: '' if str(x.ec_number_blast)!='nan' else x['top'+str(i)], axis=1)
    
    slice_pred['top0']=slice_pred['top0'].apply(lambda x: '' if x=='-' else x)
    slice_pred=slice_pred.iloc[:,np.r_[0:5,21:26]]
    slice_pred.reset_index(drop=True, inplace=True)

    return slice_pred


if __name__ == '__main__':

    # 1. 定义数据目录
    DATADIR = r'''/home/shizhenkun/codebase/BioUniprot/data/benchmark/data/'''
    RESULTSDIR = r'''/home/shizhenkun/codebase/BioUniprot/data/benchmark/results/'''
    MODELDIR = r'''/home/shizhenkun/codebase/BioUniprot/data/benchmark/model'''


    # 2. 读入数据
    train = pd.read_feather(DATADIR+'train.feather').iloc[:,:4]
    test = pd.read_feather(DATADIR+'test.feather')
    train_fasta = DATADIR+'train.fasta'
    test_fasta = DATADIR+'test.fasta'

    # EC-标签字典
    dict_ec_label = np.load(DATADIR + 'ec_label_dict.npy', allow_pickle=True).item()


    file_blast_res = RESULTSDIR + r'test_blast_res.tsv'


    # 2. 获取序列比对结果
    blast_res = bcommon.getblast(query_fasta=test_fasta, ref_fasta=train_fasta, results_file=file_blast_res)
    blast_res = blast_add_label(blast_df=blast_res, trainset=train)



    # 3.获取酶-非酶预测结果
    isEnzyme_model = MODELDIR + '/isenzyme.model'
    testX, testY = get_test_set(data=test)

    isEnzyme_pred, isEnzyme_pred_prob = get_isEnzymeRes(querydata=testX, model_file=isEnzyme_model)

    # pd.DataFrame(isEnzyme_pred).to_csv(RESULTSDIR+'slice_isEnzyme_pred.tsv', sep='\t', index=None)
    
    # 4.获取Slice预测结果
    # 4.1 准备slice所用文件
    bcommon.prepare_slice_file( x_data=testX, 
                                y_data=testY['ec_number'], 
                                x_file=DATADIR+'slice_test_x.txt', 
                                y_file=DATADIR+'slice_test_y.txt', 
                                ec_label_dict=dict_ec_label
                            )
    # 4.2 获得预测结果
    slice_pred = get_slice_res(slice_query_file= DATADIR + 'slice_test_x.txt', model_path=  MODELDIR, res_file= RESULTSDIR + 'slice_results.txt')

    # 4.3 对预测结果排序
    slice_pred_rank, slice_pred_prob = sort_results(slice_pred)

    # 4.4 将结果翻译成EC号
    slice_pred_ec = translate_slice_pred(slice_pred=slice_pred_rank, ec2label_dict = dict_ec_label, test_set=test)
    slice_pred_ec['isEnzyme_pred_xg'] = isEnzyme_pred


   
    slice_pred_ec = slice_pred_ec.merge(blast_res, on='id', how='left')
    slice_pred_ec = run_integrage(slice_pred=slice_pred_ec)

    slice_pred_ec.to_csv(RESULTSDIR+'slice_pred.tsv', sep='\t', index=None)

    print('ok')