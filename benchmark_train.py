from pickle import FALSE
import pandas as pd
import numpy as np
import joblib
from tools.ucTools import ucTools
from tqdm import tqdm
from jax_unirep import get_reps
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import os
import subprocess


#region 获取酶训练的数据集
def get_enzyme_train_set(traindata):
    """[获取酶训练的数据集]

    Args:
        traindata ([DataFrame]): [description]

    Returns:
        [DataFrame]: [trianX, trainY]
    """
    train_X = traindata.iloc[:,4:]
    train_Y = traindata['isemzyme'].astype('int')
    return train_X, train_Y
#endregion

#region 获取几功能酶训练的数据集
def get_howmany_train_set(train_data):
    """[获取几功能酶训练的数据集]
    Args:
        train_data ([DataFrame]): [完整训练数据]
    Returns:
        [DataFrame]: [[train_x, trian_y]]
    """
    train_data = train_data[train_data.isemzyme] #仅选用酶数据
    train_X = train_data.iloc[:,4:1904]
    train_Y =train_data['functionCounts'].astype('int')
    return train_X, pd.DataFrame(train_Y)

#endregion

#region 获取EC号训练的数据集
def get_ec_train_set(train_data, ec_label_dict):
    """[获取EC号训练的数据集]
    Args:
        ec_label_dict: ([dict]): [ec to label dict]
        train_data ([DataFrame]): [description]
    Returns:
        [DataFrame]: [[train_x, trian_y]]
    """
    train_data = train_data[train_data.isemzyme] #仅选用酶数据
    train_data = train_data[train_data.functionCounts ==1] #仅选用单功能酶数据
    train_data['ec_label'] = train_data.ec_number.apply(lambda x: ec_label_dict.get(x))
    train_X = train_data.iloc[:,4:1904]
    train_Y =train_data['ec_label']
    return train_X, pd.DataFrame(train_Y)

#endregion

#region 训练是否是酶模型
def train_isenzyme(X,Y, model_file, force_model_update=False):
    """[训练是否是酶模型]

    Args:
        X ([DataFrame]): [特征数据]
        Y ([DataFrame]): [标签数据]
        model_file ([string]): [模型的存放路径]
        force_model_update (bool, optional): [是否强制更新模型]. Defaults to False.

    Returns:
        [object]: [训练好的模型]
    """
    if os.path.exists(model_file) and (force_model_update==False):
        return
    else:
        model = XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, n_jobs=-2, eval_metric='mlogloss')
        print(model)
        model.fit(X, Y.ravel())
        joblib.dump(model, model_file)
        print('XGBoost模型训练完成')
        return model
#endregion

def importance_features_top(model, x_train, topN=10):
    """打印模型的重要指标，排名topN指标"""
    print("打印XGBoost重要指标")
    feature_importances_ = model.feature_importances_
    feature_names = x_train.columns
    importance_col = pd.DataFrame([*zip(feature_names, feature_importances_)],  columns=['features', 'weight'])
    importance_col_desc = importance_col.sort_values(by='weight', ascending=False)
    print(importance_col_desc.iloc[:topN, :])

#region 构建几功能酶模型
def train_howmany_enzyme(data_x, data_y, model_file, force_model_update=False):
    """[构建几功能酶模型]

    Args:
        data_x ([DataFrame]): [X训练数据]
        data_y ([DataFrame]): [Y训练数据]

    Returns:
        [object]: [训练好的模型]
    """
    if os.path.exists(model_file) and (force_model_update==False):
        return
    else:
        x_train, x_vali, y_train, y_vali = train_test_split(data_x,np.array(data_y).ravel(),test_size=0.3,random_state=1)
        eval_set = [(x_train, y_train), (x_vali, y_vali)]
        
        model = XGBClassifier(min_child_weight=6, max_depth=15, objective='multi:softmax', num_class=11, use_label_encoder=False)
        print("-" * 100)
        print("几功能酶xgboost模型：", model)
        model.fit(x_train, y_train, eval_metric="mlogloss", eval_set=eval_set, verbose=True)
        # # 打印重要性指数
        importance_features_top(model, x_train, topN=50)
        # 保存模型
        joblib.dump(model, model_file)
        return model
#endregion


def make_ec_label(train_label, test_label, file_save):
    if os.path.exists(file_save):
        print('ec label dict already exist')
        return
    ecset = sorted( set(list(train_label) + list(test_label)))
    ec_label_dict = {k: v for k, v in zip(ecset, range(len(ecset)))}
    np.save(file_save, ec_label_dict)
    return  ec_label_dict
    print('字典保存成功')



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

def prepare_slice_file(x_data, y_data, x_file, y_file, ec_label_dict):
    if ~os.path.exists(x_file):
         to_file_matrix(file=x_file, ds=x_data.round(4), col_num=1900, stype='feature')
    if ~os.path.exists(y_file):        
        to_file_matrix(file=y_file,ds=y_data, col_num=max(ec_label_dict.values()), stype='label')
    print('Write success')


def train_ec_slice(trainX, trainY, modelPath, force_model_update=False):
    if os.path.exists(modelPath+'param') and (force_model_update==False):
        print('model exist')
        return

    cmd = ''' ./slice_train {0} {1} {2} -m 100 -c 300 -s 300 -k 700 -o 32 -t 32 -C 1 -f 0.000001 -siter 20 -stype 0 -q 0 '''.format(trainX, trainY, modelPath)
    print(cmd)
    os.system(cmd)
    print('train finished')

#region 需要写fasta的dataFame格式
def save_train2fasta(dataset, file_out):
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


if __name__ =="__main__":
    # 1. 定义数据目录
    DATADIR = r'''/home/shizhenkun/codebase/BioUniprot/data/benchmark/data/'''
    RESULTSDIR = r'''/home/shizhenkun/codebase/BioUniprot/data/benchmark/results/'''
    MODELDIR = r'''/home/shizhenkun/codebase/BioUniprot/data/benchmark/model'''

    UPDATE_MODEL = False #强制模型更新标志

    # 2. 读入数据
    train = pd.read_feather(DATADIR+'train.feather')
    test = pd.read_feather(DATADIR+'test.feather')
    train_fasta = DATADIR+'train.fasta'
    test_fasta = DATADIR+'test.fasta'

    #3. 「酶｜非酶」模型训练
    enzyme_X, enzyme_Y = get_enzyme_train_set(train)
    train_isenzyme(X=enzyme_X, Y=enzyme_Y, model_file=MODELDIR+'isenzyme.model', force_model_update=UPDATE_MODEL)

    #4. 「几功能酶训练模型」训练

    howmany_X, howmay_Y = get_howmany_train_set(train)
    train_howmany_enzyme(data_x=howmany_X, data_y=(howmay_Y-1), model_file=MODELDIR+'howmany_enzyme.model', force_model_update=UPDATE_MODEL)

    #4. 加载EC号训练数据
    if os.path.exists(DATADIR+'ec_label_dict.npy'):
        dict_ec_label = np.load(DATADIR+'ec_label_dict.npy', allow_pickle=True).item()
    else:
        dict_ec_label = make_ec_label(train_label=train['ec_number'], test_label=test['ec_number'], file_save= DATADIR+'ec_label_dict.npy')

    ec_X, ec_Y = get_ec_train_set(train_data=train, ec_label_dict=dict_ec_label )
    
    #5. 构建Slice文件
    file_slice_train_x = DATADIR + 'slice_train_x.txt'
    file_slice_train_y = DATADIR + 'slice_train_y.txt'
    ec_Y['tags'] = 1
    prepare_slice_file(x_data=ec_X, y_data=ec_Y, x_file=file_slice_train_x, y_file=file_slice_train_y, ec_label_dict=dict_ec_label)
    
    #6. 训练Slice模型

    train_ec_slice(trainX=file_slice_train_x, trainY=file_slice_train_y, modelPath=MODELDIR)


    print('训练完成')