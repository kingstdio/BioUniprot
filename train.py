import pandas as pd
import numpy as np
import joblib
from tools.ucTools import ucTools
from tqdm import tqdm
from jax_unirep import get_reps
from xgboost import XGBClassifier
import os


table_head = [  'id', 
                'name',
                'isemzyme',
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


#region 加载EC号与标签字典
def load_ec_lable_dict(filepath):
    """加载EC号与标签字典

    Args:
        filepath ([string]): [字典文件]

    Returns:
        [dict]: [dict_ec_label，dict_label_ec]
    """
    dict_ec_label= np.load(filepath, allow_pickle=True).item()
    dict_label_ec = dict(zip(dict_ec_label.values(), dict_ec_label.keys()))
    return dict_ec_label, dict_label_ec
#endregion

#region 按照分割日期加载训练数据
def load_train_file(split_date, cnx_ecnumber, type='enzyme'):
    """[按照分割日期加载训练数据]

    Args:
        split_date ([string]): [截止日期]

    Returns:
        [DataFrame]]: [训练数据]
    """
    if type=='enzyme':
        sql ='''
        SELECT id, isemzyme, "isMultiFunctional", "functionCounts", ec_number, ec_specific_level, seq from tb_sprot where date_integraged::date < '{0}';
        '''.format(split_date)
    if type=='ec':
        sql ='''
        SELECT id, isemzyme, "isMultiFunctional", "functionCounts", ec_number, ec_specific_level, seq from tb_sprot where date_integraged::date < '{0}' and ec_specific_level=4;;
        '''.format(split_date)
    sprot_data = pd.read_sql_query(sql, cnx_ecnumber)
    return sprot_data
#endregion

#region 给训练集加上EC标签
def add_ec_label(data, ec_label_dict):
    """给训练集加上EC标签

    Args:
        data ([DataFame]]): [训练数据]
        ec_label_dict ([dict]): [EC与标签的转化字典]]

    Returns:
        [DataFame]: [添加标签数据后的DataFrame]
    """
    data['ec_label'] = data.ec_number.apply(lambda x: ec_label_dict.get(x))
    data['enzyme_label'] = data.ec_number.apply(lambda x: 0 if x=='-' else 1)
    return data
#endregion

def load_unirep(file):
    data = pd.read_feather(file)
    return data

#region 获取酶训练的数据集
def get_enzyme_train_set(train_date, eclablefile, featurefile, cnx, trainX='enzyme_train_x.feather', trainY='enzyme_train_y.feather' ):
    """[获取酶训练的数据集]

    Args:
        train_date ([string]): [训练数据集截取的时间]
        eclablefile ([stging]): [ec to label 字典文件]
        featurefile ([string]): [unirep后的特征文件]
        cnx ([string]): [数据库连接符]
        trainX (str, optional): [已有的训练数据X]. Defaults to 'enzyme_train_x.feather'.
        trainY (str, optional): [已有的训练数据Y]. Defaults to 'enzyme_train_y.feather'.

    Returns:
        [type]: [description]
    """
    if os.path.exists(trainX):
        trian_X = pd.read_feather(trainX)
        train_Y = pd.read_feather(trainY)
    else:
        #加载训练数据
        dict_ec_label, dict_label_ec = load_ec_lable_dict(eclablefile)
        train_data = load_train_file(split_date= train_date, cnx_ecnumber=cnx, type='ec')
        
        #添加训练标签
        train_data_labeled  = add_ec_label(train_data, dict_ec_label)

        unirep_data = load_unirep(featurefile)

        res_data = train_data_labeled.merge(unirep_data, on='id', how='left')

        train_Y = res_data[['id', 'seq', 'enzyme_label']]
        trian_X = res_data.iloc[:,10:1910]

        trian_X.to_feather(trainX)
        train_Y.to_feather(trainY)

    return trian_X, train_Y
#endregion


def xgmain(X_train_std, Y_train, model_file):
    model = XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, n_jobs=-2, eval_metric='mlogloss')
    model.fit(X_train_std, Y_train.ravel())
    joblib.dump(model, model_file)
    print('XGBoost模型训练完成')

def train_isenzyme(X,Y, model_file):
    xgmain(X,Y,model_file)


def get_ec_train_set(train_date, eclabelfile, featurefile, cnx, trainX='ec_train_x.feather', trainY='ec_train_y.feather'):
    if os.path.exists(trainX):
        trian_X = pd.read_feather(trainX)
        train_Y = pd.read_feather(trainY)
    else:
        dict_ec_label, dict_label_ec = load_ec_lable_dict(eclabelfile)
        train_data = load_train_file(split_date= train_date, cnx_ecnumber=cnx)
        train_data_labeled  = add_ec_label(train_data, dict_ec_label)
        unirep_data = load_unirep(featurefile)
        res_data = train_data_labeled.merge(unirep_data, on='id', how='left')
        train_Y = res_data[['id', 'seq', 'ec_label']]
        trian_X = res_data.iloc[:,9:1909]
        trian_X.to_feather(trainX)
        train_Y.to_feather(trainY)

    return trian_X, train_Y

if __name__ =="__main__":
    #1. 数据库连接
    uctools =  ucTools('172.16.25.20')
    cnx_ecnumber = uctools.db_conn()

    #2. 加载「酶｜非酶」训练数据
    DATE_TRAIN ='2018-01-01'

    file_enzyme_train_x = './data/preprocess/enzyme_train_x.feather'
    file_enzyme_train_y = './data/preprocess/enzyme_train_y.feather'

    enzyme_X, enzyme_Y = get_enzyme_train_set( train_date=DATE_TRAIN, 
                                eclablefile='./data/dict_ec_label.npy', 
                                featurefile='./data/sprot_unirep.feather', 
                                cnx=cnx_ecnumber,
                                trainX=file_enzyme_train_x,
                                trainY=file_enzyme_train_y
                              )
    
   

    #3. 「酶｜非酶」模型训练
    # file_isenzyme_model = './model/isenzyme.model'
    # train_isenzyme(enzyme_X, enzyme_Y.enzyme_label, model_file=file_isenzyme_model)

    #4. 加载EC号训练数据
    file_ec_train_x = './data/preprocess/ec_train_x.feather'
    file_ec_train_y = './data/preprocess/ec_train_y.feather'
    ec_X, ec_Y = get_ec_train_set(train_date=DATE_TRAIN,
                                    eclabelfile='./data/dict_ec_label.npy',
                                    featurefile='./data/sprot_unirep.feather',
                                    cnx=cnx_ecnumber,
                                    trainX=file_ec_train_x,
                                    trainY=file_ec_train_y
                                )



    # print(train_data_labeled)

    
    print('ok')