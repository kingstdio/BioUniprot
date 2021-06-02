from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn import linear_model, datasets
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost
from xgboost import XGBClassifier

import pandas as pd
import numpy as np
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

def table2fasta(table, file_out):
    file = open(file_out, 'w')
    for index, row in table.iterrows():
        file.write('>{0}\n'.format(row['id']))
        file.write('{0}\n'.format(row['seq']))
    file.close()
    print('Write finished')
    
#氨基酸字典(X未知项用于数据对齐)
prot_dict = dict(
                    A=1,  R=2,  N=3,  D=4,  C=5,  E=6,  Q=7,  G=8,  H=9,  O=10, I=11, L=12, 
                    K=13, M=14, F=15, P=16, U=17, S=18, T=19, W=20, Y=21, V=22, B=23, Z=24, X=0
                )

# one-hot 编码
def dna_onehot(Xdna):
    listtmp = list()
    for index, row in Xdna.iterrows():
        row = [prot_dict[x] if x in prot_dict else x for x in row['seq']]
        listtmp.append(row)
    return pd.DataFrame(listtmp)


def lrmain(X_train_std, Y_train, X_test_std, Y_test):
    logreg = linear_model.LogisticRegression(solver = 'lbfgs', n_jobs=-2)
    logreg.fit(X_train_std, Y_train)
    predict = logreg.predict(X_test_std)
    lrpredpro = logreg.predict_proba(X_test_std)
    groundtruth = Y_test
    predictprob = lrpredpro[:,1]
    return groundtruth, predict, predictprob

def svmmain(X_train_std, Y_train, X_test_std, Y_test):
    svcmodel = SVC(probability=True, kernel='rbf', tol=0.001)
    svcmodel.fit(X_train_std, Y_train.ravel(), sample_weight=None)
    predict = svcmodel.predict(X_test_std)
    predictprob =svcmodel.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob[:,1]

    
def xgmain(X_train_std, Y_train, X_test_std, Y_test):
    model = XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, n_jobs=-2, eval_metric='mlogloss')
    model.fit(X_train_std, Y_train.ravel())
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob[:,1]

def dtmain(X_train_std, Y_train, X_test_std, Y_test):
    model = tree.DecisionTreeClassifier()
    model.fit(X_train_std, Y_train.ravel())
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob[:,1]

def rfmain(X_train_std, Y_train, X_test_std, Y_test):
    model = RandomForestClassifier(oob_score=True, random_state=10, n_jobs=-2)
    model.fit(X_train_std, Y_train.ravel())
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob[:,1]

def gbdtmain(X_train_std, Y_train, X_test_std, Y_test):
    model = GradientBoostingClassifier(random_state=10)
    model.fit(X_train_std, Y_train.ravel())
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob[:,1]

def evaluate(baslineName, X_train_std, Y_train, X_test_std, Y_test):

    if baslineName == 'lr':
        groundtruth, predict, predictprob = lrmain (X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName == 'svm':
        groundtruth, predict, predictprob = svmmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='xg':
        groundtruth, predict, predictprob = xgmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='dt':
        groundtruth, predict, predictprob = dtmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='rf':
        groundtruth, predict, predictprob = rfmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='gbdt':
        groundtruth, predict, predictprob = gbdtmain(X_train_std, Y_train, X_test_std, Y_test)
    else:
        print('Baseline Name Errror')

    acc = metrics.accuracy_score(groundtruth, predict)
    precision = metrics.precision_score(groundtruth, predict, zero_division=1 )
    recall = metrics.recall_score(groundtruth, predict)
    f1 = metrics.f1_score(groundtruth, predict)
    auroc = metrics.roc_auc_score(groundtruth, predictprob)
    auprc = metrics.average_precision_score(groundtruth, predictprob)
    tn, fp, fn, tp = metrics.confusion_matrix(groundtruth, predict).ravel()

    npv = tn/(fn+tn+1.4E-45)
    
    print(baslineName, '\t\t%f' %acc,'\t%f'% precision,'\t\t%f'%npv,'\t%f'% recall,'\t%f'% f1, '\t%f'% auroc,'\t%f'% auprc, '\t', 'tp:',tp,'fp:',fp,'fn:',fn,'tn:',tn)

def run_baseline(X_train, Y_train, X_test, Y_test):
    methods=['lr', 'xg', 'dt', 'rf', 'gbdt']
    print('baslineName', '\t', 'accuracy','\t', 'precision(PPV) \t NPV \t\t', 'recall','\t', 'f1', '\t\t', 'auroc','\t\t', 'auprc', '\t\t confusion Matrix')
    for method in methods:
        evaluate(method, X_train, Y_train, X_test, Y_test)
    
    
def static_interval(data, span):
    """[summary]

    Args:
        data ([dataframe]): [需要统计的数据]
        span ([int]): [统计的间隔]

    Returns:
        [type]: [完成的统计列表]]
    """
    res = []
    count = 0
    for i in range(int(len(data)/span) + 1):
        lable = str(i*span) + '-' + str((i+1)* span -1 )
        num = data[(data.length>=(i*span)) & (data.length<(i+1)*span)]['count'].sum()
        res += [[lable, num]]
    return res
        

        
def getblast(train, test):
    """[两组数据进行blast比对]

    Args:
        train ([DataFrame]): [被比对的数据]
        test ([DataFrame]): [比对数据]

    Returns:
        [DataFrame]: [description]
    """
    table2fasta(train, '/tmp/train.fasta')
    table2fasta(test, '/tmp/test.fasta')
    
    cmd1 = r'diamond makedb --in /tmp/train.fasta -d /tmp/train.dmnd'
    cmd2 = r'diamond blastp -d /tmp/train.dmnd  -q  /tmp/test.fasta -o /tmp/test_fasta_results.tsv -b5 -c1 -k 1'
    cmd3 = r'rm -rf /tmp/*.fasta /tmp/*.dmnd /tmp/*.tsv'
    print(cmd1)
    os.system(cmd1)
    print(cmd2)
    os.system(cmd2)
    res_data = pd.read_csv('/tmp/test_fasta_results.tsv', sep='\t', names=['id', 'sseqid', 'pident', 'length','mismatch','gapopen','qstart','qend','sstart','send','evalue','bitscore'])
    os.system(cmd3)
    return res_data


def evaluateBlast(res_data, trainset, testset, matchitem='ec_number'):

    id_map_ec = trainset[['id', matchitem]].append(testset[['id', matchitem]],ignore_index=True)
    id_ec_dict = {v: k for v,k in zip( id_map_ec['id'], id_map_ec[matchitem])} 
    res_data['is_match']=res_data.apply(lambda x: (id_ec_dict.get(x['id'])== id_ec_dict.get(x['sseqid'])), axis=1)


    correct = sum(res_data['is_match'])
    find  = len(res_data)
    total = len(testset)
    print('Total query records are: {0}'.format(total))
    print('Matched records are: {0}'.format(find))
    print('Accuracy: {0}({1}/{2})'.format(correct/total, correct, total))
    print('Pricision: {0}({1}/{2})'.format(correct/find, correct, find))
    print('Recall: {0}({1}/{2})'.format(find/total, find, total))