import pandas as pd
import numpy as np


def get_emzyme_rep(balstpred, slicepred):
    if str(balstpred) == str(slicepred):
        return balstpred
    if (str(balstpred) == 'uncertain') & (str(slicepred)=='True'):
        return slicepred


if __name__ =='__main__':

    data = pd.read_excel('./results/1515/final_pred.xlsx', index_col=0)
    data['emzyme_pred_final']=data[['isemzyme_blast', 'isenzyme_slice_pred']].apply(lambda x: get_emzyme_rep(x[0], x[1]))
    print(data)
    print('ok')