# 1. 定义数据目录
DATADIR = r'''/home/shizhenkun/codebase/BioUniprot/data/benchmark/data/'''
RESULTSDIR = r'''/home/shizhenkun/codebase/BioUniprot/data/benchmark/results/'''
MODELDIR = r'''/home/shizhenkun/codebase/BioUniprot/data/benchmark/model'''


TRAIN_FEATURE = DATADIR+'train.feather'
TEST_FEATURE = DATADIR+'test.feather'
TRAIN_FASTA = DATADIR+'train.fasta'
TEST_FASTA = DATADIR+'test.fasta'
FILE_SLICE_TRAINX = DATADIR + 'slice_train_x.txt'
FILE_SLICE_TRAINY  = DATADIR + 'slice_train_y.txt'
FILE_SLICE_TESTX = DATADIR + 'slice_test_x.txt'
FILE_SLICE_TESTY  = DATADIR + 'slice_test_y.txt'
FILE_EC_LABEL_DICT = DATADIR + 'ec_label_dict.npy'

FILE_TRANSFER_DICT = DATADIR + 'ec_transfer_dict.npy'


ISENZYME_MODEL = MODELDIR+'/isenzyme.model'
HOWMANY_MODEL = MODELDIR+'/howmany_enzyme.model'
ISENZYME_DB_MODEL = MODELDIR + '/isenzyme.dmnd' #是否是酶blast 数据库
EC_DB_MODEL = MODELDIR + 'ec.dmnd'              #EC号blast数据库


FILE_BLAST_RESULTS = RESULTSDIR + r'test_blast_res.tsv'
FILE_SLICE_RESULTS = RESULTSDIR + 'slice_results.txt'
FILE_INTE_RESULTS  =   RESULTSDIR+'slice_pred.tsv'
FILE_DEEPEC_RESULTS = RESULTSDIR + r'deepec/DeepEC_Result.txt'
FILE_ECPRED_RESULTS = RESULTSDIR + r'ecpred/ecpred.tsv'

FILE_EVL_RESULTS = RESULTSDIR + r'evaluation_table.xlsx'

UPDATE_MODEL = True #强制模型更新标志
VALIDATION_RATE = 0.1 #模型训练时验证集的比例
FEATURE_NUM = 1900

# 训练参数
SAMPLING_BIT = 6 #采样精度
# SLICE
TRAIN_USE_ONLY_SINGLE_FUNCTION = True #只用单功能酶
TRAIN_USE_SPCIFIC_EC_LEVEL = 3  #训练用ec级别大于n位的
TRAIN_USE_ONLY_ENZYME = True    #只用酶数据进行训练
TRAIN_BLAST_IDENTITY_THRES = 20 #比对结果identity阈值


BLAST_TABLE_HEAD = ['id', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send',
                    'evalue', 'bitscore']