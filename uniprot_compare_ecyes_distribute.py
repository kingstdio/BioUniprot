import numpy as np
import pandas as pd
import os
import subprocess
import sys


def run(programid):
    counter =0
    query_data = np.loadtxt('./data/temp/sub_queryed_'+programid,dtype=str, delimiter='\t')
    query_data = np.array(query_data)
    subprocess = programid

    filtered_data = []
    for line in query_data:
        #排除第一行
        if counter == 0:
            counter +=1
            continue
        #写零时查询文件
        query_file = open('./data/temp/query.fasta','w')
        query_file.write('>{0}\n'.format(line[0]))
        query_file.write('{0}'.format(line[2]))
        query_file.close()

        #进行比对
        os.system('./diamond blastp -d ./data/ecyes_reference.dmnd -q ./data/temp/query.fasta -o ./data/temp/query_result.tsv -k 1 -b5 -c1 --quite')

        result_file = np.loadtxt('./data/temp/query_result.tsv',dtype=str, delimiter='\t')
        result_data = np.array(result_file)

        # 没有比对到结果
        if len (result_data) < 1:
            continue

        # 比对到结果进行保留判断    
        seq_length = len(line[2])
        inentity =  round(float(result_data[2]),4)
        coverage =  round(100*float(result_data[3])/seq_length,4)
        if 50.0 < inentity < 99.0 and 70.0 < coverage:
            line = np.append(line, [inentity, coverage])
            filtered_data.append(line)

        counter+=1
        if counter%200==0:
            print(counter)

    #保存文件
    # np.savetxt("./results/query/uniprot_trembl_ecraw_2_queryed_"+programid+".tsv", np.array(filtered_data), delimiter="\t", fmt='%s')

#region 拆分文件
def splitDataSet(file_in, linenum):
    """[按行拆分文件]

    Args:
        file_in ([str]): [需要拆分的文件]
        linenum ([int]): [每个文件包含的数据行数]
    """
    counter =0
    infile = open(file_in,'r')
    inlines = infile.readlines()
    infile.close()

    filenum = 1
    temparray = []
    for line in inlines:
        if counter == 0:
            counter +=1
            continue
        temparray.append(line)
        counter +=1
        if counter%linenum == 0:
            outfile = open("./data/temp/sub_queryed_"+str(filenum)+".tsv", 'w')
            outfile.writelines(temparray)
            outfile.close()
            temparray = []
            filenum +=1

    outfile = open("./data/temp/sub_queryed_"+str(filenum)+".tsv", 'w')
    outfile.writelines(temparray)
    outfile.close()
    print("split finished, total {} files".format(filenum))
#endregion

#region 将列表文件转化为fasta格式
def uniprot_records_to_fasta(infile, outfile):
    """[将列表文件转化为fasta格式]

    Args:
        infile ([type]): [输入文件]
        outfile ([type]): [输出文件]
    """

    lines = open(infile,'r')
    inlines = lines.readlines()
    fasta_file = open(outfile,'w')
    for line in inlines:
        linearray = line.split('\t')
        fasta_file.write('>{0}\n'.format(linearray[0]))
        fasta_file.write('{0}\n'.format(linearray[2]))
    fasta_file.close()
#endregion

#region 按行拆分文件并将其转化为fasta格式
def splitDataSet_and_2fasta(file_in, linenum):
    """[按行拆分文件并将其转化为fasta格式]

    Args:
        file_in ([str]): [需要拆分的文件]
        linenum ([int]): [每个文件包含的数据行数]
    """
    counter =0
    infile = open(file_in,'r')
    inlines = infile.readlines()
    infile.close()

    filenum = 1
    temparray = []

    fasta_file = open("./temp/sub_queryed_"+str(filenum)+".fasta",'w')
    
    for line in inlines:
        counter+=1

        linearray = line.split('\t')
        fasta_file.write('>{0}\n'.format(linearray[0]))
        fasta_file.write('{0}'.format(linearray[2]))

        if counter%linenum == 0:
            filenum +=1
            fasta_file.close()
            fasta_file = open("./temp/sub_queryed_"+str(filenum)+".fasta",'w')
            
    fasta_file.close()    
    print("split and convert finished, total {} files".format(filenum))    
#endregion

def diamondCompare(infile, outfile):
    os.system('./diamond blastp -d ./data/ecyes_reference.dmnd -q '+infile+' -o '+outfile+' -k 1 -b5 -c1')


def get_data_set_from_compared_results(file_in, file_out):
    infile = open(file_in,'r')
    inlines = infile.readlines()
    infile.close()
    res_array =[]
    counter =0
    for line in inlines:
        counter +=1
        linearray = line.split('\t')
        
        
        inentity =  float(linearray[2])
        # coverage =  round(100*float(result_data[3])/seq_length,4)
        if 50.0 < inentity < 99.0:
            res_array.append(line)
        if counter %200000 ==0:
            print(counter)
    np.savetxt(file_out, np.array(res_array), delimiter="\t", fmt='%s')

def match_flile(file_base, file_filter):
    base_data = pd.read_csv(file_base,sep='\t')
    print("bigf")
    filter_data = pd.read_csv(file_filter, sep='\t', names=['id', 'sseqid', 'pident', 'length','mismatch','gapopen','qstart','qend','sstart','send','evalue','bitscore'])
    print('smallf')
    res = pd.merge(base_data,filter_data, on=['id'], how='inner')
    res.to_csv('./results/uniprot_trembl_ecraw_matched.tsv',index=False, sep='\t')

def filter_coverage(file_in,file_out):
    base_data = pd.DataFrame(pd.read_csv(file_in,sep='\t'))
    
    resArray =[]
    for index, row in base_data.iterrows():
        length = len(row['seq'])
        ak = float(row['C'])
        coverage = ak/length

        if coverage>0.7:
           resArray.append(np.array(row))     
        
        if index %200000 ==0:
            print('{0}/{1}'.format(len(resArray),index))

    res = pd.DataFrame(resArray)
    res.to_csv(file_out,index=False, sep='\t')


def concat_2_data():
    data = pd.read_csv('./results/uniprot_trembl_ecraw_compared.tsv', sep='\t', names=['id', 'sseqid', 'pident', 'length','mismatch','gapopen','qstart','qend','sstart','send','evalue','bitscore'])
    rawdata = pd.read_csv('./results/uniprot_trembl_ecraw.tsv', sep='\t')

    data1 = data[data['pident']<99.0]
    data1 = data1[data1['pident']>50]

    rawdata['seqlength'] = rawdata['seq'].map(lambda x : len(x))
    print('add seqlength finished')
    counter = 0
    res_array = []
    for index, row in data1.iterrows():
        
        res = rawdata[rawdata['id'] == row['id']]
        res = res.iloc[0].values
        res = np.append(res, row.values).flatten()
        res_array.append(res)
        
        counter +=1
        if counter == 10000:
            print(counter)
            break
    con_data = np.array(res_array)
    np.savetxt("./results/query/uniprot_trembl_ecraw_concanated.tsv", con_data, delimiter="\t", fmt='%s')

if __name__ =="__main__":
    LINENUM = 300000
    # file_in = './results/uniprot_trembl_ecraw_2.tsv'
    # run(str(sys.argv[1]))
    # splitDataSet(file_in, LINENUM)
    # uniprot_records_to_fasta('./data/temp/sub_queryed_1.tsv', './manualQuerytest.fasta')
    # splitDataSet_and_2fasta('results/ecraw.tsv', 100)
    # uniprot_records_to_fasta('results/uniprot_trembl_ecraw_2.tsv','results/uniprot_trembl_ecraw_2.fasta')

    # get_data_set_from_compared_results('./results/uniprot_trembl_ecraw_compared.tsv', './results/uniprot_trembl_ecraw_compared_filtered.tsv')
    # match_flile('./results/uniprot_trembl_ecraw.tsv','./results/uniprot_trembl_ecraw_compared_filtered.tsv')
    # filter_coverage('./results/uniprot_trembl_ecraw_matched.tsv', './results/uniprot_trembl_ecraw_final_full.tsv')
    # filter_coverage('results/uniprot_trembl_ecraw_split_00', './results/uniprot_trembl_ecraw_final_full.tsv')

    # concat_2_data()


    splitDataSet_and_2fasta('data/sprot_with_ec.tsv', 135118)
    
