import numpy as np
import pandas as pd
import os
import subprocess
import sys


def run(programid):
    counter =0
    query_data = np.loadtxt('./results/uniprot_trembl_ecraw_split_'+programid,dtype=str, delimiter='\t')
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
        os.system('./diamond blastp -d ./data/ecyes_reference.dmnd -q ./data/temp/query.fasta -o ./data/temp/query_result.tsv -k 1 -b5 -c1')

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
    np.savetxt("./results/query/uniprot_trembl_ecraw_2_queryed_"+programid+".tsv", np.array(filtered_data), delimiter="\t", fmt='%s')

def splitDataSet():
    counter =0
    infile = open('./results/uniprot_trembl_ecraw_2.tsv','r')
    inlines = infile.readlines()
    infile.close()
    counter =0
    filecounter =0
    temparray =[]
    for line in inlines:
        temparray.append(line)
        counter +=1
        if counter >=10000:
            np.savetxt("./results/uniprot_trembl_ecraw_2_tmp_"+filecounter+".tsv", np.array(temparray), delimiter="\t", fmt='%s')
            counter =0
            filecounter +=1
            temparray = []
            print(filecounter)
    print('success')



    

if __name__ =="__main__":
    run(str(sys.argv[1]))
    # splitDataSet()
