import pandas as pd
import numpy as np


def table2fasta(table, file_out):
    file = open(file_out, 'w')
    for index, row in table.iterrows():
        file.write('>{0}\n'.format(row['id']))
        file.write('{0}\n'.format(row['seq']))
    file.close()
    print('Write finished')
