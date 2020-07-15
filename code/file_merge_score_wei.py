import pandas as pd
import sys
from collections import defaultdict
import datetime
import numpy as np

def fill(res, df, wei):
    for data in df.values:
        q, p, v = data
        q, p = int(q), int(p)
        key = str(q) + ',' + str(p)
        res[key].append(v*wei)
    return res

def mergeScore(input_files,wei, output):
    res = defaultdict(list)
    
    wei = np.array([float(i) for i in wei.split(',')])
    wei = wei/wei.sum()
    
    for i,file in enumerate(input_files.split(',')):
        print("current file: {}".format(file))
        df = pd.read_csv(file, sep=',', header=None)
        print(df.shape)

        res = fill(res, df, wei[i])
        print('res length: {}'.format(len(res)))

    with open(output, 'w') as fout:
        for key in res:
            fout.write(key + ',' + str(sum(res[key])) + '\n')

if __name__ == '__main__':
    input_files,wei,output = sys.argv[1:]
    mergeScore(input_files, wei, output)