import pandas as pd
import sys
from collections import defaultdict
import datetime

def fill(res, df):
    for data in df.values:
        q, p, v = data
        q, p = int(q), int(p)
        key = str(q) + ',' + str(p)
        res[key].append(v)
    return res

def mergeScore(input_files, output):
    res = defaultdict(list)
    for file in input_files.split(','):
        print("current file: {}".format(file))
        df = pd.read_csv(file, sep=',', header=None)
        print(df.shape)

        res = fill(res, df)
        print('res length: {}'.format(len(res)))

    with open(output, 'w') as fout:
        for key in res:
            fout.write(key + ',' + str(sum(res[key]) / len(res[key])) + '\n')

if __name__ == '__main__':
    input_files,output = sys.argv[1:]
    mergeScore(input_files, output)