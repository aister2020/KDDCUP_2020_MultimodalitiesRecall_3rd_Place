import pandas as pd
import sys
import os, pickle
from collections import defaultdict
def fill(res, df):
    for data in df.values:
        q, p, v = data
        q, p = int(q), int(p)
        key = str(q) + ',' + str(p)
        res[key].append(v)
    return res

def mergeScore(input_prefix,cv_num, output):
    res = defaultdict(list)
    for i in range(cv_num):
        print("current cv fold: {}".format(i))
        df = pd.read_csv(input_prefix+str(i), sep=',', header=None)
        print(df.shape)

        res = fill(res, df)
        print('res length: {}'.format(len(res)))

    with open(output, 'w') as fout:
        for key in res:
            fout.write(key + ',' + str(sum(res[key]) / len(res[key])) + '\n')

if __name__ == '__main__':
    input_prefix, cv_num = sys.argv[1:]
    cv_num = int(cv_num)
    output = input_prefix + 'merged'
    mergeScore(input_prefix,cv_num, output)