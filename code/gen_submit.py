"""
@input:
score_file format: qid,pid,score,label
pos_file format: json file, key: qid, value: list of pid
"""
import math
EVAL_NUM = 5
import sys
score_file, submit_file = sys.argv[1:]
submit_fout = open(submit_file, 'w')
submit_fout.write('query-id,product1,product2,product3,product4,product5\n')
import json
from collections import defaultdict
predict = defaultdict(list)
for l in open(score_file, 'r'):
    qid, pid, s = l.strip().split(",")
    predict[qid].append([pid, float(s)])

print('[INFO] length of predition: ', len(predict))

for qid in predict:
    p = sorted(predict[qid], key=lambda x:x [1], reverse=True)[:EVAL_NUM]
    submit_fout.write(qid + ',' + ','.join([x[0] for x in p]) + '\n')
print("[INFO] gen submit done.")