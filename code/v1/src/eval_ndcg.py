"""
@input:
score_file format: qid,pid,score,label
pos_file format: json file, key: qid, value: list of pid
"""
import math
import os

EVAL_NUM = 5
def cal_IDCG(n):
    assert(n >= 1)
    res = 0
    for i in range(1, n + 1):
        res += 1 / math.log(i+1)
    return res
def cal_DCG(hit, k=EVAL_NUM):
    assert(len(hit) == k)
    res = 0
    for idx, h in enumerate(hit):
        res += h / math.log(idx + 2)
    return res
IDCG = {}
for i in range(1, EVAL_NUM + 1):
    IDCG[i] = cal_IDCG(i)


import sys
import pickle
score_file, pos_file, submit_file, query_ids_file = sys.argv[1:]
if query_ids_file != 'all':
    query_ids = pickle.load(open(query_ids_file, 'rb'))
    query_ids = set([q.split(',')[0] for q in query_ids])
else:
    query_ids = None
submit_fout = open(submit_file, 'w')
submit_fout.write('query-id,product1,product2,product3,product4,product5\n')
import json
pos = json.load(open(pos_file, 'r'))
for i in pos:
    pos[i] = set([str(j) for j in pos[i]])

#score = [l.strip().split(',') for l in open(score_file, 'r')]
from collections import defaultdict
predict = defaultdict(list)
for l in open(score_file, 'r'):
    qid, pid, s = l.strip().split(",")
    predict[qid].append([pid, float(s)])

print('[INFO] length of predition: ', len(predict))
print('[INFO] length of groundtruth: ', len(pos))

valid_query_product = None
if os.path.exists('../../../user_data/valid_query_product.pkl'):
    valid_query_product = pickle.load(open('../../../user_data/valid_query_product.pkl','rb'))

ndcg = 0.0
cnt = 0.0
for qid in predict:
    if query_ids_file != 'all':
        if qid not in query_ids: continue
    cnt += 1
    p = sorted(predict[qid], key=lambda x:x [1], reverse=True)[:EVAL_NUM]
    hit = [1 if x[0] in pos[qid] else 0 for x in p]
    if len(hit) < EVAL_NUM: hit = hit + [0 for _ in range(EVAL_NUM - len(hit))]
    dcg = cal_DCG(hit)
    pos_num = len(pos[qid])
    idcg = IDCG[5] if pos_num >= 5 else IDCG[pos_num]
    ndcg += dcg / idcg
    print('[INSPECTION] query is {}, predict products are {}, ndcg is {}'.format(qid, p, dcg / idcg))
    submit_fout.write(qid + ',' + ','.join([x[0] for x in p]) + '\n')

ndcg = ndcg / cnt
print('length of predict is {}'.format(cnt))
print("[INFO] ndcg for file {} is {}".format(score_file, ndcg))

if valid_query_product is not None:
    valid_qids = set(predict.keys())
    train_pids = {pid for qid,pid in valid_query_product if qid not in valid_qids}
    
    
    for kkk in [1]:
        print('current threshold',kkk)
        predict1 = {}
        predict2 = {}
        
        
        for qid,items in predict.items():
            pids = set(item[0] for item in items)
            if len((pids & train_pids))<=kkk:
                predict1[qid] = items
            else:
                predict2[qid] = items
                
        
        ndcg = 0.0
        cnt = 0.0
        for qid in predict1:
            cnt += 1
            p = sorted(predict1[qid], key=lambda x:x [1], reverse=True)[:EVAL_NUM]
            hit = [1 if x[0] in pos[qid] else 0 for x in p]
            if len(hit) < EVAL_NUM: hit = hit + [0 for _ in range(EVAL_NUM - len(hit))]
            dcg = cal_DCG(hit)
            pos_num = len(pos[qid])
            idcg = IDCG[5] if pos_num >= 5 else IDCG[pos_num]
            ndcg += dcg / idcg
    
        ndcg = ndcg / (cnt+1e-9)
        print('length of predict1 is {}'.format(cnt))
        print("[INFO] n_predict1_dcg for file {} is {}".format(score_file, ndcg))
        
        
        ndcg = 0.0
        cnt = 0.0
        for qid in predict2:
            cnt += 1
            p = sorted(predict2[qid], key=lambda x:x [1], reverse=True)[:EVAL_NUM]
            hit = [1 if x[0] in pos[qid] else 0 for x in p]
            if len(hit) < EVAL_NUM: hit = hit + [0 for _ in range(EVAL_NUM - len(hit))]
            dcg = cal_DCG(hit)
            pos_num = len(pos[qid])
            idcg = IDCG[5] if pos_num >= 5 else IDCG[pos_num]
            ndcg += dcg / idcg
    
        ndcg = ndcg / (cnt+1e-9)
        print('length of predict2 is {}'.format(cnt))
        print("[INFO] n_predict2_dcg for file {} is {}".format(score_file, ndcg))
    
    

    predict1 = {}
    predict2 = {}
    predict3 = {}
    predict4 = {}
    for qid,items in predict.items():
        pids = set(item[0] for item in items)
        length = len((pids & train_pids))
        if length<=1:
            predict1[qid] = items
        elif length<= 9:
            predict2[qid] = items
        elif length <= 17:
            predict3[qid] = items
        else:
            predict4[qid] = items
    
    
    preds = [predict1,predict2,predict3,predict4]
    
    for k,pred in enumerate(preds):
        cur_fold = k + 1
        ndcg = 0.0
        cnt = 0.0
        for qid in pred:
            cnt += 1
            p = sorted(pred[qid], key=lambda x:x [1], reverse=True)[:EVAL_NUM]
            hit = [1 if x[0] in pos[qid] else 0 for x in p]
            if len(hit) < EVAL_NUM: hit = hit + [0 for _ in range(EVAL_NUM - len(hit))]
            dcg = cal_DCG(hit)
            pos_num = len(pos[qid])
            idcg = IDCG[5] if pos_num >= 5 else IDCG[pos_num]
            ndcg += dcg / idcg
    
        ndcg = ndcg / (cnt+1e-9)
        print('length of predict_fold_{} is {}'.format(cur_fold,cnt))
        print("[INFO] n_predict_fold_{}_dcg for file {} is {}".format(cur_fold,score_file, ndcg))
    
    

    
    