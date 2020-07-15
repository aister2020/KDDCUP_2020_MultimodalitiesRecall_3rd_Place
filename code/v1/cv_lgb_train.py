# -*- coding: utf-8 -*-


import numpy as np
import pickle
from constants import *
import time
import json
import sys
import os
import utils
import base64
import generate_tf_record
import pandas as pd
import lightgbm as lgb
from datetime import datetime
import math
from collections import defaultdict

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


def create_sample(features,qidpid_vecs,wv):
    features_id = [features[0], features[5], features[1],features[6], features[7], features[8], features[9], features[3], features[4]]
    
    qid_pid_vec = qidpid_vecs[(features[0], features[5])]
    features_vec = np.concatenate([np.array([wv[i] for i in features[2]]).mean(axis=0), \
                                   np.array(features[10]).reshape(-1,4).mean(axis=0), np.array(features[11]).reshape(-1,2048).mean(axis=0),\
                                np.array(qid_pid_vec[0]),
                                qid_pid_vec[1],qid_pid_vec[2],
                                qid_pid_vec[3],qid_pid_vec[4]
                                   
                                   
                                ])
    
    return features_id,features_vec
        
    
def create_dataframe(example):
    columns1 = ["origin_query_id","origin_product_id","query_id","product_id","image_h","image_w","num_boxes","words_len","lastword"]
    
    feas = [i[0] for i in example]
    fea_vecs = [i[1] for i in example]
    
    columns2 = ["vec{}".format(i) for i in range(fea_vecs[0].shape[0])]
    
    df1 = pd.DataFrame(np.stack(feas),columns=columns1)
    df2 = pd.DataFrame(np.stack(fea_vecs),columns=columns2)
    
    df = pd.concat([df1,df2],axis=1)
    return df



if __name__ == "__main__":
    if len(sys.argv[1:])>0:
        vec_input_date, lambdarank,with_productid,use_category, seed = sys.argv[1:]

    
    print('vec_input_date',vec_input_date)    
    print('lambdarank',lambdarank)
    print('with_productid',with_productid)
    print('use_category',use_category)
    print('seed',seed)
    
    lambdarank = int(lambdarank)
    with_productid = int(with_productid)
    use_category = int(use_category)
    seed = int(seed)
    
    np.random.seed(seed)
    
    
    feat_imp_dir = '../user_data/feat_imp/'
    lgb_model_dir = '../user_data/lgb_model/'
    output_dir = 'training/output/testA_lgb/'
    valid_output_dir = 'training/output/lgb_prediction/'
    
    
    if not os.path.exists(feat_imp_dir):
        os.makedirs(feat_imp_dir)
    
    if not os.path.exists(lgb_model_dir):
        os.makedirs(lgb_model_dir)
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(valid_output_dir):
        os.makedirs(valid_output_dir)
    
    version = datetime.now().strftime("%m%d%H%M%S")
    
    
    word_dict_file = word_dict_path
    query_dict_file = interid_query_dict_path
    product_dict_file = interid_product_dict_path
    
    
    answer_dict = json.load(open(valid_answer, 'r'))
    answer_set = set()
    for key, value in answer_dict.items():
        for pid in value:
            answer_set.add((key, pid))

    
    qidpid_vecs = pickle.load(open("training/output/prediction_vec_{}/prediction_{}".format(vec_input_date,vec_input_date),'rb'))
    from collections import defaultdict
    predict = defaultdict(list)
    for qid,pid in qidpid_vecs:
        
        predict[qid].append([pid, qidpid_vecs[(qid,pid)][0]])
    
    ndcgs = []
    ndcg = 0.0
    cnt = 0.0
    for qid in predict:
        cnt += 1
        p = sorted(predict[qid], key=lambda x:x [1], reverse=True)[:EVAL_NUM]
        hit = [1 if x[0] in answer_dict[str(qid)] else 0 for x in p]
        if len(hit) < EVAL_NUM: hit = hit + [0 for _ in range(EVAL_NUM - len(hit))]
        dcg = cal_DCG(hit)
        pos_num = len(answer_dict[str(qid)] )
        idcg = IDCG[5] if pos_num >= 5 else IDCG[pos_num]
        ndcg += dcg / idcg
        
    ndcg = ndcg / cnt
    
    print('length of predict is {}'.format(cnt))
    print("[INFO] ndcg is {}".format(ndcg))
    ndcgs.append(ndcg)
    print('ndcg mean',np.mean(ndcgs))
    
    word_dict = pickle.load(open(word_dict_file, 'rb'))
    product_dict = pickle.load(open(product_dict_file, 'rb'))
    query_dict = pickle.load(open(query_dict_file, 'rb'))
    cv_fold = 5
    
    keys = list(answer_dict.keys())
    np.random.shuffle(keys)
    split_num = int(len(keys)/cv_fold)
    
    
    
    
    
    wv = pickle.load(open(blend_word2vec_path,'rb'))
    
    
    
    all_test_example = []
    
    test_qidpid_vecs = pickle.load(open("training/output/testA_vec/submit_{}".format(vec_input_date),'rb'))
    
    with open(testA_tsv, 'r') as fin:
        header = fin.readline()
        for line in fin:
            
            features = line.strip('\n').split('\t')
            
            features = generate_tf_record.parseTrainFileLine(features, product_dict, query_dict, word_dict, )
                
            example = create_sample(features,test_qidpid_vecs,wv)
            all_test_example.append(example)
    
    df_test = create_dataframe(all_test_example)
    
    if with_productid==1:
        drop_cols = ["origin_query_id","origin_product_id","query_id"]
        categories = ["product_id"]
    else:
        drop_cols = ["origin_query_id","origin_product_id","query_id","product_id"]
        categories = []
        
    if use_category != 1:
        categories = []
        
        
    X_test = df_test.drop(drop_cols,axis=1)
    all_preds = []
    
    
    
    
    def modeling(train_X, train_Y,train_group, test_X, test_Y,test_gorup, categoricals, mode, OPT_ROUNDS=600):
        
        EARLY_STOP = 100
        OPT_ROUNDS = OPT_ROUNDS
        MAX_ROUNDS = 1000
        params = {
            'boosting': 'gbdt',
            # 'metric' : 'binary_logloss',
            'metric' : ['ndcg'],
            'objective': 'binary',
            'learning_rate': 0.02,
            'max_depth': -1,
            'min_child_samples': 20,
            'max_bin': 255,
            'subsample': 0.85,
            'subsample_freq': 10,
            'colsample_bytree': 0.8,
            'min_child_weight': 0.001,
            'subsample_for_bin': 200000,
            'min_split_gain': 0,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'num_leaves':63,
            'seed': seed,
            'nthread': 16,
            'scale_pos_weight': 1,
            'eval_at': 5
            #'is_unbalance': True,
        }
        
        if lambdarank == 1:
            params['objective'] = "lambdarank"
        
        print(params)
        print('Now Version {}'.format(version))
        if mode == 'valid':
            print('Start train and validate...')
            print('feature number:', len(train_X.columns))
            feat_cols = list(train_X.columns)
            
            dtrain = lgb.Dataset(data=train_X, label=train_Y, feature_name=feat_cols, group=train_group)
            dvalid = lgb.Dataset(data=test_X, label=test_Y, feature_name=feat_cols,group=test_gorup )
            model = lgb.train(params,
                              dtrain,
                              categorical_feature=categoricals,
                              num_boost_round=MAX_ROUNDS,
                              early_stopping_rounds=EARLY_STOP,
                              verbose_eval=50,
                              valid_sets=[dvalid],
                              valid_names=['valid']
                              )
            importances = pd.DataFrame({'features':model.feature_name(),
                                    'importances':model.feature_importance()})
            importances.sort_values('importances',ascending=False,inplace=True)
            print(importances)
            importances.to_csv( (feat_imp_dir+'{}_imp.csv').format(version), index=False )
            return model
        else:
            print('Start training... Please set OPT-ROUNDS.')
            feat_cols = list(train_X.columns)
            dtrain = lgb.Dataset(data=train_X, label=train_Y, feature_name=feat_cols, group=train_group)
            print('feature number:', len(train_X.columns))
            print('feature :', train_X.columns)
            model = lgb.train(params,
                              dtrain,
                              categorical_feature=categoricals,
                              num_boost_round=OPT_ROUNDS,
                              verbose_eval=50,
                              valid_sets=[dtrain],
                              valid_names='train'
                              )
            
            importances = pd.DataFrame({'features':model.feature_name(),
                                    'importances':model.feature_importance()})
            importances.sort_values('importances',ascending=False,inplace=True)
            importances.to_csv( (feat_imp_dir+'{}_imp.csv').format(version), index=False )
            
            
            return model
    
    all_valid_preds = []
    
    ndcgs = []
    
    for i in range(cv_fold):
    
        if i == cv_fold-1:
            valid_idx = keys[i*split_num:]
        else:
            valid_idx = keys[i * split_num:(i+1) * split_num]
    
        train_idx = set(keys) - set(valid_idx)
        train_keys = train_idx
        print('train keys len',len(train_keys))
        
        all_example = []
        pos = 0
        neg = 0
        train_num = 0
        valid_num = 0
        idx = 0
        
        with open(valid_tsv, 'r') as fin:
            header = fin.readline()
            for line in fin:
                features = line.strip('\n').split('\t')
                if (features[-1], int(features[0])) in answer_set:
                    features = features + [1]
                    pos += 1
                else:
                    features = features + [0]
                    neg += 1
                all_example.append(features)
                idx += 1
    
        np.random.shuffle(all_example)
        train_example = []
        train_label = []
        valid_example = []
        valid_label = []
        
        for features in all_example:
            if features[-2] in train_keys:
                train_label.append(features[-1])
                features = generate_tf_record.parseTrainFileLine(features[:-1], product_dict, query_dict, word_dict, )
                
                example = create_sample(features, qidpid_vecs,wv)
                train_example.append(example)
                
                train_num += 1
            else:
                valid_label.append(features[-1])
                features = generate_tf_record.parseTrainFileLine(features[:-1], product_dict, query_dict, word_dict, )
                
                example = create_sample(features, qidpid_vecs,wv)
                valid_example.append(example)
               
                valid_num += 1
                
    
        print("pos num:{},neg num:{}, train num:{},valid num:{}".format(pos, neg,train_num,valid_num))
        
        df_train = create_dataframe(train_example)
        df_train['label'] = train_label
        df_valid = create_dataframe(valid_example)
        df_valid['label'] = valid_label
        
        df_train = df_train.sort_values("origin_query_id")
        df_valid = df_valid.sort_values("origin_query_id")
        
        train_label = df_train.pop('label')
        valid_label = df_valid.pop('label')
        
        train_group = df_train["origin_query_id"].value_counts().sort_index().values
        valid_group = df_valid["origin_query_id"].value_counts().sort_index().values
        
        
        train_X = df_train.drop(drop_cols,axis=1)
        valid_X = df_valid.drop(drop_cols,axis=1)
        model = modeling(train_X, train_label,train_group , valid_X, valid_label,valid_group, categories, "valid")
        
        pred = model.predict(valid_X)
        model.save_model( lgb_model_dir+'{}.model_cv{}'.format(version,i) )
        
        pred_test = model.predict(X_test)
        all_preds.append(pred_test)
        
        
        
        predict = defaultdict(list)
        for qid, pid, s in zip(df_valid['origin_query_id'].values,df_valid['origin_product_id'],pred):
            predict[qid].append([pid, float(s)])
            all_valid_preds.append([qid,pid,s])
            
        ndcg = 0.0
        cnt = 0.0
        for qid in predict:
            cnt += 1
            p = sorted(predict[qid], key=lambda x:x [1], reverse=True)[:EVAL_NUM]
            hit = [1 if x[0] in answer_dict[str(qid)] else 0 for x in p]
            if len(hit) < EVAL_NUM: hit = hit + [0 for _ in range(EVAL_NUM - len(hit))]
            dcg = cal_DCG(hit)
            pos_num = len(answer_dict[str(qid)] )
            idcg = IDCG[5] if pos_num >= 5 else IDCG[pos_num]
            ndcg += dcg / idcg
            
        ndcg = ndcg / cnt
        
        print('length of predict is {}'.format(cnt))
        print("[INFO] ndcg is {}".format(ndcg))
        ndcgs.append(ndcg)
        
    print('ndcgs',ndcgs)
    print('ndcg mean',np.mean(ndcgs))
    
    test_pred = np.array(all_preds).mean(axis=0)
    qid = df_test['origin_query_id'].values
    pid = df_test['origin_product_id'].values
    
    
    valid_out_file_path = valid_output_dir+"prediction_{}".format(version)
    with open(valid_out_file_path,'w') as out_file:
        for idx in range(len(all_valid_preds)):
            cur = all_valid_preds[idx]
            out_file.write(','.join([str(j) for j in cur]) + '\n')
    
    
    out_file_path = output_dir+"lgb{}".format(version)
    
    with open(out_file_path,'w') as out_file:
        for idx in range(len(test_pred)):
            ctr = test_pred[idx]
            out_file.write(
                str(qid[idx]) + ',' + str(pid[idx]) + ',' + str(ctr) + "\n")

