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

from cv_lgb_train import create_sample,create_dataframe

if __name__ == "__main__":
    if len(sys.argv[1:])>0:
        lgb_model, vec_input_date, lambdarank,with_productid,use_category, seed = sys.argv[1:]

    print('lgb_model',lgb_model)
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
    
    lgb_model_dir = '../../user_data/lgb_model/'
    output_dir = '../../user_data/output/testA_lgb/'
    
    if not os.path.exists(lgb_model_dir):
        os.makedirs(lgb_model_dir)
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    word_dict_file = word_dict_path
    query_dict_file = interid_query_dict_path
    product_dict_file = interid_product_dict_path
    
    word_dict = pickle.load(open(word_dict_file, 'rb'))
    product_dict = pickle.load(open(product_dict_file, 'rb'))
    query_dict = pickle.load(open(query_dict_file, 'rb'))
    cv_fold = 5
    
    wv = pickle.load(open(blend_word2vec_path,'rb'))
    
    all_test_example = []
    
    test_qidpid_vecs = pickle.load(open("../../user_data/output/testA_vec/submit_{}".format(vec_input_date),'rb'))
    
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
    for i in range(cv_fold):
    
        lgb_model_path = '../../user_data/lgb_model/{}.model_cv{}'.format(lgb_model,i) 
        model = lgb.Booster(model_file=lgb_model_path)
        
        pred_test = model.predict(X_test)
        all_preds.append(pred_test)
        
    
    test_pred = np.array(all_preds).mean(axis=0)
    qid = df_test['origin_query_id'].values
    pid = df_test['origin_product_id'].values
    
    out_file_path = output_dir+"lgb{}".format(lgb_model)
    
    with open(out_file_path,'w') as out_file:
        for idx in range(len(test_pred)):
            ctr = test_pred[idx]
            out_file.write(
                str(qid[idx]) + ',' + str(pid[idx]) + ',' + str(ctr) + "\n")