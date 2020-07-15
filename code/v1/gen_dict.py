# -*- coding: utf-8 -*-
import pickle
import sys
import pandas as pd
import utils

from constants import *

def read_csv(path):
    product_ids = []
    queries = []
    query_ids = []
    with open(path, 'r') as fin:
        header = fin.readline().strip('\n').split('\t')
        for line in fin:
            line = line.strip('\n').split('\t')
            product_ids.append(int(line[0]))
            
            queries.append(utils.get_split_text(line[7]))
            query_ids.append(int(line[8]))
    
    return pd.DataFrame({'product_id':product_ids,'query':queries,'query_id':query_ids})


def add_to_dict(df,query_dict,product_dict,word_dict,train=True):
    if train:
        for query in sorted(set(df['query'])):
            if query not in query_dict:
                query_dict[query] = len(query_dict)+1
            
        for product in sorted(set(df['product_id'])):
            if product not in product_dict: 
                product_dict[product] = len(product_dict) + 1
            
    word_cnt = {}
    for string in df['query']:
        words = utils.split_text(string)
        for word in words:
            if word not in word_cnt:
                word_cnt[word] = 1
            else:
                word_cnt[word] += 1
    
    word_cnt = sorted(word_cnt.items(),key=lambda x:(x[1],x[0]),reverse = True)
        
    for word,i in word_cnt:
        if word not in word_dict:
            word_dict[word] = len(word_dict) + 1
    
    
def get_dict(train_tsv, valid_tsv, testA_tsv,testB_tsv=""):
    df_train = read_csv(train_tsv)
    df_valid = read_csv(valid_tsv)
    df_testA = read_csv(testA_tsv)
    
    if testB_tsv != "":
        df_testB = read_csv(testB_tsv)
   
    
    if testB_tsv != "":
        dfs = [df_train,df_valid,df_testA,df_testB] 
        trains = [True,True,False,False]
    else:
        dfs = [df_train,df_valid,df_testA]
        trains = [True,True,False]
        
    query_dict = {}
    product_dict = {} 
    word_dict = {}
    
    
    for is_train,df in zip(trains,dfs):
        add_to_dict(df,query_dict,product_dict,word_dict,is_train)
    
    print('query_dict',len(query_dict))
    print('product_dict',len(product_dict))
    print('word_dict',len(word_dict))
    # pickle.dump(query_dict, open(query_dict_path, 'wb'))
    # pickle.dump(product_dict, open(product_dict_path, 'wb'))
    pickle.dump(word_dict, open(word_dict_path, 'wb'))


if __name__ == '__main__':
    get_dict(train_tsv,valid_tsv,testA_tsv,testB_tsv)
    
    
    