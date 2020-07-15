# -*- coding: utf-8 -*-


import numpy as np
import pickle
from constants import *
import utils

test_query = []
test_product_id = set()
with open(testA_tsv, 'r') as fin:
    header = fin.readline()
    for line in fin:
        features = line.strip('\n').split('\t')
        product_id = features[5]
        test_product_id.add(product_id)
        test_query.append(utils.get_sorted_split_text(features[-2]))

        

product_id_set = set()
train_query = []


# with open(train_tsv, 'r') as fin:
#     header = fin.readline()
#     for line in fin:
#         features = line.strip('\n').split('\t')
#         product_id = features[5]
#         if product_id in test_product_id:
#             product_id_set.add(product_id)
        
#         train_query.append(utils.get_sorted_split_text(features[-2]))

with open(valid_tsv, 'r') as fin:
    header = fin.readline()
    for line in fin:
        features = line.strip('\n').split('\t')
        product_id = features[5]
        if product_id in test_product_id:
            product_id_set.add(product_id)
        train_query.append(utils.get_sorted_split_text(features[-2]))
        


product_ids = sorted(product_id_set)

product_id_dict = {image_id:(i+1) for i,image_id in enumerate(product_ids)}


query_set = set(test_query) & set(train_query)
queries = sorted(query_set)

print('product_id_dict',len(product_id_dict))

query_id_dict = {query:(i+1) for i,query in enumerate(queries)}

print('query_id_dict',len(query_id_dict))

pickle.dump(product_id_dict,open(interid_product_dict_path,'wb'))
pickle.dump(query_id_dict,open(interid_query_dict_path,'wb'))