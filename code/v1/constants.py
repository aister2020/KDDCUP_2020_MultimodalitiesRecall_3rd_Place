# -*- coding: utf-8 -*-
import os

data_dir = '../../data/'
input_dir = '../../user_data/v1/'

train_tsv = data_dir+'train/train.tsv'
valid_tsv = data_dir+'valid/valid.tsv'
testA_tsv = data_dir+'testB/testB.tsv'
testB_tsv = ''

external_resources_dir = '../../external_resources/'

if not os.path.exists(input_dir):
    os.makedirs(input_dir)

valid_answer = data_dir+'valid/valid_answer.json'

glove_src = external_resources_dir+'glove.42B.300d.txt'

glove_path = input_dir+'glove.pkl'
word2vec_path = input_dir+'word2vec.pkl'
blend_word2vec_path = input_dir+'blend_word2vec.pkl'


ndcg_valid_tfrecord = input_dir+'officialValid.tfrecord'
ndcg_testA_tfrecord = input_dir+'officialTestA.tfrecord'


word_dict_path = input_dir+'word_dict.pkl'
# query_dict_path = input_dir+'query_dict.pkl'
# product_dict_path = input_dir+'product_dict.pkl'

interid_product_dict_path = input_dir+'interid_product_dict.pkl'
interid_query_dict_path = input_dir+'interid_query_dict.pkl'

# trichar_dict_path = input_dir+'trichar_dict.pkl'
# trichar2vec_path = input_dir+'trichar2vec.pkl'


word2cluster_path = input_dir+'word2cluster_{}.pkl'


valid_query_prodcut_pkl = input_dir+'valid_query_product.pkl'



SEED = 2020

THREAD = 8