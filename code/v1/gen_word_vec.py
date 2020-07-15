# -*- coding: utf-8 -*-


from constants import *
import numpy as np
import pickle
import utils
from gensim.models import Word2Vec

np.random.seed(SEED)

def glove_vec():
    word2vec = {}
    with open(glove_src) as fin:
        for line in fin:
            line = line.strip('\n')
            space = line.index(' ')
            word = line[:space]
            vec = line[space+1:]
            vec = np.array(vec.split(' '),dtype='float32')

            word2vec[word] = vec

    word_dict = pickle.load(open(word_dict_path,'rb'))
    not_in_words = []
    vecs = np.zeros([22001,300],dtype='float32')
    i = 0
    for k,v in word_dict.items():
        if k in word2vec:
            vecs[v] = word2vec[k]
        else:
            # vecs[v] = np.random.uniform(-1,1,size=300)
            not_in_words.append(k)
            i += 1



    print('{} words not in dict'.format(i))

    pickle.dump(vecs, open(glove_path,'wb'))

    return vecs


def word2vec():
    query = []
    with open(train_tsv) as fin:
        header = fin.readline()
        for line in fin:
            features = line.strip('\n').split('\t')
            query.append(features[-2])

    valid_test_query = []
    with open(valid_tsv) as fin:
        header = fin.readline()
        for line in fin:
            features = line.strip('\n').split('\t')
            valid_test_query.append(features[-2])

    with open(testA_tsv) as fin:
        header = fin.readline()
        for line in fin:
            features = line.strip('\n').split('\t')
            valid_test_query.append(features[-2])

    query.extend(list(set(valid_test_query)))

    all_query = []
    for q in query:
        all_query.append(utils.split_text(q))

    word2vec = Word2Vec(all_query, size=300, window=7, min_count=1, workers=4, iter=10)

    word_dict = pickle.load(open(word_dict_path,'rb'))
    not_in_words = []
    vecs = np.zeros([22001,300],dtype='float32')
    i = 0
    for k,v in word_dict.items():
        if k in word2vec:
            vecs[v] = word2vec[k]
        else:
            # vecs[v] = np.random.uniform(-1,1,size=300)
            not_in_words.append(k)
            i += 1
            

    print('{} words not in dict'.format(i))
    print('word not in dict:',not_in_words)
    pickle.dump(vecs, open(word2vec_path,'wb'))

    return vecs


def blend():
    word2vec = pickle.load(open(word2vec_path,'rb'))
    glove = pickle.load(open(glove_path,'rb'))
    pickle.dump((word2vec+glove)/2, open(blend_word2vec_path,'wb'))

if __name__ == "__main__":

    word2vec = word2vec()
    glove = glove_vec()
    blend()




