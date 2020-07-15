import pandas as pd
import numpy as np
import base64
import tensorflow as tf
import sys
import utils
import pickle
from utils import read_large_file
from constants import *
from functools import reduce

import codecs
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def append_p_idx(s, i):
    return s + '_' + str(i)



def cover(range1,range2):
    left = max(range1[0],range2[0])
    right = min(range1[1],range2[1])

    if right>left:
        return right-left
    else:
        return 0

def parseTrainFileLine(features, product_dict, query_dict, word_dict):
    if features[5] in product_dict:
        product_id = product_dict[features[5]]
    else:
        product_id = 0
    text = utils.get_sorted_split_text(features[7])
    if text in query_dict:
        query_id = query_dict[text]
    else:
        query_id = 0
    words = [word_dict[x] if x in word_dict else 0 for x in utils.split_text(text)]
    
    words = words[-100:]
    res = [int(features[-1]), int(query_id), words, len(words), words[-1],
           int(features[0]), product_id, int(features[1]), int(features[2]), int(features[3]), np.frombuffer(base64.b64decode(features[4]), dtype=np.float32).reshape(-1).tolist(),
           np.frombuffer(base64.b64decode(features[5]), dtype=np.float32).reshape(-1).tolist(), np.frombuffer(base64.b64decode(features[6]), dtype=np.int64).reshape(-1).tolist()]


    height= res[7]
    width = res[8]
    num_box = res[9]
    boxes = res[10]

    split = 3
    image_area = int(np.sqrt((height*width))/10)
    height_slice = [i*height/split for i in range(split+1)]
    width_slice = [i*width/split for i in range(split+1)]

    position = []
    hei = []
    wid = []
    area = []
    for i in range(num_box):
        t,l,b,r = boxes[(i*4):(i+1)*4]

        hei_cover = []
        for j in range(len(height_slice)-1):
            range1 = (t,b)
            range2 = (height_slice[j],height_slice[j+1])
            hei_cover.append(cover(range1,range2))

        wid_cover = []
        for j in range(len(width_slice)-1):
            range1 = (l,r)
            range2 = (width_slice[j],width_slice[j+1])
            wid_cover.append(cover(range1,range2))


        hei_max = np.argmax(hei_cover)
        wei_max = np.argmax(wid_cover)

        position.append(int(hei_max*split+wei_max))

        hei.append(int((b-t)/5))
        wid.append(int((r-l)/5))

        area.append(int((np.sqrt((b-t)*(r-l)))/10))

    res[7] = int(res[7]/10)
    res[8] = int(res[8]/10)
    res += [image_area,position,hei,wid,area]
    return res

def create_tf_example(features, product_dict, query_dict, word_dict, mode=1, label=None, extra_preds=None, sample_type=None):
    features = parseTrainFileLine(features, product_dict, query_dict, word_dict)
    fd = {}

    if label is not None:
        fd['peudo-label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    
    if extra_preds is not None:
        fd['extra_preds'] = tf.train.Feature(float_list=tf.train.FloatList(value=extra_preds))
    
    if sample_type is not None:
        fd['sample_type'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[sample_type]))
    
    fd['ori_query_id'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(features[0])]))
    fd['query_id'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(features[1])]))
    fd['query'] = tf.train.Feature(int64_list=tf.train.Int64List(value=features[2]))
    fd['query_words_num'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[features[3]]))
    fd['last_word'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[features[4]]))
    for i in range(mode):
        offset = i * 0 + 6
        fd[append_p_idx('ori_product_id', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(features[offset - 1])]))
        fd[append_p_idx('product_id', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(features[offset + 0])]))
        fd[append_p_idx('height', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(features[offset + 1])]))
        fd[append_p_idx('width', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(features[offset + 2])]))
        fd[append_p_idx('num_boxes', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(features[offset + 3])]))
        fd[append_p_idx('boxes', i)] = tf.train.Feature(float_list=tf.train.FloatList(value=features[offset + 4]))
        fd[append_p_idx('boxes_features', i)] = tf.train.Feature(float_list=tf.train.FloatList(value=features[offset + 5]))
        fd[append_p_idx('boxes_labels', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=features[offset + 6]))
        fd[append_p_idx('image_area', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=[features[offset + 7]]))
        fd[append_p_idx('boxes_position', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=features[offset + 8]))
        fd[append_p_idx('boxes_height', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=features[offset + 9]))
        fd[append_p_idx('boxes_width', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=features[offset + 10]))
        fd[append_p_idx('boxes_area', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=features[offset + 11]))
    tf_example = tf.train.Example(features=tf.train.Features(feature=fd))
    return tf_example



def create_tf_example_list(features, product_dict, query_dict, word_dict, mode):
    all_features = [parseTrainFileLine(feature, product_dict, query_dict, word_dict) for feature in features]
    fd = {}

    
    features = reduce(lambda x,y: x+y,all_features)
    fd['peudo-label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
    fd['ori_query_id'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(features[0])]))
    fd['query_id'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(features[1])]))
    fd['query'] = tf.train.Feature(int64_list=tf.train.Int64List(value=features[2]))
    fd['query_words_num'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[features[3]]))
    fd['last_word'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[features[4]]))
    for i in range(mode):
        offset = i * len(all_features[0]) + 6
        fd[append_p_idx('ori_product_id', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(features[offset - 1])]))
        fd[append_p_idx('product_id', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(features[offset + 0])]))
        fd[append_p_idx('height', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(features[offset + 1])]))
        fd[append_p_idx('width', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(features[offset + 2])]))
        fd[append_p_idx('num_boxes', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(features[offset + 3])]))
        fd[append_p_idx('boxes', i)] = tf.train.Feature(float_list=tf.train.FloatList(value=features[offset + 4]))
        fd[append_p_idx('boxes_features', i)] = tf.train.Feature(float_list=tf.train.FloatList(value=features[offset + 5]))
        fd[append_p_idx('boxes_labels', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=features[offset + 6]))
        fd[append_p_idx('image_area', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=[features[offset + 7]]))
        fd[append_p_idx('boxes_position', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=features[offset + 8]))
        fd[append_p_idx('boxes_height', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=features[offset + 9]))
        fd[append_p_idx('boxes_width', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=features[offset + 10]))
        fd[append_p_idx('boxes_area', i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=features[offset + 11]))
    tf_example = tf.train.Example(features=tf.train.Features(feature=fd))
    return tf_example

def main(train_label_path, output_tfrecord_path, product_dict_file, query_dict_file, word_dict_file):
    word_dict = pickle.load(open(word_dict_file, 'rb'))
    product_dict = pickle.load(open(product_dict_file, 'rb'))
    query_dict = pickle.load(open(query_dict_file, 'rb'))
    idx = 0
    with codecs.open(train_label_path, 'r', encoding='utf-8') as file_hander:
        file_hander.readline()
        with tf.python_io.TFRecordWriter(output_tfrecord_path) as writer:
            cnt = 0
            for block in read_large_file(file_hander):
                print("block: {}".format(cnt))
                cnt += 1
                for b in block:
                    features = b.strip('\n').split('\t')
                    idx += 1
                    example = create_tf_example(features, product_dict, query_dict, word_dict)
                    writer.write(example.SerializeToString())

    print('total num',idx)

if __name__ == "__main__":
    # train_label_path, output_tfrecord_path, product_dict_file, query_dict_file, word_dict_file, neg_num = sys.argv[1:]
    # neg_num = int(neg_num)
    # main(train_label_path, output_tfrecord_path, product_dict_file, query_dict_file, word_dict_file, neg_num)
    # main(valid_tsv, ndcg_valid_tfrecord, interid_product_dict_path, interim_query_dict_path, word_dict_path)
    main(testA_tsv, ndcg_testA_tfrecord, interid_product_dict_path, interid_query_dict_path, word_dict_path)