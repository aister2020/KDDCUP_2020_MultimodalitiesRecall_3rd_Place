# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import utils.flag_setup as flag_setup

INT64_VARLEN = tf.VarLenFeature(dtype=tf.int64)

class Data_Loader(object):
    def __init__(self, model_conf, mode):
        self.model_conf = model_conf
        self.mode = mode
        self.features = []
        self.cont_fea = []
        self.cont_fea_last_dim = []
        self.neg_num = model_conf['model']['neg_num']
        self.eval_neg_num = model_conf['model']['eval_neg_num']
        self.features = []
        self.features_sp = []
        self.features_seq = []
        self.cont_fea = []
        self.cont_fea_last_dim = []

        self.cont_fix_fea = []
        self.cont_fix_fea_last_dim = []
        for field in model_conf["data_schema"]["features"]:
            if field["type"] == "embedding":
                self.features.append(field["name"])

            if field["type"] == "embedding_sp":
                self.features_sp.append(field["name"])

            if field["type"] == "embedding_seq":
                self.features_seq.append(field["name"])

            if field['type'] == 'countinous':
                self.cont_fea.append(field['name'])
                self.cont_fea_last_dim.append(field['last_dim'])

            if field['type'] == 'countinous_fix':
                self.cont_fix_fea.append(field['name'])
                self.cont_fix_fea_last_dim.append(field['last_dim'])
        
        self.extra_preds = False
        if "extra_preds" in model_conf["data_schema"]:
            self.extra_preds = True
            self.extra_pred_num =  model_conf["model"]["extra_pred_num"]
            
        self.sample_type = False
        if "sample_type" in model_conf["data_schema"]:
            self.sample_type = True
        
        self.batch_size = model_conf["model"]["batch_size"]
        self.epoch = model_conf["model"]["epoch"]
        self.epoch_samples = flag_setup.FLAGS.epoch_samples
        if mode == "local":
            self.train_data = flag_setup.FLAGS.train_train_data
            self.eval_data = flag_setup.FLAGS.train_valid_data

        self.job_name = "worker"
        if mode == "afo":
            self.job_name = flag_setup.FLAGS.job_name

        if self.job_name == "chief":
            self.epoch = self.epoch * 2


    @staticmethod
    def int64_list_feature(list_value):
        values = []
        for each in list_value:
            values.append(tf.train.Feature(int64_list=tf.train.Int64List(value=each)))
        return tf.train.FeatureList(feature=values)

    def generate_tfrecord_dataset(self, stage):
        def _get_input_files():
            if self.mode == "local":
                if stage == tf.estimator.ModeKeys.TRAIN:
                    if ',' in self.train_data:
                        return self.train_data.split(',')
                    elif os.path.isfile(self.train_data):
                        return self.train_data
                    elif os.path.isdir(self.train_data):
                        return [os.path.join(self.train_data, l) for l in os.listdir(self.train_data)]
                    else:
                        raise Exception("train data %s not exist." % self.train_data)
                if stage == tf.estimator.ModeKeys.EVAL:
                    if os.path.isfile(self.eval_data):
                        return self.eval_data
                    elif os.path.isdir(self.eval_data):
                        return [os.path.join(self.eval_data, l) for l in os.listdir(self.eval_data)]
                    else:
                        raise Exception("train data %s not exist." % self.eval_data)
            if self.mode == "afo":
                """注意在分布式环境下，传入的数据目录默认是本机的inputs下。然后我们需要把这个文件夹下的文件整理出来用"""


                file_names = []
                if flag_setup.FLAGS.job_name == 'worker' or flag_setup.FLAGS.job_name == 'evaluator':
                    file_list = os.listdir("inputs")
                    for current_file_name in file_list:
                        file_path = os.path.join("inputs", current_file_name)
                        file_names.append(file_path)
                return file_names

        def _parse_tfrecord_example(raw_serialized, schema):
            features = tf.parse_example(raw_serialized, schema)
            label_tensor = features.pop("peudo-label")
            label_tensor = tf.sparse_tensor_to_dense(label_tensor, default_value=0)
            return features, label_tensor

        def _input_fn():
            if self.mode == "afo":
                dataset = tf.data.AfoDataset()         #构造统一分发AFODataset
            else:
                filenames = _get_input_files()
                print('input filenames',filenames)
                dataset = tf.data.TFRecordDataset(filenames)
                if stage == tf.estimator.ModeKeys.TRAIN:
                    dataset = dataset.shuffle(self.epoch_samples)
                    dataset = dataset.repeat(self.epoch)
                else:
                    dataset = dataset.repeat(1)
                    
            if stage == tf.estimator.ModeKeys.TRAIN:
                neg_num = self.neg_num
            else:
                neg_num = self.eval_neg_num 
            example_schema = self.get_schema(neg_num)
            if self.extra_preds:
                example_schema["extra_preds"] = tf.FixedLenFeature(shape=[self.extra_pred_num],dtype=tf.float32)
                
            if self.sample_type:
                 example_schema["sample_type"] = tf.FixedLenFeature(shape=[],dtype=tf.int64)
                 
            example_schema = dict(sorted(example_schema.items(), key=lambda k: k[0]))
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.map(lambda value: _parse_tfrecord_example(value, example_schema))
            dataset = dataset.prefetch(10)

            return dataset

        return _input_fn()


    def get_schema(self,neg_num):
        def _build_example_schema(example_schema, feature_name, feature_schema):
            if feature_name in self.model_conf['data_schema']['query_features']:
                example_schema[feature_name] = feature_schema
            elif feature_name in self.model_conf['data_schema']['item_features']:
                for i in range(1 + neg_num):
                    example_schema[feature_name + '_' + str(i)] = feature_schema
            else:
                raise Exception("feature name {} has no owner.".format(feature_name))
            return example_schema
        example_schema = {}
        example_schema["peudo-label"] = INT64_VARLEN
        for k in self.features_sp:
            example_schema = _build_example_schema(example_schema, k, tf.VarLenFeature(dtype=tf.int64))
        for k in self.features:
            example_schema = _build_example_schema(example_schema, k, tf.FixedLenFeature(shape=[],dtype=tf.int64))
        for k in self.features_seq:
            example_schema = _build_example_schema(example_schema, k, tf.FixedLenSequenceFeature([], tf.int64, True))
        for k, dim in zip(self.cont_fea, self.cont_fea_last_dim):
            example_schema = _build_example_schema(example_schema, k, tf.FixedLenSequenceFeature([dim], tf.float32, True))
        for k,dim in zip(self.cont_fix_fea, self.cont_fix_fea_last_dim):
            example_schema = _build_example_schema(example_schema, k, tf.FixedLenFeature([dim], tf.float32))
        example_schema = dict(sorted(example_schema.items(), key=lambda k: k[0]))
        return example_schema

    def serving_example_input_receiver_fn(self):
        """把我们的数据解析成一个example的格式, 其输出的格式和tain的input格式一致"""
        serialized_tf_example = tf.placeholder(dtype=tf.string, name='input_example_tensor')
        # 这里的examples就定义了入参的名称
        receiver_tensors = {'examples': serialized_tf_example}
        neg_num = 0
        example_schema = self.get_schema(neg_num)
        features = tf.parse_example(serialized_tf_example, example_schema)
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)






