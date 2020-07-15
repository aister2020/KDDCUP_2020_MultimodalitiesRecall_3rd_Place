# -*- coding: utf-8 -*-
import tensorflow as tf
import utils.flag_setup as flag_setup
import numpy as np
from helper import append_idx
import helper
import pickle
class ATTPredict(object):
    def __init__(self, model_json, mode):
        blend_word2vec_path = '../../../user_data/blend_word2vec.pkl'
        if mode == 'afo':
            blend_word2vec_path = 'training.tar.gz/src/blend_word2vec.pkl'
        
        self.cur_run_mode = mode
        self.model_json = model_json
        self.neg_num = model_json["model"]["neg_num"]
        self.eval_neg_num = model_json["model"]["eval_neg_num"]
        self.NUM_NAME = 'num_boxes'
        self.run_id = flag_setup.FLAGS.run_id
        self.model_name = model_json["model"]["model_name"]
        self.hidden_layer = model_json["model"]["hidden_layers"]
        self.embedding_size = {}
        for field in model_json["data_schema"]["features"]:
            if field['type'] == 'embedding' or field['type'] == 'embedding_sp' or field['type'] == 'embedding_seq':
                self.embedding_size[field["name"]] = field["max"] + 1

        self.learning_rate = model_json["model"]["learning_rate"]
        self.epoch = model_json["model"]["epoch"]
        self.batch_size = model_json["model"]["batch_size"]
        self.job_name = "worker"
        if mode == "afo":
            self.job_name = flag_setup.FLAGS.job_name
        self.word_vecs = pickle.load(open(blend_word2vec_path,'rb'))
        print('word2vec path',blend_word2vec_path)
        
        self.use_sample_type = model_json["model"]["use_sample_type"]   

    def cal_logit(self, query_seq,query_num,query_mask, query_emb, query_lastword, boxes_embedding, label_embedding,\
                 boxes_num,\
                      height, width, area, item_mask, mode):
        
        
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        
        
        # with tf.variable_scope("item_semantic", reuse=tf.AUTO_REUSE):
            
            
            
        with tf.variable_scope("cross", reuse=tf.AUTO_REUSE):
            # label_features, boxes_area_ratio_embedding, left_id_ratio_embedding, width_ratio_embedding, top_id_ratio_embedding,\
            #                                 heigth_ratio_embedding,boxes_position,boxes_height,boxes_width,boxes_area = label_embedding
            
            # boxes_concat = tf.concat([boxes_embedding]+label_embedding,axis=-1)
            
            # boxes_concat_shape1 = tf.shape(boxes_concat)[1]
            
             
            with tf.variable_scope('query_semantic'):
                # query_emb = tf.layers.dropout(query_emb, rate=0.1, training=training)
                # query_emb = helper.tanh_sigmoid(query_emb,300)
                # query_emb = tf.layers.batch_normalization(query_emb, training=training)
                # query_emb = tf.layers.dropout(query_emb, rate=0.1, training=training)
                query_emb =  tf.layers.dense(query_emb,300,activation=tf.nn.relu)
                query_emb = helper.layer_norm(query_emb,300)
                query_emb = tf.layers.dropout(query_emb, rate=0.1, training=training)
                
            
            with tf.variable_scope('item_semantic',reuse=tf.AUTO_REUSE):
                boxes_embedding = tf.layers.dense(boxes_embedding,units=1024,activation=tf.nn.relu)
                boxes_embedding = helper.layer_norm(boxes_embedding,1024)
                boxes_embedding = tf.layers.dropout(boxes_embedding, rate=0.1, training=training)
                
                # boxes_embedding = helper.tanh_sigmoid(boxes_embedding,384)
                boxes_embedding = tf.layers.dense(boxes_embedding,units=512,activation=tf.nn.relu)
                boxes_embedding = helper.layer_norm(boxes_embedding,512)
                boxes_embedding = tf.layers.dropout(boxes_embedding, rate=0.1, training=training)
                
                
                # boxes_embedding = helper.tanh_sigmoid(boxes_embedding,300)
                boxes_embedding = tf.layers.dense(boxes_embedding,units=300,activation=tf.nn.relu)
                boxes_embedding = helper.layer_norm(boxes_embedding,300)
                boxes_embedding = tf.layers.dropout(boxes_embedding, rate=0.1, training=training)
                
                # label_embedding = helper.tanh_sigmoid(tf.concat(label_embedding,axis=-1),300)
                label_embedding = tf.layers.dense(tf.concat(label_embedding,axis=-1),units=300,activation=tf.nn.relu)
                label_embedding = helper.layer_norm(label_embedding,300)
                label_embedding = tf.layers.dropout(label_embedding, rate=0.1, training=training)
                
                
                boxes_concat = tf.concat([boxes_embedding,label_embedding],axis=-1)
                boxes_value = tf.layers.dense(boxes_concat,300,activation=tf.nn.relu)
                # boxes_value = tf.layers.tanh_sigmoid(boxes_concat,300)
                boxes_value = helper.layer_norm(boxes_value,300)
                boxes_value = tf.layers.dropout(boxes_value, rate=0.1, training=training)
                
                # boxes_concat = helper.conditional_layer_norm_with_query(boxes_concat,300,conditional_input=query_emb)
                # boxes_concat = tf.layers.dropout(boxes_concat, rate=0.1, training=training)
            
                
            # query_query = query_emb
            # # query_query = tf.concat([query_emb,query_lastword],axis=-1)
            # boxes_key = boxes_concat
            # boxes_value = boxes_concat
            
            
            
                
        
            with tf.variable_scope("query_image_image_attention", reuse=tf.AUTO_REUSE):
                # query_query = tf.layers.dense(query_emb,100)
                # boxes_key = tf.layers.dense(boxes_concat,100)
                
                # query_query = helper.tanh_sigmoid(query_emb,100)
                # boxes_key = helper.tanh_sigmoid(boxes_concat,100)
                
                query_query = query_emb
                boxes_key = boxes_value
                
                att_query_image_image,softmax_score = helper.dot_attention_with_query(query_query, boxes_key, boxes_value, mask=item_mask,scale_dot=True)
                
                # att_query_image_image,softmax_score = helper.attention_with_query(query_query, boxes_key, boxes_value, mask=item_mask)
                
                # att_query_image_image = tf.reduce_sum(boxes_value*tf.expand_dims(item_mask,axis=2),axis=1)/tf.expand_dims(tf.cast(boxes_num,dtype=tf.float32),axis=1)
                
                query_out = query_emb
                # query_out = helper.tanh_sigmoid(query_emb, 300)
                # query_out = tf.layers.batch_normalization(query_out, training=training)
                # query_out = tf.layers.dropout(query_out, rate=0.1, training=training)
                
                image_out = att_query_image_image
                # image_out = helper.tanh_sigmoid(att_query_image_image, 300)
                # image_out = tf.layers.batch_normalization(image_out, training=training)
                # image_out = tf.layers.dropout(image_out, rate=0.1, training=training)

                # image_out = helper.conditional_layer_norm(image_out,units=300,conditional_input=query_out)
           
                # image_out = tf.layers.batch_normalization(image_out, training=training)
                # image_out = tf.layers.dropout(image_out, rate=0.1, training=training)
                # att_query_image_image,softmax_score = helper.attention_with_query(query_query, boxes_value, boxes_value, mask=item_mask, activation=None, scale=None)
                
                
                # att_query_image_image_ex = tf.expand_dims(att_query_image_image, axis=1)
                # image_tile = tf.tile(att_query_image_image_ex,[1,tf.shape(query_seq)[1],1])
                # seq = tf.concat([query_seq,image_tile],axis=-1)
                
                # seq_dense = self.add_layer(seq, 700, 300, activation_function=tf.nn.tanh, name='seq_dense')
                
                # seq_dense = seq_dense * query_mask
                # # query_object_match = tf.reduce_max(seq_dense,axis=1)
                # query_object_match = tf.reduce_sum(seq_dense,axis=1)#/tf.expand_dims(tf.cast(query_num,dtype=tf.float32),axis=1)
            
                # image_mean = tf.reduce_sum(boxes_value*tf.expand_dims(item_mask,axis=2),axis=1) /tf.expand_dims(tf.cast(boxes_num,dtype=tf.float32),axis=1)
               
            
            concat_out = tf.concat([query_out*image_out,height,width,area], axis=1)
            # concat_out = tf.concat([query_emb, att_query_image_image, height,width,area], axis=1)
            concat_out = tf.layers.batch_normalization(concat_out, training=training)
            concat_out = tf.layers.dropout(concat_out, rate=0.1, training=training)
         
                
         
        deep_out,logit = self.add_fc_layers(concat_out, name='dense', mode=mode)
        return logit,softmax_score,query_out,image_out,deep_out


    def model_fn(self, features, labels, mode, params):
        neg_num = self.neg_num
        if mode == tf.estimator.ModeKeys.PREDICT:
            neg_num = 0
            tf.logging.info("neg_num:")
            tf.logging.info(neg_num)
        if mode == tf.estimator.ModeKeys.EVAL:
            neg_num = self.eval_neg_num
        def _embedding_simple(name, embedding_ids, embedding_size, embedding_dim):
            X = tf.get_variable(name, [embedding_size, embedding_dim],
                                initializer=tf.truncated_normal_initializer(0.0, 1e-5), trainable=True)
            out_tensor = tf.gather(X, embedding_ids)
            return out_tensor
        def _embedding(f, embedding_dim, is_sp=False, idx=None, init_vec=None, fea_name=None):
            with tf.variable_scope("input_embedding", reuse=tf.AUTO_REUSE):
                if idx is not None:
                    feature_name = append_idx(f, idx)
                else:
                    feature_name = f
                    
                if fea_name is not None:
                    feature_name = fea_name
                    
                if init_vec is None:
                    emb_var = tf.get_variable("emb_" + str(f), [self.embedding_size[f], embedding_dim],
                                          initializer=tf.truncated_normal_initializer(0.0, 1e-5), trainable=True)
                else:
                    emb_var = tf.get_variable("emb_" + str(f), [self.embedding_size[f], embedding_dim],
                                              initializer=tf.constant_initializer(init_vec),
                                              trainable=True
                                              )
                if is_sp:
                    out_tensor = tf.nn.embedding_lookup_sparse(emb_var, features[feature_name], None, combiner="mean")
                else:
                    out_tensor = tf.gather(emb_var, features[feature_name])
                return out_tensor

        training = (mode == tf.estimator.ModeKeys.TRAIN)
        
        pos_emb = tf.get_variable("pos_embedding", [100, 100], initializer=tf.truncated_normal_initializer(0.0, 1e-5), trainable=True)
        with tf.variable_scope("query_semantic"):
            query_emb_size = self.model_json['model']['query_embedding_size']
            query_emb = _embedding('query', query_emb_size, init_vec=self.word_vecs)
            cur_pos_emb = tf.expand_dims(pos_emb[0:tf.shape(query_emb)[1]], axis=0)
            cur_pos_emb = tf.tile(cur_pos_emb, [tf.shape(query_emb)[0], 1, 1])
            query_seq = tf.concat([query_emb, cur_pos_emb], axis=-1)
            
            # query_seq = query_emb
            query_num = features['query_words_num']
            query_mask = tf.expand_dims(tf.sequence_mask(query_num, dtype=tf.float32), axis=2)
            # /tf.expand_dims(tf.cast(query_num,dtype=tf.float32),axis=1)
            # query_emb = self.query_semantic_layer(tf.reduce_sum(query_seq * query_mask, axis=1), query_emb_size + 100, mode=mode)
            query_emb = tf.reduce_sum(query_seq * query_mask, axis=1)
            query_lastword = None # _embedding('last_word', query_emb_size, is_sp=False, init_vec=self.word_vecs)
            # query_emb = tf.layers.dropout(query_emb, rate=0.1, training=training)
        
        image_feature_mean = tf.constant([[helper.image_feature_mean]])
        image_feature_std = tf.constant([[helper.image_feature_std]])       
        
        item_masks = []
        boxes_embeddings = []
        label_embeddings = []
        boxes_num = []
        
        height_embs = []
        width_embs = []
        area_embs = []
        for i in range(neg_num + 1):
            with tf.variable_scope("boxes", reuse=tf.AUTO_REUSE):
                
                boxes_features = features[append_idx('boxes_features', i)]
                # boxes_features = (boxes_features-image_feature_mean)/image_feature_std
                label_feature_embedding = _embedding('boxes_labels', 300, idx=i)
                # label_features = tf.layers.dropout(label_features, rate=0.1, training=training)
                boxes_position_embedding = _embedding('boxes_position', 20, idx=i)
                boxes_height_embedding = _embedding('boxes_height', 20, idx=i)
                boxes_width_embedding = _embedding('boxes_width', 20, idx=i)
                boxes_area_embedding = _embedding('boxes_area', 20, idx=i)

                num = features[append_idx('num_boxes', i)]
                
                boxes_masks = tf.sequence_mask(num, dtype=tf.float32)
            
                
                boxes_coordinate = tf.clip_by_value(tf.cast(features[append_idx('boxes', i)], tf.float32), 0, 1000)
                tf.logging.info("boxes_coordinate shape:")
                tf.logging.info(boxes_coordinate.get_shape().as_list())
                img_height = tf.expand_dims(tf.cast(features[append_idx('height', i)], tf.float32), axis=1)
                img_width = tf.expand_dims(tf.cast(features[append_idx('width', i)], tf.float32), axis=1)
                tf.logging.info("img_width shape:")
                tf.logging.info(img_width.get_shape().as_list())
                boxes_width = tf.cast(features[append_idx('boxes_width',i)],dtype=tf.float32)
                boxes_height = tf.cast(features[append_idx('boxes_height',i)],dtype=tf.float32)
                boxes_area_ratio = tf.cast(features[append_idx('boxes_area',i)],dtype=tf.float32)/tf.expand_dims(tf.cast(features[append_idx("image_area",i)], tf.float32), axis=1)
                boxes_area_ratio_ids = tf.clip_by_value(tf.cast(boxes_area_ratio / 0.1, tf.int64), 0, 10)
                left_id_ratio_ids = tf.clip_by_value(tf.cast( boxes_coordinate[:, :, 1] / (img_width*10) / 0.1, tf.int64), 0, 10)
                width_ratio_ids = tf.clip_by_value(tf.cast((boxes_width*5) / (img_width*10) / 0.1, tf.int64), 0, 10)
                top_id_ratio_ids = tf.clip_by_value(tf.cast( boxes_coordinate[:, :, 0] / (img_height*10) / 0.1, tf.int64), 0, 10)
                heigth_ratio_ids = tf.clip_by_value(tf.cast((boxes_height*5) / (img_height*10) / 0.1, tf.int64), 0, 10)
                
                boxes_area_ratio_embedding = _embedding_simple('boxes_area_ratio', boxes_area_ratio_ids, 11, 20)
                left_id_ratio_embedding = _embedding_simple('boxes_left_ratio', left_id_ratio_ids, 11, 20)
                width_ratio_embedding = _embedding_simple('boxes_width_ratio', width_ratio_ids, 11, 20)
                top_id_ratio_embedding = _embedding_simple('boxes_top_ratio', top_id_ratio_ids, 11, 20)
                heigth_ratio_embedding = _embedding_simple('boxes_height_ratio', heigth_ratio_ids, 11, 20)
                label_embeddings.append([label_feature_embedding, boxes_area_ratio_embedding, left_id_ratio_embedding, width_ratio_embedding, top_id_ratio_embedding,
                                            heigth_ratio_embedding,boxes_position_embedding,boxes_height_embedding,boxes_width_embedding,boxes_area_embedding])
                
                boxes_embeddings.append(boxes_features)
                boxes_num.append(num)
                
                height =  _embedding('height', 20, idx=i)
                width = _embedding('width', 20, idx=i)
                image_area = _embedding('image_area', 20, idx=i)
                height_embs.append(height)
                width_embs.append(width)
                area_embs.append(image_area)
                
                item_masks.append(boxes_masks)
            
        tf.logging.info("query_in:")
        tf.logging.info(tf.shape(query_emb))

        logit,softmax_score,query_out,image_out,deep_out = self.cal_logit(query_seq,query_num,query_mask, query_emb, query_lastword, boxes_embeddings[0], label_embeddings[0],boxes_num[0],
                                   height_embs[0],width_embs[0],area_embs[0], item_masks[0], mode=mode)
            
        if self.cur_run_mode=='afo':
            every_n_iter = 5000
        else:
            every_n_iter = 200
        logging_hook = tf.train.LoggingTensorHook(every_n_iter=every_n_iter,tensors={'softmax_score': softmax_score})
        
        logit = tf.reshape(logit, [-1, 1])
        predict = tf.sigmoid(logit)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predict_dict = {"prediction": predict,"query_out":query_out,"image_out":image_out,"deep_out":deep_out,"query_emb":query_emb}
            export_output = {'serving': tf.estimator.export.PredictOutput(predict_dict)}
            return tf.estimator.EstimatorSpec(mode, predictions=predict_dict, export_outputs=export_output)

        global_step = tf.train.get_global_step()
        if neg_num > 0:
            score = [tf.reshape(logit, [-1, 1])]
            for i in range(1, neg_num + 1):
                logit,softmax_score,query_out,image_out,deep_out  = self.cal_logit(query_seq,query_num,query_mask, query_emb, query_lastword, boxes_embeddings[i], label_embeddings[i],boxes_num[i],
                                       height_embs[i],width_embs[i],area_embs[i], item_masks[i], mode=mode)
                score.append(tf.reshape(logit, [-1, 1]))
            score = tf.concat(score, axis=1)
            prob = tf.nn.softmax(score, axis=1)
            predict = prob[:, 0]
            loss = -tf.reduce_mean(tf.log(predict))
        else:
            label = tf.reshape(tf.cast(labels, tf.float32), [-1, 1])
            if self.use_sample_type==1:
                stepsize = 300
                iteration = tf.cast(global_step,tf.float32)

                beta = 0.7**(1+iteration/stepsize)
                
                extra_preds = tf.reshape(tf.cast(features['extra_preds'], tf.float32), [-1, 1])
                soft_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=extra_preds, logits=logit))
                
                label = tf.reshape(tf.cast(labels, tf.float32), [-1, 1])
                hard_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
                
                loss = beta*hard_loss + (1-beta) * soft_loss
                
                
                # weights = tf.constant([0,1,1,1],dtype=tf.float32)
                # sample_type = features['sample_type']

                # loss_weights = tf.gather(weights,tf.reshape(sample_type,[-1,1]))
                # loss = loss * loss_weights
                # loss = tf.reduce_mean(loss)
            else:
                
                loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
            # auc = tf.metrics.auc(labels, predict)

            # logging_hook = tf.train.LoggingTensorHook(every_n_iter=100,
            #                                           tensors={'auc': auc[0]})

        # 有loss和auc，可以定义eval的返回了
        if mode == tf.estimator.ModeKeys.EVAL:
            auc = tf.metrics.auc(labels, predict)
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={"eval-auc": auc})

        assert mode == tf.estimator.ModeKeys.TRAIN
        decay_steps = self.model_json['model']['decay_steps']
        decay_rate = self.model_json['model']['decay_rate']
        
        tf.summary.scalar('train-loss', loss)
        global_step = tf.train.get_global_step()
        lr = tf.train.exponential_decay(learning_rate=self.learning_rate, global_step=global_step, decay_steps=decay_steps, decay_rate=decay_rate)
        
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(loss, global_step=global_step)
        tf.logging.info('all_variable:{}'.format(tf.all_variables()))
        train_op = tf.group([train_op, update_ops])
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,)

            # training_hooks=[logging_hook])


    def add_fc_layers(self, deep_in, mode, name):
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        """各层的定义"""
        with tf.variable_scope("dense_layers", reuse=tf.AUTO_REUSE):
            deep_out = deep_in
            for idx, unit in enumerate([300,150]):
                
                deep_out = tf.layers.dense(deep_out, units=unit, activation=tf.nn.tanh, name=name + "_" + str(idx))
                gate = tf.layers.dense(deep_out, units=unit, activation=tf.sigmoid, name=name + "_gate"  + str(idx))
                deep_out = deep_out * gate
                
                deep_out = tf.layers.batch_normalization(deep_out, training=training)
                deep_out = tf.layers.dropout(deep_out, rate=0.1, training=training)
            deep_predict = tf.layers.dense(deep_out, units=1, name=name + "_" + "final")
            return deep_out,deep_predict
