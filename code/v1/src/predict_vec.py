import tensorflow as tf
from tensorflow.contrib import predictor
import utils.flag_setup as flag_setup
#from utils.flag_setup import FLAGS
import json
import pickle

def predict(model_conf, src_file, out_file_path):
    savedmodel_path = flag_setup.FLAGS.timestamped_saved_model
    clf = predictor.from_saved_model(savedmodel_path)
    dataset = tf.data.TFRecordDataset(src_file, num_parallel_reads=8)
    dataset = dataset.batch(model_conf['model']['batch_size'])
    iterator = dataset.make_one_shot_iterator()
    one_batch = iterator.get_next()
    #y_one_batch = tf.parse_example(one_batch, {'label': tf.FixedLenFeature([], tf.int64)})
    pid_batch = tf.parse_example(one_batch, {'ori_product_id_0': tf.FixedLenFeature([], tf.int64)})
    qid_batch = tf.parse_example(one_batch, {'ori_query_id': tf.FixedLenFeature([], tf.int64)})
    #dataset = dataset.batch(model_conf['model']['batch_size'])
    #iterator = dataset.make_one_shot_iterator()
    #one_batch = iterator.get_next()
    
    outputs = {}
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            batch = 0
            while not coord.should_stop():
                batch_data, pid, qid = sess.run([one_batch, pid_batch, qid_batch])
                pid = pid['ori_product_id_0']
                qid = qid['ori_query_id']
                predicts = clf({"examples": batch_data})
                
                pred = predicts['prediction']
                query_out = predicts["query_out"]
                image_out = predicts["image_out"]
                deep_out = predicts["deep_out"]
                query_emb = predicts["query_emb"]
                # print("predicts shape: ", pred.shape)
                
                for idx in range(len(pred)):
                    ctr = pred[idx]
                    qout = query_out[idx]
                    iout = image_out[idx]
                    dout = deep_out[idx]
                    emb = query_emb[idx]
                    outputs[(qid[idx],pid[idx])] = [ctr,qout,iout,dout,emb]
                    
                batch = batch + 1
                # print(str(batch * model_conf['model']['batch_size']) + " evaluate done...")
            print("Evaluate Finished!")
        except tf.errors.OutOfRangeError:
            print('Done evaluating -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
     
    pickle.dump(outputs,open(out_file_path,'wb'))
        
    
def main(_):
    model_conf = json.load(open(flag_setup.FLAGS.model_conf, 'r'))
    predict(model_conf, flag_setup.FLAGS.eval_file, flag_setup.FLAGS.output_file)

if __name__ == "__main__":
    #flag_setup.flag_setup()
    tf.logging.info("----start---")
    tf.app.run()