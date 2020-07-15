# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import json

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("script_mode", "local", "local or afo")
flags.DEFINE_string("run_id", "", "if not given, a `yyyyMMddHHmmSS` will be generated automatically")
flags.DEFINE_string("model_conf", "", "name of model definition json file")
flags.DEFINE_integer("epoch_samples", -1, "epoch_samples")
flags.DEFINE_integer("epochs", -1, "epochs")

# 由AFO喂入的参数
flags.DEFINE_string("ps_hosts", "", "comma-separated hosts list with host format in host:port")
flags.DEFINE_string("worker_hosts", "", "comma-separated hosts list with host format in host:port")
flags.DEFINE_string("chief_hosts", "", "comma-separated hosts list with host format in host:port")
flags.DEFINE_string("evaluator_hosts", "", "comma-separated hosts list with host format in host:port")
flags.DEFINE_string("job_name", "", "role")
flags.DEFINE_integer("task_index", 0, "task index")
flags.DEFINE_string("eval_file", "NULL", "task index")
flags.DEFINE_string("output_file", "NULL", "task index")
flags.DEFINE_string("timestamped_saved_model", "NULL", "task index")
flags.DEFINE_string("train_train_data", "NULL", "train train")
flags.DEFINE_string("train_valid_data", "NULL", "train valid")
flags.DEFINE_string("valid_data", "NULL", "train valid")
flags.DEFINE_string("train_data", "NULL", "train valid")
flags.DEFINE_string("model_ckpt_dir", "NULL", "train valid")
flags.DEFINE_string("batch_size", "120", "batch size")
flags.DEFINE_bool("use_socket", True, "data agent use socket or not.")
flags.DEFINE_integer("data_port", 0, "data agent port if agent use socket .")
flags.DEFINE_integer("shm_size", 0, "data agent shr mem size if agent use shr mem.")
flags.DEFINE_string("shm_name", "", "data agent shr name size if agent use shr mem.")
flags.DEFINE_bool ("async_save_ckpt", False, "async_save_ckpt")
flags.DEFINE_string("warm_start_id", "", "")
def parse_hosts(hosts_str):
    return [x.strip() for x in hosts_str.split(',') if x.strip()]

def flag_setup():
    if FLAGS.script_mode == "afo":
        print(os.environ['TF_CONFIG'])

        cluster_config = {
            'chief': parse_hosts(FLAGS.chief_hosts),
            'worker': parse_hosts(FLAGS.worker_hosts),
            'ps': parse_hosts(FLAGS.ps_hosts),
            'evaluator': parse_hosts(FLAGS.evaluator_hosts)
        }

        os.environ['TF_CONFIG'] = json.dumps({
            'cluster': cluster_config,
            'task': {
                'type': FLAGS.job_name,
                'index': FLAGS.task_index}
            }
        )
