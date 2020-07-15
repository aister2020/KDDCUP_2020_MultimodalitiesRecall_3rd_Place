# -*- coding: utf-8 -*-

import tensorflow as tf
import datetime
import os

import utils.flag_setup as flag_setup
import utils.json_reader as json_reader
import model.estimator_builder as estimator_builder

tf.logging.set_verbosity(tf.logging.INFO)


def local_run(model_json):
    epoch_steps = int(flag_setup.FLAGS.epoch_samples/model_json['model']['batch_size'])+1
    print('epoch_steps',epoch_steps)
    run_config = tf.estimator.RunConfig(
        model_dir=os.path.join(model_json["export"]["model_dir"], flag_setup.FLAGS.run_id),
        # save_checkpoints_steps=epoch_steps,
        save_checkpoints_secs=model_json["export"]["checkpoint_secs"],
        save_summary_steps=model_json["export"]["summary_steps"],
        keep_checkpoint_max=model_json["export"]["max_checkpoints"],
    )

    estimator, train_spec, eval_spec, data_loader = estimator_builder.create_estimator_and_specs(run_config, model_json)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # 保存savedModel用来做serving
    estimator.export_savedmodel(
        export_dir_base=os.path.join(model_json["export"]["savedmodel_dir"], flag_setup.FLAGS.run_id),
        serving_input_receiver_fn=data_loader.serving_example_input_receiver_fn
    )


def local_predict_export(model_json):
    run_config = tf.estimator.RunConfig(
        model_dir=os.path.join(model_json["export"]["model_dir"], flag_setup.FLAGS.warm_start_id),
        # save_checkpoints_steps=epoch_steps,
        save_checkpoints_secs=model_json["export"]["checkpoint_secs"],
        save_summary_steps=model_json["export"]["summary_steps"],
        keep_checkpoint_max=model_json["export"]["max_checkpoints"],
    )

    estimator, data_loader = estimator_builder.create_estimator_predict(run_config, model_json)

    # 保存savedModel用来做serving
    estimator.export_savedmodel(
        export_dir_base=os.path.join(model_json["export"]["savedmodel_dir"], flag_setup.FLAGS.run_id),
        serving_input_receiver_fn=data_loader.serving_example_input_receiver_fn
    )


def afo_run(model_json):
    epoch_steps = int(flag_setup.FLAGS.epoch_samples/model_json['model']['batch_size'])+1

    tf.disable_chief_training(shut_ratio=0.8, slow_worker_delay_ratio=1.2)
    run_config = tf.estimator.RunConfig(
        model_dir=os.path.join(model_json["export"]["model_dir"], flag_setup.FLAGS.run_id),
        save_checkpoints_steps=epoch_steps,
        save_summary_steps=model_json["export"]["summary_steps"],
        keep_checkpoint_max=model_json["export"]["max_checkpoints"])

    estimator, train_spec, eval_spec, data_loader = estimator_builder.create_estimator_and_specs(run_config, model_json)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    tf.logging.warn(flag_setup.FLAGS.job_name + " finished training at " + str(datetime.datetime.now().time()))

    if flag_setup.FLAGS.job_name == "chief":
        # 保存savedModel用来做serving
        estimator.export_savedmodel(
            export_dir_base=os.path.join(os.path.join(model_json["export"]["savedmodel_dir"], flag_setup.FLAGS.run_id), 'epoch'),
            serving_input_receiver_fn=data_loader.serving_example_input_receiver_fn
        )


def main(unused_argv):
    # 加载模型配置
    if flag_setup.FLAGS.model_conf:

        model_conf = json_reader.load_json(flag_setup.FLAGS.model_conf)
        
        if flag_setup.FLAGS.epochs is not None and flag_setup.FLAGS.epochs!=""  and flag_setup.FLAGS.epochs > 0:
            model_conf['model']['epoch'] = flag_setup.FLAGS.epochs
            
        print('epochs',model_conf['model']['epoch'])
        
        if flag_setup.FLAGS.script_mode == "local":
            local_run(model_conf)

        if flag_setup.FLAGS.script_mode == "afo":
            afo_run(model_conf)
            
        if flag_setup.FLAGS.script_mode == "local_predict_export":
            local_predict_export(model_conf)
    else:
        tf.logging.info('can not load model_conf file %s' % flag_setup.FLAGS.model_conf)


if __name__ == "__main__":
    flag_setup.flag_setup()
    tf.app.run()
