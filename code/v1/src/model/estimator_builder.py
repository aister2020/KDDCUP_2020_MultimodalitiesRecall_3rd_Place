# -*- coding: utf-8 -*-
import tensorflow as tf
import utils.flag_setup as flag_setup
import model.att as att
import model.att_list as att_list
import reader.data_set as data_set
import os
import model.image_sim as image_sim
import model.att_kn as att_kn
import model.att_interid as att_interid
import model.att_predict as att_predict

class EvalResultsExporter(tf.estimator.Exporter):
    """Passed into an EvalSpec for saving the result of the final evaluation
    step locally or in Google Cloud Storage.
    """

    def __init__(self, name, model_json, run_id, serving_input_receiver_fn):
        assert name, '"name" argument is required.'
        self._name = name
        self._model_json = model_json
        self._run_id = run_id
        self._serving_input_receiver_fn = serving_input_receiver_fn
        self.num = 0
        self.batch_size = model_json['model']['batch_size']
        self.epoch_steps = int(flag_setup.FLAGS.epoch_samples / model_json['model']['batch_size']) + 1
    @property
    def name(self):
        return self._name

    def export(self, estimator, export_path, checkpoint_path,
               eval_result, is_the_final_export):
        if self._model_json['mode'] == 'local': return None

        current_epoch = self.num
        saved_path = os.path.join(os.path.join(self._model_json["export"]["savedmodel_dir"], self._run_id), 'epoch{}'.format(current_epoch))
        if tf.gfile.Exists(saved_path):
            tf.logging.info("no export at steps: {}".format(self.num * self.epoch_steps * self.batch_size))
            return None

        tf.logging.info("Exporting at steps: {}".format(self.num * self.epoch_steps * self.batch_size))
        tf.gfile.MakeDirs(saved_path)
        tf.logging.info(('EvalResultsExporter (name: %s) '
                         'running after final evaluation.') % self._name)
        tf.logging.info('export_path: %s' % export_path)
        tf.logging.info('eval_result: %s' % eval_result)
        estimator.export_savedmodel(
            export_dir_base=saved_path,
            serving_input_receiver_fn=self._serving_input_receiver_fn
        )
        self.num += 1
        return saved_path


def create_estimator_and_specs(run_config, model_json):
    if model_json['model']['model_name'] == 'att':
        model = att.ATT(model_json, flag_setup.FLAGS.script_mode)
    elif model_json['model']['model_name'] == 'att_list':
        model = att_list.ATTList(model_json, flag_setup.FLAGS.script_mode)
    elif model_json['model']['model_name'] == 'image_sim':
        model = image_sim.ImageSim(model_json, flag_setup.FLAGS.script_mode)
    elif model_json['model']['model_name'] == 'att_kn':
        model = att_kn.ATTKN(model_json, flag_setup.FLAGS.script_mode)
    elif model_json['model']['model_name'] == 'att_interid':
        model = att_interid.ATTInterID(model_json, flag_setup.FLAGS.script_mode)
        
    epoch_steps = int(flag_setup.FLAGS.epoch_samples / model_json['model']['batch_size']) + 1
    epochs =  model_json['model']['epoch']
    max_steps = epoch_steps*epochs

    data_loader = data_set.Data_Loader(model_json, flag_setup.FLAGS.script_mode)

    model_params = tf.contrib.training.HParams()
    if model_json['mode'] == 'local':
        if flag_setup.FLAGS.warm_start_id is not None and flag_setup.FLAGS.warm_start_id!="":
            estimator = tf.estimator.Estimator(
                model_fn=model.model_fn,
                config=run_config,
                params=model_params,
                warm_start_from=tf.estimator.WarmStartSettings(
                    ckpt_to_initialize_from='../../../user_data/export/checkpoint/{}'.format(
                        flag_setup.FLAGS.warm_start_id),vars_to_warm_start=['(?!global_step:0)(?!beta1_power:0)(?!beta2_power:0)(?!extra).*'])
                # flag_setup.FLAGS.warm_start_id),vars_to_warm_start=['cross/boxes_features','boxes(?!/input_embedding/emb_height)(?!/input_embedding/emb_image_area)(?!/input_embedding/emb_width)'])
            )
        else:
            estimator = tf.estimator.Estimator(
                model_fn=model.model_fn,
                config=run_config,
                params=model_params
            )
    else:
        if flag_setup.FLAGS.warm_start_id is not None and flag_setup.FLAGS.warm_start_id!="":
            estimator = tf.estimator.Estimator(
                model_fn=model.model_fn,
                config=run_config,
                params=model_params,
                warm_start_from=tf.estimator.WarmStartSettings(ckpt_to_initialize_from="viewfs://hadoop-meituan/user/hadoop-mining/huangjianqiang/data/kdd/model/{}".format(
                        flag_setup.FLAGS.warm_start_id),vars_to_warm_start=['(?!global_step:0)(?!beta1_power:0)(?!beta2_power:0)(?!extra).*'])
            )
        else:
            estimator = tf.estimator.Estimator(
                model_fn=model.model_fn,
                config=run_config,
                params=model_params,

            )

    if model_json['mode'] == 'local':
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: data_loader.generate_tfrecord_dataset(stage=tf.estimator.ModeKeys.TRAIN),
            max_steps=max_steps
        )
    else:
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: data_loader.generate_tfrecord_dataset(stage=tf.estimator.ModeKeys.TRAIN),
            max_steps=model_json['model']['max_step']
        )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: data_loader.generate_tfrecord_dataset(stage=tf.estimator.ModeKeys.EVAL),
        throttle_secs=model_json["export"]["checkpoint_interval_secs"],
        start_delay_secs=30,
        steps=500,
        exporters=EvalResultsExporter('eval-saved-model', model_json, flag_setup.FLAGS.run_id, data_loader.serving_example_input_receiver_fn)
    )

    return estimator, train_spec, eval_spec, data_loader



def create_estimator_predict(run_config, model_json):
    model = att_predict.ATTPredict(model_json, flag_setup.FLAGS.script_mode)

    data_loader = data_set.Data_Loader(model_json, flag_setup.FLAGS.script_mode)

    model_params = tf.contrib.training.HParams()
    estimator = tf.estimator.Estimator(
        model_fn=model.model_fn,
        config=run_config,
        params=model_params,
        # warm_start_from=tf.estimator.WarmStartSettings(
        #     ckpt_to_initialize_from='../../../user_data/export/checkpoint/{}'.format(
        #         flag_setup.FLAGS.warm_start_id),vars_to_warm_start=['(?!global_step:0)(?!beta1_power:0)(?!beta2_power:0)(?!extra).*'])
        # flag_setup.FLAGS.warm_start_id),vars_to_warm_start=['cross/boxes_features','boxes(?!/input_embedding/emb_height)(?!/input_embedding/emb_image_area)(?!/input_embedding/emb_width)'])
    )


    return estimator, data_loader