# -*- coding: utf-8 -*-
import json
import tensorflow as tf


def load_json(json_file_path):
    with open(json_file_path, "r") as config_file:
        try:
            json_conf = json.load(config_file)
            return json_conf
        except Exception:
            tf.logging.error("load json file %s error" % json_file_path)
