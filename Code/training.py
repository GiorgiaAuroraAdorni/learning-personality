#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from model_input import create_input_fn
from model_input import model_fn


def main(argv):
    session_config = tf.ConfigProto()

    config = tf.estimator.RunConfig(model_dir='Model/',
                                    save_summary_steps=50,
                                    session_config = session_config)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       params={},
                                       config=config)

    hooks = [tf.train.ProfilerHook(save_steps=200, output_dir='Timeline/')]

    estimator.train(input_fn=create_input_fn("Dataset/TFRecords/train.tfrecords.gz"),
                    hooks=hooks)

    eval_result = estimator.evaluate(input_fn=create_input_fn("Dataset/TFRecords/test.tfrecords.gz"))

    print('\nRoot Mean Square Error: {rmse:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':
    # Execute code only if executing as script
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
