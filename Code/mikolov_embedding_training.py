#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from mikolov_embedding_model import model_fn
from mikolov_embedding_model import create_input_fn


def main(argv):

    train_data = "Dataset/TFRecords/train.tfrecords.gz"
    test_data = "Dataset/TFRecords/test.tfrecords.gz"

    session_config = tf.ConfigProto()

    config = tf.estimator.RunConfig(model_dir='Model/Mikolov/010/',
                                    save_summary_steps=50,
                                    session_config=session_config,
                                    # log_step_count_steps=1
                                    )

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       params={'vocabulary_size':60000 + 1, 'embedding_size':250, 'num_sampled': 50},
                                       config=config)

    input_fn = create_input_fn(train_data, is_training=True)

    hooks = [tf.train.ProfilerHook(save_steps=200, output_dir='Timeline/')]  

    idx = 0

    for idx in range(0, 10):
        estimator.train(input_fn=input_fn, hooks=hooks)

        eval_result = estimator.evaluate(input_fn=create_input_fn("Dataset/TFRecords/test.tfrecords.gz", is_training=False))

        idx += 1

    estimator.predict(input_fn=create_input_fn(test_data, is_training=False))


if __name__ == '__main__':
    # Execute code only if executing as script
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
