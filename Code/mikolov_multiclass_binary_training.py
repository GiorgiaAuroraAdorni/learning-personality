#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from mikolov_multiclass_binary_model import model_fn
from mikolov_multiclass_binary_model import create_input_fn


def main(argv):

    train_data = "Dataset/TFRecords/train.tfrecords.gz"
    test_data = "Dataset/TFRecords/test.tfrecords.gz"

    session_config = tf.ConfigProto()

    config = tf.estimator.RunConfig(model_dir='Model/Mikolov/Binary/004/',
                                    save_summary_steps=50,
                                    session_config=session_config)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       params={'vocabulary_size':60000 + 1, 'embedding_size':250, 'num_sampled': 50},
                                       config=config)


    input_fn = create_input_fn(train_data)

    hooks = [tf.train.ProfilerHook(save_steps=200, output_dir='Timeline/')]  # init_hook]

    for idx in range(20):
        estimator.train(input_fn=input_fn, hooks=hooks)

        eval_result = estimator.evaluate(input_fn=create_input_fn("Dataset/TFRecords/test.tfrecords.gz"))

    predictions = estimator.predict(input_fn=create_input_fn(test_data))

    input = create_input_fn(test_data)
    dataset = input()
    iterator = dataset.make_initializable_iterator()
    feature, label = iterator.get_next()

    file = open("Model_Dataset/Mikolov/Binary/004.txt", "w")
    ids = 0

    with tf.Session() as sess:
        tf.tables_initializer().run()
        sess.run(iterator.initializer)

        while True:
            try:
                label_ = sess.run(label)

                file.write('id')
                file.write('\t')
                file.write('real')
                file.write('\t')
                file.write('predicted')
                file.write('\n')

                for l in label_:
                    predicted = predictions.__next__()

                    # file.write('id')
                    file.write(str(ids))
                    file.write('\t')
                    file.write(str(l))
                    file.write('\t')
                    file.write(str(predicted['ocean']))
                    file.write('\n')

                    ids += 1

                if ids % 500 == 0:
                    print(ids)
            except tf.errors.OutOfRangeError:
                break
    file.close()


if __name__ == '__main__':
    # Execute code only if executing as script
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
