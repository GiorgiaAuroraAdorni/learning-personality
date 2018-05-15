#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from model_input import model_fn
from model_input import create_input_fn


def main(argv):

    train_data = "Dataset/TFRecords/train.tfrecords.gz"
    test_data = "Dataset/TFRecords/test.tfrecords.gz"

    session_config = tf.ConfigProto()

    config = tf.estimator.RunConfig(model_dir='Model/003/',
                                    save_summary_steps=50,
                                    session_config=session_config,
                                    # log_step_count_steps=1
                                    )

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       params={},
                                       config=config)

    input_fn = create_input_fn(train_data)

    hooks = [tf.train.ProfilerHook(save_steps=200, output_dir='Timeline/')]  # init_hook]

    idx = 0

    for idx in range(0, 10):
        estimator.train(input_fn=input_fn,
                    hooks=hooks)

        eval_result = estimator.evaluate(input_fn=create_input_fn("Dataset/TFRecords/test.tfrecords.gz"))
        print('\nRoot Mean Square Error: {rmse:0.3f}\n'.format(**eval_result))

        idx += 1

    predictions = estimator.predict(input_fn=create_input_fn(test_data))

    input = create_input_fn(test_data)
    dataset = input()
    iterator = dataset.make_initializable_iterator()
    feature, label = iterator.get_next()

    file = open("Model_Dataset/prediction_model_003.txt", "w")
    ids = 0

    with tf.Session() as sess:
        tf.tables_initializer().run()
        sess.run(iterator.initializer)

        while True:
            try:
                label_ = sess.run(label)

                for l in label_:
                    predicted = predictions.__next__()
                    file.write('id:')
                    file.write(str(ids))
                    file.write('\t')
                    file.write('real: ')
                    file.write(str(l))
                    file.write('\t')
                    file.write('predicted: ')
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
