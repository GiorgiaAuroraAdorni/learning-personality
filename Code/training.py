#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from model_input import model_fn
from model_input import create_input_fn


def main(argv):

    train_data = "Dataset/TFRecords/train.tfrecords.gz"
    test_data = "Dataset/TFRecords/test.tfrecords.gz"

    session_config = tf.ConfigProto()

    config = tf.estimator.RunConfig(model_dir='Model/005/',
                                    save_summary_steps=50,
                                    session_config=session_config,
                                    # log_step_count_steps=1
                                    )

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       params={},
                                       config=config)

    input_fn = create_input_fn(train_data)

    hooks = [tf.train.ProfilerHook(save_steps=200, output_dir='Timeline/')]  # init_hook]

    #idx = 0

    #for idx in range(0, 6):
    # estimator.train(input_fn=input_fn, hooks=hooks)

    eval_result = estimator.evaluate(input_fn=create_input_fn("Dataset/TFRecords/test.tfrecords.gz"))
    print('\nRoot Mean Square Error: {rmse:0.3f}\n'.format(**eval_result))

    #idx += 1

    predictions = estimator.predict(input_fn=create_input_fn(test_data))

    input = create_input_fn(test_data)
    dataset = input()
    iterator = dataset.make_initializable_iterator()
    feature, label = iterator.get_next()

    file = open("Model_Dataset/Mean/prediction_model_003.txt", "w")
    ids = 0

    # OCEAN1 = [-0.012779600918293, -0.015215029940009117, 0.009056995622813702, 0.012603215873241425,
    #           0.11371849477291107]

    # OCEAN2 = [-0.026229945942759514, -0.017903760075569153, 0.018462106585502625, 0.008296631276607513,
    #           0.13911594450473785]

    OCEAN3 = [-0.1894543617963791, 0.21709837019443512, 0.1195567324757576, -0.29764291644096375, -0.29123809933662415]

    with tf.Session() as sess:
        tf.tables_initializer().run()
        sess.run(iterator.initializer)

        while True:
            try:
                label_ = sess.run(label)

                file.write('id')
                file.write('\t')
                file.write('deviation')
                file.write('\n')

                for l in label_:
                    predicted = predictions.__next__()

                    # dev1 = predicted['ocean'] - OCEAN1;
                    # dev2 = predicted['ocean'] - OCEAN2;
                    dev3 = predicted['ocean'] - OCEAN3;

                    # file.write('id')
                    file.write(str(ids))
                    file.write('\t')
                    # file.write('real: ')
                    # file.write(str(l))
                    # file.write('\t')
                    # file.write('predicted: ')
                    # file.write(str(predicted['ocean']))
                    # file.write('\t')
                    # file.write('mean: ')
                    # file.write('[-0.012779600918293, -0.015215029940009117, 0.009056995622813702, 0.012603215873241425, 0.11371849477291107]')
                    # file.write('\t')
                    # file.write('deviation: ')
                    file.write(str(dev3))
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
