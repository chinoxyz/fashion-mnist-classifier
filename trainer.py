import os

import tensorflow as tf

from constants import MAX_PIXEL_VALUE
from models import get_model
from preprocess import preprocess_min_max
from util import mnist_reader
from util.sampleset import SampleGenerator, SampleSet


def restore_best_model(session, model_saver):
    model_path = os.path.join('checkpoints', 'model')
    model_saver.restore(session, model_path)


def save_model(session, model_saver):
    model_path = os.path.join('checkpoints', 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_saver.save(session, model_path)


def run(
        training_batch_size,
        validation_batch_size,
        test_batch_size,
        model_params,
        epochs,
        learning_rate,
        training_show_every,
        check_validation_every,
        validation_split,
):
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    X_train = preprocess_min_max(X_train, 0, MAX_PIXEL_VALUE)
    X_test = preprocess_min_max(X_test, 0, MAX_PIXEL_VALUE)

    sampleset = SampleSet(X_train, y_train, validation_split)
    test_sample_gen = SampleGenerator(X_test, y_test)

    with tf.Graph().as_default() as g:
        with tf.Session(graph=g).as_default() as sess:

            features_tensor, labels_tensor, loss_tensor, prediction_tensor = get_model(
                **model_params
            )

            correct_predictions_tensor = tf.reduce_sum(
                tf.cast(
                    tf.equal(
                        tf.cast(prediction_tensor, tf.int32),
                        labels_tensor
                    ),
                    tf.float32
                )
            )

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            minimize_op = optimizer.minimize(loss_tensor)
            model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

            #####
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            global_step = 0
            best_validation_loss = None
            for epoch in range(epochs):
                for x_batch, y_batch in sampleset.get_training_batches(training_batch_size):
                    global_step += 1
                    _, training_loss = sess.run(
                        [minimize_op, loss_tensor],
                        feed_dict={features_tensor: x_batch, labels_tensor: y_batch})

                    if (global_step % training_show_every == 0):

                        print('At Epoch: %d at Step: %d Training Loss is: %f' % (epoch, global_step, training_loss))

                    if (global_step % check_validation_every == 0):
                        validation_loss = 0
                        for x_batch_validation, y_batch_validation in sampleset.get_validation_batches(
                                validation_batch_size):
                            validation_loss_batch = sess.run(
                                loss_tensor,
                                feed_dict={features_tensor: x_batch_validation, labels_tensor: y_batch_validation})
                            validation_loss += validation_loss_batch
                        print(
                            '--At Epoch: %d at Step: %d Validation Loss is: %f' % (epoch, global_step, validation_loss))
                        if (best_validation_loss is None or validation_loss < best_validation_loss):
                            print('Best model found!')
                            save_model(session=sess, model_saver=model_saver)
                            best_validation_loss = validation_loss

            restore_best_model(sess, model_saver)

            correct_predictions = 0
            for x_test_batch, y_test_batch in test_sample_gen.get_batches(test_batch_size):
                correct_predictions += sess.run(
                    correct_predictions_tensor,
                    feed_dict={features_tensor: x_test_batch, labels_tensor: y_test_batch})
            accuracy = (correct_predictions / test_sample_gen.get_num_samples())
            print('Accuracy: %f' % accuracy)
