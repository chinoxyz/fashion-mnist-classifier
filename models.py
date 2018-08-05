import tensorflow as tf


def linear_classifier(features_tensor, NUM_CLASSES):
    with tf.variable_scope('rnn') as scope:
        logits = tf.contrib.layers.fully_connected(
            features_tensor,
            NUM_CLASSES,
            activation_fn=None,
        )
        scope.reuse_variables()
        weight = tf.get_variable('fully_connected/weights')

    return logits, [weight]


def get_model(model_name, l2_param, NUM_CLASSES):
    features_tensor = tf.placeholder(tf.float32, [None, 784])
    labels_tensor = tf.placeholder(tf.int32, [None])
    batch_size_tensor = tf.shape(labels_tensor)[0]
    labels_one_hot = tf.one_hot(labels_tensor, NUM_CLASSES)

    if model_name == 'linear':
        logits, weights = linear_classifier(features_tensor, NUM_CLASSES)
    else:
        raise ValueError('No model with name %s', model_name)

    l2_value = 0
    for weight in weights:
        l2_value += tf.reduce_sum(tf.square(weight))
    l2_regularization = l2_param * l2_value / tf.cast(batch_size_tensor, tf.float32)
    loss_tensor = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels_one_hot,
        logits=logits)
    ) + l2_regularization

    prediction_tensor = tf.reshape(tf.argmax(logits, 1), [-1])


    return features_tensor, labels_tensor, loss_tensor, prediction_tensor
