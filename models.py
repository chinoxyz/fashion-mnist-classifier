import tensorflow as tf

from constants import NUM_CLASSES


def linear_classifier(features_tensor):
    with tf.variable_scope('linear') as scope:
        logits = tf.contrib.layers.fully_connected(
            features_tensor,
            NUM_CLASSES,
            activation_fn=None,
        )
        scope.reuse_variables()
        weight = tf.get_variable('fully_connected/weights')

    return logits, [weight]


def multiple_feed_forward(
        features_tensor
):
    input_ln = tf.contrib.layers.layer_norm(features_tensor)
    with tf.variable_scope('fc1') as scope:
        fc1 = tf.contrib.layers.fully_connected(
            input_ln,
            300,
            activation_fn=tf.nn.elu,
        )
        scope.reuse_variables()
        w1 = tf.get_variable('fully_connected/weights')

    fc1_ln = tf.contrib.layers.layer_norm(fc1)

    with tf.variable_scope('fc2') as scope:
        logits = tf.contrib.layers.fully_connected(
            fc1_ln,
            NUM_CLASSES,
            activation_fn=None,
        )
        scope.reuse_variables()
        w2 = tf.get_variable('fully_connected/weights')

    return logits, [w1, w2]


def get_model(
        model_name, l2_param,
        extra_params={}
):
    features_tensor = tf.placeholder(tf.float32, [None, 784])
    labels_tensor = tf.placeholder(tf.int32, [None])
    batch_size_tensor = tf.shape(labels_tensor)[0]
    labels_one_hot = tf.one_hot(labels_tensor, NUM_CLASSES)

    if model_name == 'linear':
        logits, weights = linear_classifier(
            features_tensor=features_tensor,
            **extra_params
        )
    elif model_name == 'multiple_fc':
        logits, weights = multiple_feed_forward(
            features_tensor=features_tensor,
            **extra_params
        )
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
