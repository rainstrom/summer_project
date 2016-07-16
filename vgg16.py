import tensorflow as tf
import numpy as np
from layers import *

def inference(input, keep_prob=1.0, train_conv123=True, train_conv45=True, train_fc67=True):
    output = conv(input, 3, 3, 64, 1, 1, name='conv1_1', trainable=train_conv123)
    output = conv(output, 3, 3, 64, 1, 1, name='conv1_2', trainable=train_conv123)
    output = max_pool(output, 2, 2, 2, 2, name='pool1')
    output = conv(output, 3, 3, 128, 1, 1, name='conv2_1', trainable=train_conv123)
    output = conv(output, 3, 3, 128, 1, 1, name='conv2_2', trainable=train_conv123)
    output = max_pool(output, 2, 2, 2, 2, name='pool2')
    output = conv(output, 3, 3, 256, 1, 1, name='conv3_1', trainable=train_conv123)
    output = conv(output, 3, 3, 256, 1, 1, name='conv3_2', trainable=train_conv123)
    output = conv(output, 3, 3, 256, 1, 1, name='conv3_3', trainable=train_conv123)
    output = max_pool(output, 2, 2, 2, 2, name='pool3')
    output = conv(output, 3, 3, 512, 1, 1, name='conv4_1', trainable=train_conv45)
    output = conv(output, 3, 3, 512, 1, 1, name='conv4_2', trainable=train_conv45)
    output = conv(output, 3, 3, 512, 1, 1, name='conv4_3', trainable=train_conv45)
    output = max_pool(output, 2, 2, 2, 2, name='pool4')
    output = conv(output, 3, 3, 512, 1, 1, name='conv5_1', trainable=train_conv45)
    output = conv(output, 3, 3, 512, 1, 1, name='conv5_2', trainable=train_conv45)
    output = conv(output, 3, 3, 512, 1, 1, name='conv5_3', trainable=train_conv45)
    output = max_pool(output, 2, 2, 2, 2, name='pool5')
    output = fc(output, 4096, name='fc6', trainable=train_fc67)
    output = dropout(output, keep_prob, name='drop6')
    output = fc(output, 4096, name='fc7', trainable=train_fc67)
    output = dropout(output, keep_prob, name='drop7')
    logits = fc(output, 101, relu=False, name='fc8_rgb')
    # output = softmax(output, name='prob'))
    return logits

def loss(logits, labels, weight_decay=0.005):
    labels = tf.cast(labels, tf.int64)
    # cross_entropy_loss
    cross_entropy_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy_loss = tf.reduce_mean(cross_entropy_per_example, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_loss)
    # l2 weight decay
    # weights = tf.get_collection('weights')
    # assert len(weights) > 0
    # l2loss = tf.add_n([tf.nn.l2_loss(weight) for weight in weights], name='l2loss')
    # return cross_entropy_loss + weight_decay * l2loss
    return cross_entropy_loss


def accuracy(logits, labels):
    # import pdb; pdb.set_trace()
    correct_prediction = tf.equal(tf.argmax(logits,1), labels)
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
