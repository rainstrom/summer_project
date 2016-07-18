import tensorflow as tf
import numpy as np
from layers import *
import vgg16

def inference(input, lstm_keep_prob=0.5, conv_keep_prob=1.0, train_conv123=False, train_conv45=False, train_fc67=False):
    with tf.variable_scope("conv"):
        fc7 = vgg16.inference(input, conv_keep_prob, train_conv123, train_conv45, train_fc67)
        fc6 = tf.get_default_graph().get_tensor_by_name("fc6")
        fc5 = tf.get_default_graph().get_tensor_by_name("fc5")
    with tf.variable_scope("lstm"):
        lstm_state_size = 256
        batch_size = fc6.get_shape()[0].value
        lstm = rnn_cell.BasicLSTMCell(lstm_size)
        state = tf.zeros([batch_size, lstm_state_size])
        # Initial state of the LSTM memory.
        state = tf.zeros([batch_size, lstm.state_size])

        loss = 0.0
        for current_batch_of_words in words_in_dataset:
            # The value of state is updated after processing each batch of words.
            output, state = lstm(current_batch_of_words, state)

            # The LSTM output can be used to make next word predictions
            logits = tf.matmul(output, softmax_w) + softmax_b
            probabilities = tf.nn.softmax(logits)
            loss += loss_function(probabilities, target_words)
    return logits

def loss(logits, labels, weight_decay=0.001):
    labels = tf.cast(labels, tf.int64)
    # cross_entropy_loss
    cross_entropy_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy_loss = tf.reduce_mean(cross_entropy_per_example, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_loss)
    # l2 weight decay
    weights = tf.get_collection('weights')
    assert len(weights) > 0
    l2loss = tf.add_n([tf.nn.l2_loss(weight) for weight in weights], name='l2loss')
    return cross_entropy_loss, cross_entropy_loss + weight_decay * l2loss


def accuracy(logits, labels):
    # import pdb; pdb.set_trace()
    correct_prediction = tf.equal(tf.argmax(logits,1), labels)
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
