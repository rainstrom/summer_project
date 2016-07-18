import tensorflow as tf
import numpy as np
from layers import *
import vgg16
import layers

def inference(input, batch_size, num_segments, lstm_keep_prob=0.5, conv_keep_prob=1.0, train_conv123=False, train_conv45=False, train_fc67=False):
    # input size is [batch_size * num_segments, 224, 224, num_length*3/2]
    with tf.variable_scope("conv"):
        fc8 = vgg16.inference(input, conv_keep_prob, train_conv123, train_conv45, train_fc67, False)
        fc7 = tf.get_default_graph().get_tensor_by_name("conv/fc7/fc7:0")
        fc6 = tf.get_default_graph().get_tensor_by_name("conv/fc6/fc6:0")
        # output is [batch_size*num_segments, 4096]
    with tf.variable_scope("lstm"):
        hidden_size = 256
        lstm_inputs = tf.reshape(fc7, [batch_size, num_segments, 4096])

        stacked_lstm_cell_num = 1
        lstm_cells = []
        for i in range(stacked_lstm_cell_num):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=lstm_keep_prob)
            lstm_cells.append(lstm_cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells, state_is_tuple=True)
        _initial_state = cell.zero_state(batch_size, tf.float32)

        outputs = []
        state = _initial_state
        for time_step in range(num_segments):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(lstm_inputs[:, time_step, :], state)
            outputs.append(cell_output)
        final_state = state
        lstm_params = [var for var in tf.all_variables() if var.name.startswith("lstm")]
        for var in lstm_params:
            tf.add_to_collection("params", var)
    logits = layers.fc(tf.concat(0, outputs, 'concat'), 101, relu=False, name='cls')

    return logits

def loss(logits, labels, num_segments, weight_decay=0.005):
    labels = tf.cast(labels, tf.int64)
    labels = tf.tile(labels, [num_segments])
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


def accuracy(logits, labels, num_segments):
    labels = tf.cast(labels, tf.int64)
    labels = tf.tile(labels, [num_segments])
    correct_prediction = tf.equal(tf.argmax(logits,1), labels)
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
