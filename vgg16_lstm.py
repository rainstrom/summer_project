import tensorflow as tf
import numpy as np
from layers import *
import vgg16

def inference(input, batch_size, num_segments, lstm_keep_prob=0.5, conv_keep_prob=1.0, train_conv123=False, train_conv45=False, train_fc67=False):
    import pdb; pdb.set_trace()
    with tf.variable_scope("conv"):
        fc8 = vgg16.inference(input, conv_keep_prob, train_conv123, train_conv45, train_fc67)
        fc7 = tf.get_default_graph().get_tensor_by_name("conv/fc7/conv/fc7:0")
        fc6 = tf.get_default_graph().get_tensor_by_name("conv/fc6/conv/fc6:0")
        # output is [batch_size*num_segments, 4096]
    with tf.variable_scope("lstm"):
        hidden_size = 256
        lstm_inputs = tf.reshape(fc7, [batch_size, num_segments, 4096])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=lstm_keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)
        _initial_state = cell.zero_state(batch_size, tf.float32)

        outputs = []
        state = self._initial_state
        for time_step in range(num_segments):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(lstm_inputs[:, time_step, :], state)
            outputs.append(cell_output)
        final_state = state
    return cell_output

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
