import numpy as np
import tensorflow as tf

PADDING_METHOD = 'SAME'

def make_var(name, shape, initializer=tf.random_normal_initializer(stddev=0.01), regularizer=None, trainable=True):
    new_var = tf.get_variable(name, shape, initializer=initializer,
        regularizer=regularizer,
        trainable=trainable)
    tf.add_to_collection(name, new_var)
    return new_var

def conv(input,
         k_h,
         k_w,
         c_o,
         s_h,
         s_w,
         name,
         relu=True,
         padding=PADDING_METHOD,
         group=1,
         biased=True,
         trainable=True):
    # Get the number of channels in the input
    c_i = input.get_shape()[-1]
    # Verify that the grouping parameter is valid
    assert c_i % group == 0
    assert c_o % group == 0
    # Convolution for a given input and kernel
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        kernel = make_var('weights', shape=[k_h, k_w, c_i / group, c_o], trainable=trainable)
        if group == 1:
            # This is the common-case. Convolve the input without any further complications.
            output = convolve(input, kernel)
        else:
            # Split the input into groups and then convolve each of them independently
            input_groups = tf.split(3, group, input)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            output = tf.concat(3, output_groups)
        # Add the biases
        if biased:
            biases = make_var('biases', [c_o], trainable=trainable)
            output = tf.nn.bias_add(output, biases)
        if relu:
            # ReLU non-linearity
            output = tf.nn.relu(output, name=scope.name)
        return output

def relu(input, name):
    return tf.nn.relu(input, name=name)

def max_pool(input, k_h, k_w, s_h, s_w, name, padding=PADDING_METHOD):
    return tf.nn.max_pool(input,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)

def avg_pool(input, k_h, k_w, s_h, s_w, name, padding=PADDING_METHOD):
    return tf.nn.avg_pool(input,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)

# def lrn(input, radius, alpha, beta, name, bias=1.0):
#     return tf.nn.local_response_normalization(input,
#                                               depth_radius=radius,
#                                               alpha=alpha,
#                                               beta=beta,
#                                               bias=bias,
#                                               name=name)

# def concat(inputs, axis, name):
#     return tf.concat(concat_dim=axis, values=inputs, name=name)

# def add(inputs, name):
#     return tf.add_n(inputs, name=name)

def fc(input, num_out, name, relu=True, trainable=True):
    with tf.variable_scope(name) as scope:
        input_shape = input.get_shape()
        if input_shape.ndims == 4:
            # The input is spatial. Vectorize it first.
            dim = 1
            for d in input_shape[1:].as_list():
                dim *= d
            feed_in = tf.reshape(input, [-1, dim])
        else:
            feed_in, dim = (input, input_shape[-1].value)
        weights = make_var('weights', shape=[dim, num_out], trainable=trainable)
        biases = make_var('biases', [num_out], trainable=trainable)
        if relu:
            return tf.nn.relu(tf.nn.xw_plus_b(feed_in, weights, biases), name=scope.name)
        else:
            return tf.nn.xw_plus_b(feed_in, weights, biases, name=scope.name)

def softmax(input, name):
    input_shape = map(lambda v: v.value, input.get_shape())
    if len(input_shape) > 2:
        # For certain models (like NiN), the singleton spatial dimensions
        # need to be explicitly squeezed, since they're not broadcast-able
        # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
        if input_shape[1] == 1 and input_shape[2] == 1:
            input = tf.squeeze(input, squeeze_dims=[1, 2])
        else:
            raise ValueError('Rank 2 tensor input expected for softmax!')
    return tf.nn.softmax(input, name)

# def batch_normalization(input, name, scale_offset=True, relu=False):
#     # NOTE: Currently, only inference is supported
#     with tf.variable_scope(name) as scope:
#         shape = [input.get_shape()[-1]]
#         if scale_offset:
#             scale = make_var('scale', shape=shape)
#             offset = make_var('offset', shape=shape)
#         else:
#             scale, offset = (None, None)
#         output = tf.nn.batch_normalization(
#             input,
#             mean=make_var('mean', shape=shape),
#             variance=make_var('variance', shape=shape),
#             offset=offset,
#             scale=scale,
#             # TODO: This is the default Caffe batch norm eps
#             # Get the actual eps from parameters
#             variance_epsilon=1e-5,
#             name=name)
#         if relu:
#             output = tf.nn.relu(output)
#         return output

def dropout(input, keep_prob, name):
    return tf.nn.dropout(input, keep_prob, name=name)
