def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev_'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        with tf.name_scope('l2norm_'):
            l2norm = tf.sqrt(tf.reduce_sum(tf.square(var)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('l2norm/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)
