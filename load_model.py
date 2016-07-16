import numpy as np
import tensorflow as tf

def load_from_pickle(data_path, session, ignore_missing=False):
    '''Load network weights.
    data_path: The path to the numpy-serialized network weights
    session: The current TensorFlow session
    ignore_missing: If true, serialized weights for missing layers are ignored.
    '''
    data_dict = np.load(data_path).item()
    assign_ops = []
    for op_name in data_dict:
        with tf.variable_scope(op_name, reuse=True):
            for param_name, data in data_dict[op_name].iteritems():
                try:
                    var = tf.get_variable(param_name)
                    assign_ops.append(var.assign(data))
                    print("%s/%s added" % (op_name, param_name))
                except ValueError:
                    print("%s/%s not added" % (op_name, param_name))
                    if not ignore_missing:
                        raise
    session.run(assign_ops)
