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
                variable_exist = False
                try:
                    var = tf.get_variable(param_name)
                    variable_exist = True
                    assign_ops.append(var.assign(data))
                    print("%s/%s added" % (op_name, param_name))
                except ValueError:
                    if variable_exist:
                        # change shape
                        assert op_name in [u"conv1_1"]
                        new_input_channel = int(var.get_shape()[2])
                        new_data = np.zeros([data.shape[0],data.shape[1],new_input_channel,data.shape[3]],np.float32)
                        for i in range(new_input_channel):
                            new_data[:,:,i,:]=data[:,:,i%data.shape[2],:]
                        assign_ops.append(var.assign(new_data))
                        print("** %s/%s added [shape changed]" % (op_name, param_name))
                    else:
                        print("** %s/%s not added [not exist]" % (op_name, param_name))
                    if not ignore_missing:
                        raise
    session.run(assign_ops)
