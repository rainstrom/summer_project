import vgg16
import ucf101
import tensorflow as tf
import numpy as np
from load_model import load_from_pickle

learning_rate = 0.01
total_steps = 18000
decay_steps = 6000
decay_factor = 0.1
momentum = 0.9
batch_size = 100 # TODO: 100
num_segments = 1
num_length = 1
root_dir = '/scratch/xiaoyang/UCF101_opt_flows_org2'
train_list = '/home/xiaoyang/action_recognition_exp/dataset_file_examples/train_split1_avi.txt'
test_list = '/home/xiaoyang/action_recognition_exp/dataset_file_examples/val_split1_avi.txt'
test_iter = 10 # *100
test_inteval = 500
save_inteval = 3000
showing_inteval = 20
keep_prob_value_start = 0.8
keep_prob_value_end = 0.1
only_full_test = False
full_test_segments = 25
assert batch_size % full_test_segments == 0


sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=True))
sess.as_default()
# assert tf.get_default_session() == sess

batch_data = tf.placeholder(tf.float32, shape=[batch_size, 224, 224, num_segments*num_length*2], name="data")
batch_label = tf.placeholder(tf.int64, shape=[batch_size], name="label")
keep_prob = tf.placeholder("float")
global_step = tf.Variable(0, name='global_step', trainable=False)

softmax_digits = vgg16.inference(batch_data, keep_prob=keep_prob, train_conv123=True, train_conv45=True, train_fc67=True)
loss = vgg16.loss(softmax_digits, batch_label)
accuracy = vgg16.accuracy(softmax_digits, batch_label)

lr = tf.train.exponential_decay(learning_rate,
                              global_step,
                              decay_steps,
                              decay_factor,
                              staircase=False)
optimizer = tf.train.MomentumOptimizer(lr, momentum)
train_op = optimizer.minimize(loss, global_step=global_step)

saver = tf.train.Saver(tf.all_variables())
sess.run(tf.initialize_all_variables())

if not only_full_test:
    assert not only_full_test
    print("cannot find ckpt checkpoint, loading from tfmodel")
    load_from_pickle("VGG_ILSVRC_16_layers.tfmodel", sess, ignore_missing=True)
    print("loaded from tfmodel")
else:
    ckpt = tf.train.get_checkpoint_state("./weights_flow/")
    if ckpt and ckpt.model_checkpoint_path:
        print("loading from cpkt")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("loaded from cpkt")

if not only_full_test:
    data_reader = ucf101.reader(root_dir, train_list, batch_size, False, 1, 1, "FLOW")
    test_data_reader = ucf101.reader(root_dir, test_list, batch_size, True, 1, 1, "FLOW")
    keep_prob_value = 1.
    for i in range(total_steps):
        print("loading data")
        data, label = data_reader.get()
        print("running")
        (_, loss_value, acc_value, step, lr_value) = sess.run([train_op, loss, accuracy, global_step, lr], feed_dict={batch_data:data, batch_label:label, keep_prob: keep_prob_value})
        progress = min(float(step) / float(total_steps) * 2, 1.0)
        keep_prob_value = progress * keep_prob_value_end + (1 - progress) * keep_prob_value_start
        print("[step %d]: loss, %f; acc, %f; lr, %f; keep, %f" % (step, loss_value, acc_value, lr_value, keep_prob_value))
        if step % test_inteval == 0:# or step == 1:
            all_acc = []
            all_loss = []
            for k in range(test_iter):
                test_data, test_label = test_data_reader.get()
                (loss_value, acc_value) = sess.run([loss, accuracy], feed_dict={batch_data:test_data, batch_label:test_label, keep_prob: 1.0})
                all_acc.append(acc_value)
                all_loss.append(loss_value)
                print("test iter %d: acc, %f; loss, %f" % (k, acc_value, loss_value))
            print("test result: acc, %f; loss, %f" % (acc_value, loss_value))

        if step % save_inteval == 0: #or step == 1:
            save_path = saver.save(sess, "./weights_flow/flow_vgg16_iter%d.ckpt" % (step))
            print("Model saved in file: %s" % save_path)
    print("trainning finished")

print("full testing ... ... ")
full_test_data_reader = ucf101.reader(root_dir, test_list, batch_size, True, 1, 1, "FLOW", True, 25)
full_test_video_num = full_test_data_reader.get_video_num()
print("totally test %d videos" % full_test_video_num)
full_test_all_acc = []
full_test_final_acc_num = 0
for i in range(full_test_video_num * full_test_segments // batch_size):
    step = i + 1
    test_data, test_label = full_test_data_reader.get()
    (loss_value, acc_value, fc8_value) = sess.run([loss, accuracy, softmax_digits], feed_dict={batch_data:test_data, batch_label:test_label, keep_prob: 1.0})
    print("test iter %d: acc, %f; loss, %f" % (step, acc_value, loss_value))
    full_test_all_acc.append(acc_value)
    for k in range(batch_size // full_test_segments):
        fc8_value1 = fc8_value[k*full_test_segments:(k+1)*full_test_segments, :]
        test_label1 = test_label[k*full_test_segments:(k+1)*full_test_segments]
        assert all([test_label1[0]==lb for lb in test_label1])
        test_label1 = test_label1[0]
        fc8_value1_mean = np.mean(fc8_value1, axis=0)
        final_pd = np.argmax(fc8_value1_mean)
        if int(final_pd) == int(test_label1):
            full_test_final_acc_num += 1
    if step % 5 == 0:
        print("step %d: 1 frame acc is %f" % (step, np.mean(full_test_all_acc)))
        print("step %d: %d frame acc is %f" % (step, full_test_segments, float(full_test_final_acc_num) / float(step)))

print("1 frame acc is %f" % np.mean(full_test_all_acc))
print("%d frame acc is %f" % (full_test_segments, float(full_test_final_acc_num) / float(full_test_video_num)))
